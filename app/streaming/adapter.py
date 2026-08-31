import logging
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from time import monotonic
from typing import Any, Protocol

from app.multimodal.analyzer import ImageAnalyzer, image_analysis_document
from app.observability.tracing import NullTracing, Tracing
from app.multimodal.models import ImageInput
from app.sources.citations import citation_ids, select_cited_sources
from app.sources.models import PublicSource, Source
from app.streaming.events import StreamEvent
from app.visual_catalog.resolver import EquipmentResolver
from app.visual_catalog.retrieval import (
    VisualCatalogRetriever,
    visual_matches_context,
)


logger = logging.getLogger(__name__)

RAG_SEARCH_MESSAGE = "Buscando na base de conhecimento..."
TRACE_TAGS = ("agentic-rag", "chat-stream")
IMAGE_ANALYSIS_MESSAGE = "Analisando a imagem enviada..."
VISUAL_SEARCH_MESSAGE = "Comparando com o catálogo visual..."
QUERY_REWRITE_MESSAGE = "Reformulando a busca..."
WEB_SEARCH_MESSAGE = "Buscando fontes externas..."
GENERATING_MESSAGE = "Gerando a resposta..."
PUBLIC_ERROR_MESSAGE = "Não foi possível concluir a resposta."
PUBLIC_TIMEOUT_MESSAGE = "A resposta excedeu o tempo máximo de processamento."
ANSWER_CHUNK_PATTERN = re.compile(r"\S+\s*")


class StreamableRagGraph(Protocol):
    def stream(
        self,
        input: dict[str, object],
        *,
        config: dict[str, dict[str, str]],
        stream_mode: list[str],
    ) -> Iterable[tuple[str, object]]: ...


class LangGraphStreamAdapter:
    def __init__(
        self,
        *,
        graph_factory: Callable[[], StreamableRagGraph],
        image_analyzer_factory: Callable[[], ImageAnalyzer] | None = None,
        visual_catalog_retriever_factory: (
            Callable[[], VisualCatalogRetriever] | None
        ) = None,
        equipment_resolver: EquipmentResolver | None = None,
        tracing: Tracing | None = None,
        max_rag_attempts: int,
        web_search_enabled: bool = True,
        max_duration_seconds: float = 120.0,
        clock: Callable[[], float] = monotonic,
    ):
        if max_rag_attempts < 1:
            raise ValueError("max_rag_attempts must be at least 1")
        if max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive")
        self._graph_factory = graph_factory
        self._image_analyzer_factory = image_analyzer_factory
        self._visual_catalog_retriever_factory = (
            visual_catalog_retriever_factory
        )
        self._equipment_resolver = equipment_resolver
        self._tracing = tracing or NullTracing()
        self._max_rag_attempts = max_rag_attempts
        self._web_search_enabled = web_search_enabled
        self._max_duration_seconds = max_duration_seconds
        self._clock = clock

    def stream(
        self,
        *,
        conversation_id: str,
        message: str,
        thread_id: str | None = None,
        retrieval_filters: dict[str, Any] | None = None,
        images: list[ImageInput] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[StreamEvent]:
        # The vision call and the graph run are separate LLM invocations; one
        # root observation keeps them in the same trace instead of two
        # unrelated ones for a single question.
        with self._tracing.conversation(
            session_id=thread_id or conversation_id,
            tags=TRACE_TAGS,
        ):
            yield from self._stream(
                conversation_id=conversation_id,
                message=message,
                thread_id=thread_id,
                retrieval_filters=retrieval_filters,
                images=images,
                is_cancelled=is_cancelled,
            )

    def _stream(
        self,
        *,
        conversation_id: str,
        message: str,
        thread_id: str | None,
        retrieval_filters: dict[str, Any] | None,
        images: list[ImageInput] | None,
        is_cancelled: Callable[[], bool] | None,
    ) -> Iterator[StreamEvent]:
        started_at = self._clock()
        cancelled = is_cancelled or (lambda: False)
        pending_images = list(images or [])
        yield _status(
            "image_analysis" if pending_images else "rag_search",
            IMAGE_ANALYSIS_MESSAGE if pending_images else RAG_SEARCH_MESSAGE,
        )

        retrieval_attempts = 0
        final_result: tuple[str, str, list[PublicSource]] | None = None
        last_status = "image_analysis" if pending_images else "rag_search"

        internal_stream: Iterable[tuple[str, object]] | None = None
        try:
            image_context = ""
            image_document = None
            image_reference_documents = []
            resolved_equipment_ids: list[str] = []
            if pending_images:
                if self._image_analyzer_factory is None:
                    raise RuntimeError("multimodal analyzer is not configured")
                analysis = self._image_analyzer_factory().analyze(
                    message,
                    pending_images,
                )
                image_context = analysis.context_text()
                image_document = image_analysis_document(
                    analysis,
                    pending_images,
                )
                if self._visual_catalog_retriever_factory is not None:
                    yield _status("visual_search", VISUAL_SEARCH_MESSAGE)
                    last_status = "visual_search"
                    image_reference_documents = (
                        self._visual_catalog_retriever_factory().retrieve(
                            pending_images,
                            filters=retrieval_filters,
                        )
                    )
                    if image_reference_documents:
                        image_context = (
                            f"{image_context}\n\n"
                            "Candidatos recuperados do catálogo visual "
                            "proprietário (menor distância é melhor):\n"
                            f"{visual_matches_context(image_reference_documents)}"
                        )
                    if self._equipment_resolver is not None:
                        # Candidates stay as citable evidence either way; only a
                        # resolved equipment is allowed to narrow the manuals.
                        resolution = self._equipment_resolver.resolve(
                            [*analysis.identifiers, *analysis.visible_text],
                            image_reference_documents,
                        )
                        resolved_equipment_ids = resolution.equipment_ids
                        if resolution.candidates:
                            image_context = (
                                f"{image_context}\n\n"
                                "Resolução de equipamento (confronte com as "
                                "demais evidências antes de afirmar):\n"
                                f"{resolution.context_text()}"
                            )
                pending_images.clear()
                images = None
                if cancelled():
                    return
                if self._clock() - started_at > self._max_duration_seconds:
                    raise StreamDeadlineExceeded
                yield _status("rag_search", RAG_SEARCH_MESSAGE)
                last_status = "rag_search"

            graph = self._graph_factory()
            graph_input: dict[str, object] = {
                "question": message,
                "image_context": image_context,
                "image_document": image_document,
                "image_reference_documents": image_reference_documents,
                "resolved_equipment_ids": resolved_equipment_ids,
            }
            if retrieval_filters:
                graph_input["retrieval_filters"] = retrieval_filters
            internal_stream = graph.stream(
                graph_input,
                config={
                    "configurable": {
                        "thread_id": thread_id or conversation_id,
                    },
                    "callbacks": self._tracing.callbacks(),
                },
                stream_mode=["updates"],
            )
            for mode, data in internal_stream:
                if cancelled():
                    return
                if self._clock() - started_at > self._max_duration_seconds:
                    raise StreamDeadlineExceeded
                if mode == "updates" and isinstance(data, Mapping):
                    for node_name, update in data.items():
                        if not isinstance(update, Mapping):
                            continue

                        if node_name == "retrieve_rag":
                            attempts = update.get("retrieval_attempts")
                            if isinstance(attempts, int):
                                retrieval_attempts = attempts

                        elif node_name == "grade_rag":
                            if update.get("rag_relevant") is True:
                                if last_status != "generating":
                                    yield _status("generating", GENERATING_MESSAGE)
                                    last_status = "generating"
                            elif retrieval_attempts < self._max_rag_attempts:
                                yield _status(
                                    "query_rewrite",
                                    QUERY_REWRITE_MESSAGE,
                                )
                                last_status = "query_rewrite"
                            elif (
                                self._web_search_enabled
                                and last_status != "web_search"
                            ):
                                yield _status("web_search", WEB_SEARCH_MESSAGE)
                                last_status = "web_search"
                            elif last_status != "generating":
                                yield _status("generating", GENERATING_MESSAGE)
                                last_status = "generating"

                        elif node_name == "rewrite_query":
                            yield _status("rag_search", RAG_SEARCH_MESSAGE)
                            last_status = "rag_search"

                        elif node_name == "grade_web":
                            if last_status != "generating":
                                yield _status("generating", GENERATING_MESSAGE)
                                last_status = "generating"

                        elif node_name in {"generate_answer", "no_answer"}:
                            final_result = _terminal_result(update)

            if final_result is None:
                raise RuntimeError("graph stream ended without a terminal result")

            answer, _source, sources = final_result
            if cancelled():
                return
            if last_status != "generating":
                yield _status("generating", GENERATING_MESSAGE)
            yield StreamEvent("sources", {"sources": sources})
            for token in ANSWER_CHUNK_PATTERN.findall(answer):
                if cancelled():
                    return
                if self._clock() - started_at > self._max_duration_seconds:
                    raise StreamDeadlineExceeded
                yield StreamEvent("token", {"text": token})

            yield StreamEvent("done", {})
        except StreamDeadlineExceeded:
            logger.warning("RAG stream deadline exceeded")
            yield StreamEvent(
                "error",
                {
                    "code": "stream_timeout",
                    "message": PUBLIC_TIMEOUT_MESSAGE,
                },
            )
        except Exception as error:
            logger.error("RAG stream failed: %s", type(error).__name__)
            yield StreamEvent(
                "error",
                {
                    "code": "stream_failed",
                    "message": PUBLIC_ERROR_MESSAGE,
                },
            )
        finally:
            close = getattr(internal_stream, "close", None)
            if callable(close):
                close()


class StreamDeadlineExceeded(TimeoutError):
    pass


def _status(stage: str, message: str) -> StreamEvent:
    return StreamEvent(
        "status",
        {
            "stage": stage,
            "message": message,
        },
    )


def _terminal_result(
    update: Mapping[str, object],
) -> tuple[str, str, list[PublicSource]]:
    answer = update.get("answer")
    source = update.get("source")
    raw_sources = update.get("sources")
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("terminal graph update does not contain an answer")
    if not isinstance(source, str) or source not in {"rag", "web", "none"}:
        raise RuntimeError("terminal graph update contains an invalid source")
    if not isinstance(raw_sources, list):
        raise RuntimeError("terminal graph update does not contain sources")

    try:
        normalized_sources = [
            Source.from_public(raw_source) for raw_source in raw_sources
        ]
        source_ids = [item.citation_id for item in normalized_sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source IDs must be unique")
        cited_sources = select_cited_sources(answer, normalized_sources)
        if {item.citation_id for item in cited_sources} != set(source_ids):
            raise ValueError("terminal sources include uncited entries")
        if source == "none":
            if normalized_sources or citation_ids(answer):
                raise ValueError("no-answer result must not contain citations")
        elif not normalized_sources:
            raise ValueError("grounded answer must contain sources")
        elif any(item.source_type != source for item in normalized_sources):
            raise ValueError("terminal source type does not match answer source")
    except ValueError as error:
        raise RuntimeError("terminal sources are inconsistent") from error

    return answer.strip(), source, [item.to_public() for item in normalized_sources]
