import logging
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Protocol

from app.sources.citations import citation_ids, select_cited_sources
from app.sources.models import PublicSource, Source
from app.streaming.events import StreamEvent


logger = logging.getLogger(__name__)

RAG_SEARCH_MESSAGE = "Buscando na base de conhecimento..."
QUERY_REWRITE_MESSAGE = "Reformulando a busca..."
WEB_SEARCH_MESSAGE = "Buscando fontes externas..."
GENERATING_MESSAGE = "Gerando a resposta..."
PUBLIC_ERROR_MESSAGE = "Não foi possível concluir a resposta."
ANSWER_CHUNK_PATTERN = re.compile(r"\S+\s*")


class StreamableRagGraph(Protocol):
    def stream(
        self,
        input: dict[str, str],
        *,
        config: dict[str, dict[str, str]],
        stream_mode: list[str],
    ) -> Iterable[tuple[str, object]]: ...


class LangGraphStreamAdapter:
    def __init__(
        self,
        *,
        graph_factory: Callable[[], StreamableRagGraph],
        max_rag_attempts: int,
    ):
        if max_rag_attempts < 1:
            raise ValueError("max_rag_attempts must be at least 1")
        self._graph_factory = graph_factory
        self._max_rag_attempts = max_rag_attempts

    def stream(
        self,
        *,
        conversation_id: str,
        message: str,
    ) -> Iterator[StreamEvent]:
        yield _status("rag_search", RAG_SEARCH_MESSAGE)

        retrieval_attempts = 0
        final_result: tuple[str, str, list[PublicSource]] | None = None
        last_status = "rag_search"

        try:
            graph = self._graph_factory()
            internal_stream = graph.stream(
                {"question": message},
                config={"configurable": {"thread_id": conversation_id}},
                stream_mode=["updates"],
            )
            for mode, data in internal_stream:
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
                            elif last_status != "web_search":
                                yield _status("web_search", WEB_SEARCH_MESSAGE)
                                last_status = "web_search"

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
            if last_status != "generating":
                yield _status("generating", GENERATING_MESSAGE)
            yield StreamEvent("sources", {"sources": sources})
            for token in ANSWER_CHUNK_PATTERN.findall(answer):
                yield StreamEvent("token", {"text": token})

            yield StreamEvent("done", {})
        except Exception as error:
            logger.error("RAG stream failed: %s", type(error).__name__)
            yield StreamEvent(
                "error",
                {
                    "code": "stream_failed",
                    "message": PUBLIC_ERROR_MESSAGE,
                },
            )


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
