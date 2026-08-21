import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any, Protocol

from langchain_core.messages import AIMessage, BaseMessage

from app.streaming.events import StreamEvent


logger = logging.getLogger(__name__)

RAG_SEARCH_MESSAGE = "Buscando na base de conhecimento..."
QUERY_REWRITE_MESSAGE = "Reformulando a busca..."
WEB_SEARCH_MESSAGE = "Buscando fontes externas..."
GENERATING_MESSAGE = "Gerando a resposta..."
PUBLIC_ERROR_MESSAGE = "Não foi possível concluir a resposta."


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
        final_result: tuple[str, str] | None = None
        emitted_token = False
        emitted_sources = False
        last_status = "rag_search"

        try:
            graph = self._graph_factory()
            internal_stream = graph.stream(
                {"question": message},
                config={"configurable": {"thread_id": conversation_id}},
                stream_mode=["updates", "messages"],
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

                elif mode == "messages":
                    message_chunk, metadata = _message_event(data)
                    if metadata.get("langgraph_node") != "generate_answer":
                        continue
                    if isinstance(message_chunk, BaseMessage) and not isinstance(
                        message_chunk,
                        AIMessage,
                    ):
                        continue
                    text = _message_text(message_chunk)
                    if not text:
                        continue
                    if last_status != "generating":
                        yield _status("generating", GENERATING_MESSAGE)
                        last_status = "generating"
                    if not emitted_sources:
                        yield StreamEvent("sources", {"sources": []})
                        emitted_sources = True
                    yield StreamEvent("token", {"text": text})
                    emitted_token = True

            if final_result is None:
                raise RuntimeError("graph stream ended without a terminal result")

            answer, _source = final_result
            if not emitted_token:
                if last_status != "generating":
                    yield _status("generating", GENERATING_MESSAGE)
                if not emitted_sources:
                    yield StreamEvent("sources", {"sources": []})
                yield StreamEvent("token", {"text": answer})

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


def _message_event(data: object) -> tuple[object, Mapping[str, Any]]:
    if not isinstance(data, tuple) or len(data) != 2:
        return data, {}
    message, metadata = data
    if not isinstance(metadata, Mapping):
        return message, {}
    return message, metadata


def _message_text(message: object) -> str:
    content = (
        message
        if isinstance(message, str)
        else getattr(message, "content", None)
    )
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    text_blocks: list[str] = []
    for block in content:
        if isinstance(block, str):
            text_blocks.append(block)
        elif isinstance(block, Mapping) and isinstance(block.get("text"), str):
            text_blocks.append(block["text"])
    return "".join(text_blocks)


def _terminal_result(update: Mapping[str, object]) -> tuple[str, str]:
    answer = update.get("answer")
    source = update.get("source")
    if not isinstance(answer, str) or not answer:
        raise RuntimeError("terminal graph update does not contain an answer")
    if not isinstance(source, str) or source not in {"rag", "web", "none"}:
        raise RuntimeError("terminal graph update contains an invalid source")
    return answer, source
