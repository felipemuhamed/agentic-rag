from dataclasses import dataclass

from app.graph.builder import build_rag_graph
from app.knowledge.retrieval.models import (
    RetrievalDiagnostics,
    RetrievalResult,
    RetrievedDocument,
)
from app.llm.rag_grader import RagGrade
from app.llm.web_grader import WebGrade
from app.sources.models import PublicSource, Source
from app.streaming.adapter import (
    GENERATING_MESSAGE,
    PUBLIC_ERROR_MESSAGE,
    WEB_SEARCH_MESSAGE,
    LangGraphStreamAdapter,
)
from app.websearch.models import WebSearchResult


@dataclass
class FakeMessage:
    content: object


RAG_SOURCE: PublicSource = {
    "id": 1,
    "citation": "[1]",
    "type": "rag",
    "title": "Guia OAuth",
    "source": "oauth.md",
    "chunkId": "chunk-1",
    "url": None,
    "page": 3,
    "section": "Tokens",
}

WEB_SOURCE: PublicSource = {
    "id": 1,
    "citation": "[1]",
    "type": "web",
    "title": "Fonte externa",
    "source": "https://example.com/oauth",
    "chunkId": None,
    "url": "https://example.com/oauth",
    "page": None,
    "section": None,
}


class FakeGraph:
    def __init__(self, events: list[object]):
        self.events = events
        self.inputs: list[
            tuple[dict[str, str], dict[str, dict[str, str]], list[str]]
        ] = []

    def stream(
        self,
        input: dict[str, str],
        *,
        config: dict[str, dict[str, str]],
        stream_mode: list[str],
    ):
        self.inputs.append((input, config, stream_mode))
        for event in self.events:
            if isinstance(event, Exception):
                raise event
            yield event


def event_names(events) -> list[str]:
    return [event.event for event in events]


def test_adapter_streams_only_final_generation_tokens() -> None:
    graph = FakeGraph(
        [
            (
                "messages",
                (
                    FakeMessage("SEGREDO DO GRADER"),
                    {"langgraph_node": "grade_rag"},
                ),
            ),
            ("updates", {"retrieve_rag": {"retrieval_attempts": 1}}),
            (
                "messages",
                (
                    FakeMessage("SEGREDO DO REWRITE"),
                    {"langgraph_node": "rewrite_query"},
                ),
            ),
            ("updates", {"grade_rag": {"rag_relevant": True}}),
            (
                "messages",
                (FakeMessage("Resposta "), {"langgraph_node": "generate_answer"}),
            ),
            (
                "messages",
                (FakeMessage("final."), {"langgraph_node": "generate_answer"}),
            ),
            (
                "updates",
                {
                    "generate_answer": {
                        "answer": "Resposta final [1].",
                        "source": "rag",
                        "sources": [RAG_SOURCE],
                    }
                },
            ),
        ]
    )
    adapter = LangGraphStreamAdapter(
        graph_factory=lambda: graph,
        max_rag_attempts=2,
    )

    events = list(
        adapter.stream(
            conversation_id="conversation-1",
            message="Pergunta",
        )
    )

    assert event_names(events) == [
        "status",
        "status",
        "sources",
        "token",
        "token",
        "token",
        "done",
    ]
    assert events[0].data["stage"] == "rag_search"
    assert events[1].data == {
        "stage": "generating",
        "message": GENERATING_MESSAGE,
    }
    assert events[2].data == {"sources": [RAG_SOURCE]}
    assert [event.data["text"] for event in events if event.event == "token"] == [
        "Resposta ",
        "final ",
        "[1].",
    ]
    assert "SEGREDO" not in "".join(event.to_sse() for event in events)
    assert events[-1].data == {}
    assert graph.inputs == [
        (
            {"question": "Pergunta"},
            {"configurable": {"thread_id": "conversation-1"}},
            ["updates"],
        )
    ]


def test_adapter_exposes_rewrite_status_without_rewrite_tokens() -> None:
    graph = FakeGraph(
        [
            ("updates", {"retrieve_rag": {"retrieval_attempts": 1}}),
            ("updates", {"grade_rag": {"rag_relevant": False}}),
            (
                "messages",
                (
                    FakeMessage("consulta interna"),
                    {"langgraph_node": "rewrite_query"},
                ),
            ),
            ("updates", {"rewrite_query": {"search_query": "nova consulta"}}),
            ("updates", {"retrieve_rag": {"retrieval_attempts": 2}}),
            ("updates", {"grade_rag": {"rag_relevant": True}}),
            (
                "updates",
                {
                    "generate_answer": {
                        "answer": "Resposta após rewrite [1].",
                        "source": "rag",
                        "sources": [RAG_SOURCE],
                    }
                },
            ),
        ]
    )
    adapter = LangGraphStreamAdapter(
        graph_factory=lambda: graph,
        max_rag_attempts=2,
    )

    events = list(adapter.stream(conversation_id="c-1", message="Pergunta"))

    statuses = [event.data["stage"] for event in events if event.event == "status"]
    assert statuses == [
        "rag_search",
        "query_rewrite",
        "rag_search",
        "generating",
    ]
    assert "".join(
        event.data["text"] for event in events if event.event == "token"
    ) == "Resposta após rewrite [1]."
    assert events[-1].event == "done"


def test_no_answer_is_sent_as_final_token_before_done() -> None:
    graph = FakeGraph(
        [
            ("updates", {"retrieve_rag": {"retrieval_attempts": 1}}),
            ("updates", {"grade_rag": {"rag_relevant": False}}),
            ("updates", {"search_web": {"web_results": []}}),
            (
                "messages",
                (
                    FakeMessage("SEGREDO DO WEB GRADER"),
                    {"langgraph_node": "grade_web"},
                ),
            ),
            ("updates", {"grade_web": {"web_relevant": False}}),
            (
                "updates",
                {
                    "no_answer": {
                        "answer": "Informação insuficiente.",
                        "source": "none",
                        "sources": [],
                    }
                },
            ),
        ]
    )
    adapter = LangGraphStreamAdapter(
        graph_factory=lambda: graph,
        max_rag_attempts=1,
    )

    events = list(adapter.stream(conversation_id="c-1", message="Pergunta"))

    assert event_names(events) == [
        "status",
        "status",
        "status",
        "sources",
        "token",
        "token",
        "done",
    ]
    statuses = [event.data for event in events if event.event == "status"]
    assert statuses[1] == {
        "stage": "web_search",
        "message": WEB_SEARCH_MESSAGE,
    }
    assert statuses[2] == {
        "stage": "generating",
        "message": GENERATING_MESSAGE,
    }
    assert "SEGREDO" not in "".join(event.to_sse() for event in events)
    assert "".join(
        event.data["text"] for event in events if event.event == "token"
    ) == "Informação insuficiente."
    assert next(event for event in events if event.event == "sources").data == {
        "sources": []
    }


def test_stream_failure_ends_with_safe_error_event() -> None:
    graph = FakeGraph(
        [
            ("updates", {"retrieve_rag": {"retrieval_attempts": 1}}),
            RuntimeError("provider secret must not leak"),
        ]
    )
    adapter = LangGraphStreamAdapter(
        graph_factory=lambda: graph,
        max_rag_attempts=2,
    )

    events = list(adapter.stream(conversation_id="c-1", message="Pergunta"))

    assert events[-1].event == "error"
    assert events[-1].data == {
        "code": "stream_failed",
        "message": PUBLIC_ERROR_MESSAGE,
    }
    assert all(event.event != "done" for event in events)
    assert "secret" not in events[-1].to_sse()


class StaticRetriever:
    def retrieve(self, query: str) -> RetrievalResult:
        document = RetrievedDocument(
            chunk_id="chunk-1",
            content="Contexto suficiente.",
            metadata={"source": "guia.md"},
            retrieval_score=0.8,
            rerank_score=0.9,
        )
        return RetrievalResult(
            documents=[document],
            diagnostics=RetrievalDiagnostics(1, 1, 1, 1),
        )


class RelevantGrader:
    def grade(self, question: str, documents: list[RetrievedDocument]) -> RagGrade:
        return RagGrade(relevant=True, confidence=1.0, reason="Suficiente.")


class IdentityQueryPreparer:
    def prepare(self, question: str, history: list[object]) -> str:
        return question


class RewriterThatMustNotRun:
    def rewrite(self, question: str, search_query: str, failure_reason: str) -> str:
        raise AssertionError("rewriter must not run")


class WebFallbackThatMustNotRun:
    def search(self, query: str) -> list[object]:
        raise AssertionError("web fallback must not run")


class WebGraderThatMustNotRun:
    def grade(self, question: str, results: list[object]) -> object:
        raise AssertionError("web grader must not run")


class StaticAnswerGenerator:
    def generate(
        self,
        question: str,
        documents: list[object],
        sources: list[Source],
    ) -> str:
        return "Resposta do grafo compilado [1]."


def test_adapter_accepts_current_compiled_langgraph_stream_shape() -> None:
    graph = build_rag_graph(
        retriever=StaticRetriever(),
        grader=RelevantGrader(),
        query_preparer=IdentityQueryPreparer(),
        query_rewriter=RewriterThatMustNotRun(),
        web_search=WebFallbackThatMustNotRun(),
        web_grader=WebGraderThatMustNotRun(),
        answer_generator=StaticAnswerGenerator(),
        max_rag_attempts=2,
    )
    adapter = LangGraphStreamAdapter(
        graph_factory=lambda: graph,
        max_rag_attempts=2,
    )

    events = list(adapter.stream(conversation_id="c-1", message="Pergunta"))

    assert "".join(
        event.data["text"] for event in events if event.event == "token"
    ) == "Resposta do grafo compilado [1]."
    sources_event = next(event for event in events if event.event == "sources")
    assert sources_event.data["sources"][0]["chunkId"] == "chunk-1"
    assert events[-1].event == "done"


class EmptyRetriever:
    def retrieve(self, query: str) -> RetrievalResult:
        return RetrievalResult(
            documents=[],
            diagnostics=RetrievalDiagnostics(0, 0, 0, 0),
        )


class InsufficientRagGrader:
    def grade(self, question: str, documents: list[object]) -> RagGrade:
        return RagGrade(relevant=False, confidence=1.0, reason="Insuficiente.")


class StaticWebSearch:
    def search(self, query: str) -> list[WebSearchResult]:
        return [
            WebSearchResult(
                title="Fonte externa",
                url="https://example.com/oauth",
                content="Evidência externa suficiente.",
                score=0.9,
                provider="test",
            )
        ]


class RelevantWebGrader:
    def grade(
        self,
        question: str,
        results: list[WebSearchResult],
    ) -> WebGrade:
        return WebGrade(relevant=True, confidence=1.0, reason="Suficiente.")


def test_compiled_web_path_has_status_and_only_final_answer_token() -> None:
    graph = build_rag_graph(
        retriever=EmptyRetriever(),
        grader=InsufficientRagGrader(),
        query_preparer=IdentityQueryPreparer(),
        query_rewriter=RewriterThatMustNotRun(),
        web_search=StaticWebSearch(),
        web_grader=RelevantWebGrader(),
        answer_generator=StaticAnswerGenerator(),
        max_rag_attempts=1,
    )
    adapter = LangGraphStreamAdapter(
        graph_factory=lambda: graph,
        max_rag_attempts=1,
    )

    events = list(adapter.stream(conversation_id="c-web", message="Pergunta"))

    assert [event.data["stage"] for event in events if event.event == "status"] == [
        "rag_search",
        "web_search",
        "generating",
    ]
    assert "".join(
        event.data["text"] for event in events if event.event == "token"
    ) == "Resposta do grafo compilado [1]."
    sources_event = next(event for event in events if event.event == "sources")
    assert sources_event.data == {"sources": [WEB_SOURCE]}
    assert events[-1].event == "done"


def test_invented_citation_fails_before_sources_or_tokens_are_emitted() -> None:
    graph = FakeGraph(
        [
            ("updates", {"retrieve_rag": {"retrieval_attempts": 1}}),
            ("updates", {"grade_rag": {"rag_relevant": True}}),
            (
                "updates",
                {
                    "generate_answer": {
                        "answer": "Referência inventada [2].",
                        "source": "rag",
                        "sources": [RAG_SOURCE],
                    }
                },
            ),
        ]
    )
    adapter = LangGraphStreamAdapter(
        graph_factory=lambda: graph,
        max_rag_attempts=1,
    )

    events = list(adapter.stream(conversation_id="c-invalid", message="Pergunta"))

    assert events[-1].event == "error"
    assert all(event.event not in {"sources", "token", "done"} for event in events)
