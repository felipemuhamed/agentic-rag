from dataclasses import dataclass

from app.graph.builder import build_rag_graph
from app.knowledge.retrieval.models import (
    RetrievalDiagnostics,
    RetrievalResult,
    RetrievedDocument,
)
from app.llm.rag_grader import RagGrade
from app.llm.web_grader import WebGrade
from app.multimodal.analyzer import ImageAnalysis
from app.multimodal.models import ImageInput
from app.sources.models import PublicSource, Source
from app.streaming.adapter import (
    GENERATING_MESSAGE,
    IMAGE_ANALYSIS_MESSAGE,
    PUBLIC_ERROR_MESSAGE,
    PUBLIC_TIMEOUT_MESSAGE,
    WEB_SEARCH_MESSAGE,
    VISUAL_SEARCH_MESSAGE,
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
            tuple[dict[str, object], dict[str, dict[str, str]], list[str]]
        ] = []

    def stream(
        self,
        input: dict[str, object],
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


class RecordingTracing:
    """Stand in for Langfuse without needing the SDK installed."""

    def __init__(self):
        self.events: list[str] = []
        self.sessions: list[str] = []

    def callbacks(self) -> list[object]:
        return ["langfuse-handler"]

    def conversation(self, *, session_id: str, tags):
        self.sessions.append(session_id)
        recorder = self.events

        class Scope:
            def __enter__(self):
                recorder.append("open")
                return self

            def __exit__(self, *exc_info):
                recorder.append("close")
                return False

        return Scope()

    def shutdown(self) -> None:
        self.events.append("shutdown")


def test_tracing_wraps_the_stream_and_reaches_the_graph() -> None:
    graph = FakeGraph(
        [
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
    tracing = RecordingTracing()
    adapter = LangGraphStreamAdapter(
        graph_factory=lambda: graph,
        max_rag_attempts=2,
        tracing=tracing,
    )

    list(
        adapter.stream(
            conversation_id="conversation-1",
            message="Pergunta",
            thread_id="thread-hash",
        )
    )

    # The trace has to close after the last event, not when the generator is
    # created, or a streamed answer would be recorded as empty.
    assert tracing.events == ["open", "close"]
    # Sessions group a conversation, so the derived thread is the identity --
    # it is a digest, never the public conversation id.
    assert tracing.sessions == ["thread-hash"]
    assert graph.inputs[0][1]["callbacks"] == ["langfuse-handler"]


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
            {
                "question": "Pergunta",
                "image_context": "",
                "image_document": None,
                "image_reference_documents": [],
                "resolved_equipment_ids": [],
            },
            {
                "configurable": {"thread_id": "conversation-1"},
                "callbacks": [],
            },
            ["updates"],
        )
    ]


class RecordingImageAnalyzer:
    def __init__(self):
        self.calls: list[tuple[str, list[ImageInput]]] = []

    def analyze(
        self,
        question: str,
        images: list[ImageInput],
    ) -> ImageAnalysis:
        self.calls.append((question, list(images)))
        return ImageAnalysis(
            summary="Painel com alarme LOS.",
            observations=["LED LOS aceso em vermelho."],
            visible_text=["LOS"],
            identifiers=["ONT XPTO"],
            uncertainties=[],
            retrieval_query="alarme LOS ONT XPTO procedimento",
        )


class RecordingVisualRetriever:
    def __init__(self, matches: list[RetrievedDocument]):
        self.matches = matches
        self.calls: list[tuple[list[ImageInput], object]] = []

    def retrieve(self, images, *, filters=None):
        self.calls.append((list(images), filters))
        return self.matches


def test_adapter_analyzes_image_before_rag_without_persisting_base64() -> None:
    graph = FakeGraph(
        [
            ("updates", {"retrieve_rag": {"retrieval_attempts": 1}}),
            ("updates", {"grade_rag": {"rag_relevant": True}}),
            (
                "updates",
                {
                    "generate_answer": {
                        "answer": "O alarme LOS indica perda de sinal [1].",
                        "source": "rag",
                        "sources": [RAG_SOURCE],
                    }
                },
            ),
        ]
    )
    analyzer = RecordingImageAnalyzer()
    image: ImageInput = {
        "name": "ont.png",
        "mime_type": "image/png",
        "data": "BASE64-SECRET",
        "sha256": "a" * 64,
    }
    adapter = LangGraphStreamAdapter(
        graph_factory=lambda: graph,
        image_analyzer_factory=lambda: analyzer,
        max_rag_attempts=1,
    )

    events = list(
        adapter.stream(
            conversation_id="c-image",
            message="O que significa este alarme?",
            images=[image],
        )
    )

    statuses = [event.data for event in events if event.event == "status"]
    assert statuses[:2] == [
        {"stage": "image_analysis", "message": IMAGE_ANALYSIS_MESSAGE},
        {"stage": "rag_search", "message": "Buscando na base de conhecimento..."},
    ]
    assert analyzer.calls == [("O que significa este alarme?", [image])]
    graph_input = graph.inputs[0][0]
    assert graph_input["question"] == "O que significa este alarme?"
    assert "Painel com alarme LOS" in str(graph_input["image_context"])
    image_document = graph_input["image_document"]
    assert isinstance(image_document, RetrievedDocument)
    assert image_document.chunk_id == "user-image:aaaaaaaaaaaaaaaa"
    assert "BASE64-SECRET" not in repr(graph_input)
    assert events[-1].event == "done"


def test_adapter_retrieves_proprietary_visual_references_before_rag() -> None:
    graph = FakeGraph(
        [
            ("updates", {"retrieve_rag": {"retrieval_attempts": 1}}),
            ("updates", {"grade_rag": {"rag_relevant": True}}),
            (
                "updates",
                {
                    "generate_answer": {
                        "answer": "É o equipamento XPTO [1].",
                        "source": "rag",
                        "sources": [RAG_SOURCE],
                    }
                },
            ),
        ]
    )
    analyzer = RecordingImageAnalyzer()
    visual_match = RetrievedDocument(
        chunk_id="visual-ref:ref-1",
        content="Referência do equipamento XPTO.",
        metadata={
            "source": "/documents/xpto-front.png",
            "equipment_id": "XPTO-1",
            "equipment_name": "Equipamento XPTO",
            "visual_distance": 0.12,
        },
        retrieval_score=-0.12,
        rerank_score=0.88,
    )
    visual_retriever = RecordingVisualRetriever([visual_match])
    image: ImageInput = {
        "name": "device.png",
        "mime_type": "image/png",
        "data": "BASE64-SECRET",
        "sha256": "b" * 64,
    }
    adapter = LangGraphStreamAdapter(
        graph_factory=lambda: graph,
        image_analyzer_factory=lambda: analyzer,
        visual_catalog_retriever_factory=lambda: visual_retriever,
        max_rag_attempts=1,
    )

    events = list(
        adapter.stream(
            conversation_id="c-visual",
            message="Qual é este equipamento?",
            images=[image],
            retrieval_filters={"tenant_id": "public"},
        )
    )

    statuses = [event.data for event in events if event.event == "status"]
    assert statuses[:3] == [
        {"stage": "image_analysis", "message": IMAGE_ANALYSIS_MESSAGE},
        {"stage": "visual_search", "message": VISUAL_SEARCH_MESSAGE},
        {"stage": "rag_search", "message": "Buscando na base de conhecimento..."},
    ]
    assert visual_retriever.calls == [([image], {"tenant_id": "public"})]
    graph_input = graph.inputs[0][0]
    assert graph_input["image_reference_documents"] == [visual_match]
    assert "XPTO-1" in str(graph_input["image_context"])
    assert "BASE64-SECRET" not in repr(graph_input)


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


def test_a_dead_end_monitoring_question_never_announces_a_rewrite() -> None:
    graph = FakeGraph(
        [
            (
                "updates",
                {
                    "search_monitoring": {
                        "monitoring_documents": [],
                        "monitoring_status": "empty",
                    }
                },
            ),
            ("updates", {"retrieve_rag": {"retrieval_attempts": 1}}),
            ("updates", {"grade_rag": {"rag_relevant": False}}),
            (
                "updates",
                {
                    "no_answer": {
                        "answer": "Consultei o sistema de monitoração e nada veio.",
                        "source": "none",
                        "sources": [],
                    }
                },
            ),
        ]
    )
    adapter = LangGraphStreamAdapter(
        graph_factory=lambda: graph,
        max_rag_attempts=3,
        web_search_enabled=True,
    )

    events = list(adapter.stream(conversation_id="c-1", message="Quais alertas?"))

    statuses = [event.data["stage"] for event in events if event.event == "status"]
    # The graph refuses right after the grader, so promising a rewrite or an
    # external search would narrate work that never happens.
    # The leading rag_search is emitted before the graph reveals its route.
    assert statuses == [
        "rag_search",
        "monitoring_search",
        "rag_search",
        "generating",
    ]
    assert "query_rewrite" not in statuses
    assert "web_search" not in statuses


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


def test_disabled_web_search_does_not_emit_external_search_status() -> None:
    graph = FakeGraph(
        [
            ("updates", {"retrieve_rag": {"retrieval_attempts": 1}}),
            ("updates", {"grade_rag": {"rag_relevant": False}}),
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
        web_search_enabled=False,
    )

    events = list(adapter.stream(conversation_id="c-1", message="Pergunta"))

    statuses = [event.data["stage"] for event in events if event.event == "status"]
    assert statuses == ["rag_search", "generating"]
    assert events[-1].event == "done"


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


def test_cancellation_stops_graph_without_done_or_error() -> None:
    cancelled = [False]

    class CancellingGraph(FakeGraph):
        def stream(self, input, *, config, stream_mode):
            yield ("updates", {"retrieve_rag": {"retrieval_attempts": 1}})
            cancelled[0] = True
            yield ("updates", {"grade_rag": {"rag_relevant": True}})

    adapter = LangGraphStreamAdapter(
        graph_factory=lambda: CancellingGraph([]),
        max_rag_attempts=1,
    )

    events = list(
        adapter.stream(
            conversation_id="c-cancel",
            message="Question",
            is_cancelled=lambda: cancelled[0],
        )
    )

    assert event_names(events) == ["status"]


def test_stream_deadline_returns_specific_safe_error() -> None:
    clock_values = iter([0.0, 2.0])
    graph = FakeGraph(
        [("updates", {"retrieve_rag": {"retrieval_attempts": 1}})]
    )
    adapter = LangGraphStreamAdapter(
        graph_factory=lambda: graph,
        max_rag_attempts=1,
        max_duration_seconds=1.0,
        clock=lambda: next(clock_values),
    )

    events = list(adapter.stream(conversation_id="c-timeout", message="Question"))

    assert events[-1].event == "error"
    assert events[-1].data == {
        "code": "stream_timeout",
        "message": PUBLIC_TIMEOUT_MESSAGE,
    }
