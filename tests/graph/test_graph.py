from app.graph.builder import build_rag_graph
from app.knowledge.retrieval.models import (
    RetrievalDiagnostics,
    RetrievalResult,
    RetrievedDocument,
)
from app.llm.answer_generator import LangChainAnswerGenerator, NO_CONTEXT_ANSWER
from app.llm.rag_grader import RagGrade
from app.llm.web_grader import WebGrade
from app.websearch.models import WebSearchResult


class IdentityQueryPreparer:
    def prepare(self, question: str, history: list[object]) -> str:
        return question


class RecordingRetriever:
    def __init__(self, events: list[str], documents: list[RetrievedDocument]):
        self.events = events
        self.documents = documents

    def retrieve(self, query: str) -> RetrievalResult:
        self.events.append(f"retrieve:{query}")
        count = len(self.documents)
        return RetrievalResult(
            documents=self.documents,
            diagnostics=RetrievalDiagnostics(count, count, count, count),
        )


class RecordingAnswerGenerator:
    def __init__(self, events: list[str]):
        self.events = events

    def generate(
        self,
        question: str,
        documents: list[object],
    ) -> str:
        identifier = getattr(
            documents[0],
            "chunk_id",
            getattr(documents[0], "url", "unknown"),
        )
        self.events.append(f"answer:{question}:{identifier}")
        return "Use o refresh token descrito no guia."


class RecordingGrader:
    def __init__(self, events: list[str], relevant: bool):
        self.events = events
        self.relevant = relevant

    def grade(
        self,
        question: str,
        documents: list[RetrievedDocument],
    ) -> RagGrade:
        chunk_ids = ",".join(document.chunk_id for document in documents)
        self.events.append(f"grade:{question}:{chunk_ids}")
        return RagGrade(
            relevant=self.relevant,
            confidence=0.95,
            reason="Decisão controlada pelo teste.",
        )


class RewriterThatMustNotRun:
    def rewrite(
        self,
        question: str,
        search_query: str,
        failure_reason: str,
    ) -> str:
        raise AssertionError("query rewriter must not run")


class WebFallbackThatMustNotRun:
    def search(self, query: str) -> list[WebSearchResult]:
        raise AssertionError("web fallback must not run")


class WebGraderThatMustNotRun:
    def grade(
        self,
        question: str,
        results: list[WebSearchResult],
    ) -> WebGrade:
        raise AssertionError("web grader must not run")


class RecordingWebSearch:
    def __init__(self, events: list[str], results: list[WebSearchResult]):
        self.events = events
        self.results = results

    def search(self, query: str) -> list[WebSearchResult]:
        self.events.append(f"web:{query}")
        return self.results


class RecordingWebGrader:
    def __init__(self, events: list[str], relevant: bool):
        self.events = events
        self.relevant = relevant

    def grade(
        self,
        question: str,
        results: list[WebSearchResult],
    ) -> WebGrade:
        urls = ",".join(result.url for result in results)
        self.events.append(f"web_grade:{question}:{urls}")
        return WebGrade(
            relevant=self.relevant,
            confidence=0.9,
            reason="Decisão web controlada pelo teste.",
        )


class RecordingQueryRewriter:
    def __init__(self, events: list[str], rewritten_queries: list[str]):
        self.events = events
        self.rewritten_queries = iter(rewritten_queries)

    def rewrite(
        self,
        question: str,
        search_query: str,
        failure_reason: str,
    ) -> str:
        self.events.append(f"rewrite:{search_query}")
        return next(self.rewritten_queries)


def test_linear_graph_answers_from_retrieved_documents() -> None:
    events: list[str] = []
    document = RetrievedDocument(
        chunk_id="oauth-1",
        content="Use o refresh token para obter um novo access token.",
        metadata={"source": "oauth.md"},
        retrieval_score=0.8,
        rerank_score=0.95,
    )
    graph = build_rag_graph(
        retriever=RecordingRetriever(events, [document]),
        grader=RecordingGrader(events, relevant=True),
        query_preparer=IdentityQueryPreparer(),
        query_rewriter=RewriterThatMustNotRun(),
        web_search=WebFallbackThatMustNotRun(),
        web_grader=WebGraderThatMustNotRun(),
        answer_generator=RecordingAnswerGenerator(events),
        max_rag_attempts=2,
    )

    result = graph.invoke({"question": "  Como renovar o token?  "})

    assert result == {
        "answer": "Use o refresh token descrito no guia.",
        "source": "rag",
    }
    assert events == [
        "retrieve:Como renovar o token?",
        "grade:Como renovar o token?:oauth-1",
        "answer:Como renovar o token?:oauth-1",
    ]

    graph_view = graph.get_graph()
    edges = {(edge.source, edge.target) for edge in graph_view.edges}
    assert edges == {
        ("__start__", "prepare_query"),
        ("prepare_query", "retrieve_rag"),
        ("retrieve_rag", "grade_rag"),
        ("grade_rag", "generate_answer"),
        ("grade_rag", "rewrite_query"),
        ("grade_rag", "search_web"),
        ("rewrite_query", "retrieve_rag"),
        ("search_web", "grade_web"),
        ("grade_web", "generate_answer"),
        ("grade_web", "no_answer"),
        ("generate_answer", "__end__"),
        ("no_answer", "__end__"),
    }


class ModelThatMustNotRun:
    def invoke(self, input: object) -> object:
        raise AssertionError("the model must not run without retrieved documents")


def test_linear_graph_fails_closed_when_retrieval_is_empty() -> None:
    events: list[str] = []
    graph = build_rag_graph(
        retriever=RecordingRetriever(events, []),
        grader=RecordingGrader(events, relevant=False),
        query_preparer=IdentityQueryPreparer(),
        query_rewriter=RewriterThatMustNotRun(),
        web_search=RecordingWebSearch(events, []),
        web_grader=RecordingWebGrader(events, relevant=False),
        answer_generator=LangChainAnswerGenerator(ModelThatMustNotRun()),
        max_rag_attempts=1,
    )

    result = graph.invoke({"question": "Pergunta sem cobertura"})

    assert result == {"answer": NO_CONTEXT_ANSWER, "source": "none"}
    assert events == [
        "retrieve:Pergunta sem cobertura",
        "grade:Pergunta sem cobertura:",
        "web:Pergunta sem cobertura",
        "web_grade:Pergunta sem cobertura:",
    ]


def test_irrelevant_documents_do_not_reach_answer_generator() -> None:
    events: list[str] = []
    document = RetrievedDocument(
        chunk_id="unrelated-1",
        content="Este trecho não responde à pergunta.",
        metadata={"source": "outro.md"},
        retrieval_score=0.4,
        rerank_score=0.1,
    )
    graph = build_rag_graph(
        retriever=RecordingRetriever(events, [document]),
        grader=RecordingGrader(events, relevant=False),
        query_preparer=IdentityQueryPreparer(),
        query_rewriter=RewriterThatMustNotRun(),
        web_search=RecordingWebSearch(events, []),
        web_grader=RecordingWebGrader(events, relevant=False),
        answer_generator=LangChainAnswerGenerator(ModelThatMustNotRun()),
        max_rag_attempts=1,
    )

    result = graph.invoke({"question": "Qual é a política de retenção?"})

    assert result == {"answer": NO_CONTEXT_ANSWER, "source": "none"}
    assert events == [
        "retrieve:Qual é a política de retenção?",
        "grade:Qual é a política de retenção?:unrelated-1",
        "web:Qual é a política de retenção?",
        "web_grade:Qual é a política de retenção?:",
    ]


class QueryMappedRetriever:
    def __init__(
        self,
        events: list[str],
        documents_by_query: dict[str, list[RetrievedDocument]],
    ):
        self.events = events
        self.documents_by_query = documents_by_query

    def retrieve(self, query: str) -> RetrievalResult:
        self.events.append(f"retrieve:{query}")
        documents = self.documents_by_query.get(query, [])
        count = len(documents)
        return RetrievalResult(
            documents=documents,
            diagnostics=RetrievalDiagnostics(count, count, count, count),
        )


class EvidenceGrader:
    def __init__(self, events: list[str], relevant_chunk_id: str):
        self.events = events
        self.relevant_chunk_id = relevant_chunk_id

    def grade(
        self,
        question: str,
        documents: list[RetrievedDocument],
    ) -> RagGrade:
        chunk_ids = ",".join(document.chunk_id for document in documents)
        self.events.append(f"grade:{question}:{chunk_ids}")
        relevant = any(
            document.chunk_id == self.relevant_chunk_id
            for document in documents
        )
        return RagGrade(
            relevant=relevant,
            confidence=0.9,
            reason=(
                "Evidência suficiente."
                if relevant
                else "A consulta não recuperou o procedimento."
            ),
        )


def test_graph_answers_after_one_rewrite() -> None:
    events: list[str] = []
    unrelated = RetrievedDocument(
        chunk_id="unrelated-1",
        content="Política geral de autenticação.",
        metadata={"source": "auth.md"},
        retrieval_score=0.5,
        rerank_score=0.2,
    )
    relevant = RetrievedDocument(
        chunk_id="oauth-1",
        content="Renove o access token usando o refresh token.",
        metadata={"source": "oauth.md"},
        retrieval_score=0.8,
        rerank_score=0.95,
    )
    rewritten_query = "renovação access token com refresh token"
    graph = build_rag_graph(
        retriever=QueryMappedRetriever(
            events,
            {
                "Como renovar o token?": [unrelated],
                rewritten_query: [relevant],
            },
        ),
        grader=EvidenceGrader(events, relevant_chunk_id="oauth-1"),
        query_preparer=IdentityQueryPreparer(),
        query_rewriter=RecordingQueryRewriter(events, [rewritten_query]),
        web_search=WebFallbackThatMustNotRun(),
        web_grader=WebGraderThatMustNotRun(),
        answer_generator=RecordingAnswerGenerator(events),
        max_rag_attempts=2,
    )

    result = graph.invoke({"question": "Como renovar o token?"})

    assert result == {
        "answer": "Use o refresh token descrito no guia.",
        "source": "rag",
    }
    assert events == [
        "retrieve:Como renovar o token?",
        "grade:Como renovar o token?:unrelated-1",
        "rewrite:Como renovar o token?",
        f"retrieve:{rewritten_query}",
        "grade:Como renovar o token?:oauth-1",
        "answer:Como renovar o token?:oauth-1",
    ]


def test_graph_stops_exactly_at_configured_attempt_limit() -> None:
    events: list[str] = []
    graph = build_rag_graph(
        retriever=RecordingRetriever(events, []),
        grader=RecordingGrader(events, relevant=False),
        query_preparer=IdentityQueryPreparer(),
        query_rewriter=RecordingQueryRewriter(
            events,
            ["consulta reescrita 1", "consulta reescrita 2"],
        ),
        web_search=RecordingWebSearch(events, []),
        web_grader=RecordingWebGrader(events, relevant=False),
        answer_generator=LangChainAnswerGenerator(ModelThatMustNotRun()),
        max_rag_attempts=3,
    )

    result = graph.invoke({"question": "Pergunta sem cobertura"})

    assert result == {"answer": NO_CONTEXT_ANSWER, "source": "none"}
    assert events == [
        "retrieve:Pergunta sem cobertura",
        "grade:Pergunta sem cobertura:",
        "rewrite:Pergunta sem cobertura",
        "retrieve:consulta reescrita 1",
        "grade:Pergunta sem cobertura:",
        "rewrite:consulta reescrita 1",
        "retrieve:consulta reescrita 2",
        "grade:Pergunta sem cobertura:",
        "web:consulta reescrita 2",
        "web_grade:Pergunta sem cobertura:",
    ]


def test_graph_answers_from_web_only_after_rag_is_exhausted() -> None:
    events: list[str] = []
    web_result = WebSearchResult(
        title="Keycloak OAuth documentation",
        url="https://www.keycloak.org/docs/latest/securing_apps/",
        content="Keycloak exposes OAuth 2.0 authorization endpoints.",
        score=0.93,
        provider="test",
    )
    graph = build_rag_graph(
        retriever=RecordingRetriever(events, []),
        grader=RecordingGrader(events, relevant=False),
        query_preparer=IdentityQueryPreparer(),
        query_rewriter=RewriterThatMustNotRun(),
        web_search=RecordingWebSearch(events, [web_result]),
        web_grader=RecordingWebGrader(events, relevant=True),
        answer_generator=RecordingAnswerGenerator(events),
        max_rag_attempts=1,
    )

    result = graph.invoke({"question": "Como configurar OAuth no Keycloak?"})

    assert result == {
        "answer": "Use o refresh token descrito no guia.",
        "source": "web",
    }
    assert web_result.metadata == {
        "source": "https://www.keycloak.org/docs/latest/securing_apps/",
        "url": "https://www.keycloak.org/docs/latest/securing_apps/",
        "title": "Keycloak OAuth documentation",
        "provider": "test",
    }
    assert events == [
        "retrieve:Como configurar OAuth no Keycloak?",
        "grade:Como configurar OAuth no Keycloak?:",
        "web:Como configurar OAuth no Keycloak?",
        "web_grade:Como configurar OAuth no Keycloak?:"
        "https://www.keycloak.org/docs/latest/securing_apps/",
        "answer:Como configurar OAuth no Keycloak?:"
        "https://www.keycloak.org/docs/latest/securing_apps/",
    ]
