from collections.abc import Sequence

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.builder import build_rag_graph
from app.knowledge.retrieval.models import (
    RetrievalDiagnostics,
    RetrievalResult,
    RetrievedDocument,
)
from app.llm.rag_grader import RagGrade
from app.sources.models import Source


class ContextAwareQueryPreparer:
    def __init__(self):
        self.calls: list[tuple[str, list[str]]] = []

    def prepare(
        self,
        question: str,
        history: Sequence[BaseMessage],
    ) -> str:
        history_contents = [str(message.content) for message in history]
        self.calls.append((question, history_contents))
        if history and question == "E no Keycloak?":
            return "Como configurar OAuth no Keycloak?"
        return question


class RecordingRetriever:
    def __init__(self):
        self.queries: list[str] = []

    def retrieve(self, query: str) -> RetrievalResult:
        self.queries.append(query)
        document = RetrievedDocument(
            chunk_id="oauth-1",
            content="Procedimento OAuth aplicável à consulta.",
            metadata={"source": "oauth.md"},
            retrieval_score=0.9,
            rerank_score=0.95,
        )
        return RetrievalResult(
            documents=[document],
            diagnostics=RetrievalDiagnostics(1, 1, 1, 1),
        )


class RelevantGrader:
    def grade(
        self,
        question: str,
        documents: list[RetrievedDocument],
    ) -> RagGrade:
        return RagGrade(relevant=True, confidence=1.0, reason="Suficiente.")


class RewriterThatMustNotRun:
    def rewrite(
        self,
        question: str,
        search_query: str,
        failure_reason: str,
    ) -> str:
        raise AssertionError("rewriter must not run")


class WebFallbackThatMustNotRun:
    def search(self, query: str) -> list[object]:
        raise AssertionError("web fallback must not run")


class WebGraderThatMustNotRun:
    def grade(self, question: str, results: list[object]) -> object:
        raise AssertionError("web grader must not run")


class QueryEchoAnswerGenerator:
    def generate(
        self,
        question: str,
        documents: list[RetrievedDocument],
        sources: list[Source],
    ) -> str:
        return f"Resposta para: {question} [1]"


def test_follow_up_uses_only_its_thread_history() -> None:
    preparer = ContextAwareQueryPreparer()
    retriever = RecordingRetriever()
    graph = build_rag_graph(
        retriever=retriever,
        grader=RelevantGrader(),
        query_preparer=preparer,
        query_rewriter=RewriterThatMustNotRun(),
        web_search=WebFallbackThatMustNotRun(),
        web_grader=WebGraderThatMustNotRun(),
        answer_generator=QueryEchoAnswerGenerator(),
        max_rag_attempts=1,
        checkpointer=InMemorySaver(),
    )
    thread_a = {"configurable": {"thread_id": "conversation-a"}}
    thread_b = {"configurable": {"thread_id": "conversation-b"}}

    graph.invoke({"question": "Como configurar OAuth?"}, config=thread_a)
    follow_up = graph.invoke({"question": "E no Keycloak?"}, config=thread_a)
    isolated = graph.invoke({"question": "E no Keycloak?"}, config=thread_b)

    assert follow_up == {
        "answer": "Resposta para: Como configurar OAuth no Keycloak? [1]",
        "source": "rag",
        "sources": [
            {
                "id": 1,
                "citation": "[1]",
                "type": "rag",
                "title": "oauth",
                "source": "oauth.md",
                "chunkId": "oauth-1",
                "url": None,
                "page": None,
                "section": None,
            }
        ],
    }
    assert isolated == {
        "answer": "Resposta para: E no Keycloak? [1]",
        "source": "rag",
        "sources": follow_up["sources"],
    }
    assert retriever.queries == [
        "Como configurar OAuth?",
        "Como configurar OAuth no Keycloak?",
        "E no Keycloak?",
    ]
    assert preparer.calls == [
        ("Como configurar OAuth?", []),
        (
            "E no Keycloak?",
            [
                "Como configurar OAuth?",
                "Resposta para: Como configurar OAuth? [1]",
            ],
        ),
        ("E no Keycloak?", []),
    ]

    messages_a = graph.get_state(thread_a).values["messages"]
    messages_b = graph.get_state(thread_b).values["messages"]
    assert [message.content for message in messages_a] == [
        "Como configurar OAuth?",
        "Resposta para: Como configurar OAuth? [1]",
        "E no Keycloak?",
        "Resposta para: Como configurar OAuth no Keycloak? [1]",
    ]
    assert [message.content for message in messages_b] == [
        "E no Keycloak?",
        "Resposta para: E no Keycloak? [1]",
    ]
