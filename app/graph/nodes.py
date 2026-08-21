from typing import Literal, Protocol

from langchain_core.messages import AIMessage, HumanMessage

from app.graph.state import RagState
from app.knowledge.retrieval.models import RetrievalResult
from app.llm.answer_generator import AnswerGenerator, NO_CONTEXT_ANSWER
from app.llm.query_preparer import QueryPreparer
from app.llm.query_rewriter import QueryRewriter
from app.llm.rag_grader import RagGrader
from app.llm.web_grader import WebGrader
from app.websearch.base import WebSearchProvider


class RetrievalService(Protocol):
    def retrieve(self, query: str) -> RetrievalResult: ...


class RagNodes:
    def __init__(
        self,
        *,
        retriever: RetrievalService,
        grader: RagGrader,
        query_preparer: QueryPreparer,
        query_rewriter: QueryRewriter,
        web_search: WebSearchProvider,
        web_grader: WebGrader,
        answer_generator: AnswerGenerator,
        max_rag_attempts: int,
    ):
        if max_rag_attempts < 1:
            raise ValueError("max_rag_attempts must be at least 1")
        self._retriever = retriever
        self._grader = grader
        self._query_preparer = query_preparer
        self._query_rewriter = query_rewriter
        self._web_search = web_search
        self._web_grader = web_grader
        self._answer_generator = answer_generator
        self._max_rag_attempts = max_rag_attempts

    def prepare_query(self, state: RagState) -> dict[str, object]:
        question = state["question"].strip()
        if not question:
            raise ValueError("question must not be empty")
        prepared_query = self._query_preparer.prepare(
            question,
            state.get("messages", []),
        ).strip()
        if not prepared_query:
            raise RuntimeError("query preparer returned an empty query")
        return {
            "question": question,
            "prepared_query": prepared_query,
            "search_query": prepared_query,
            "retrieval_attempts": 0,
            "web_results": [],
            "web_relevant": False,
        }

    def retrieve_rag(self, state: RagState) -> dict[str, object]:
        result = self._retriever.retrieve(state["search_query"])
        return {
            "documents": result.documents,
            "retrieval_diagnostics": result.diagnostics,
            "retrieval_attempts": state["retrieval_attempts"] + 1,
        }

    def grade_rag(self, state: RagState) -> dict[str, object]:
        grade = self._grader.grade(
            state["prepared_query"],
            state.get("documents", []),
        )
        return {
            "rag_relevant": grade.relevant,
            "rag_confidence": grade.confidence,
            "rag_reason": grade.reason,
        }

    def route_after_grade(
        self,
        state: RagState,
    ) -> Literal["answer", "rewrite", "web"]:
        if state["rag_relevant"]:
            return "answer"
        if state["retrieval_attempts"] < self._max_rag_attempts:
            return "rewrite"
        return "web"

    def rewrite_query(self, state: RagState) -> dict[str, object]:
        rewritten_query = self._query_rewriter.rewrite(
            state["prepared_query"],
            state["search_query"],
            state["rag_reason"],
        )
        return {"search_query": rewritten_query}

    def search_web(self, state: RagState) -> dict[str, object]:
        return {"web_results": self._web_search.search(state["search_query"])}

    def grade_web(self, state: RagState) -> dict[str, object]:
        grade = self._web_grader.grade(
            state["prepared_query"],
            state.get("web_results", []),
        )
        return {
            "web_relevant": grade.relevant,
            "web_confidence": grade.confidence,
            "web_reason": grade.reason,
        }

    def route_after_web_grade(
        self,
        state: RagState,
    ) -> Literal["answer", "no_answer"]:
        return "answer" if state["web_relevant"] else "no_answer"

    def generate_answer(self, state: RagState) -> dict[str, object]:
        from_web = state.get("web_relevant") is True
        documents = (
            state.get("web_results", [])
            if from_web
            else state.get("documents", [])
        )
        answer = self._answer_generator.generate(
            state["prepared_query"],
            documents,
        )
        return {
            "answer": answer,
            "source": "web" if from_web else "rag",
            "messages": [
                HumanMessage(content=state["question"]),
                AIMessage(content=answer),
            ],
        }

    def no_answer(self, state: RagState) -> dict[str, object]:
        return {
            "answer": NO_CONTEXT_ANSWER,
            "source": "none",
            "messages": [
                HumanMessage(content=state["question"]),
                AIMessage(content=NO_CONTEXT_ANSWER),
            ],
        }
