from typing import Literal, Protocol

from langchain_core.messages import AIMessage, HumanMessage

from app.graph.state import RagState
from app.knowledge.retrieval.models import RetrievalResult
from app.llm.answer_generator import AnswerGenerator, NO_CONTEXT_ANSWER
from app.llm.query_preparer import QueryPreparer
from app.llm.query_rewriter import QueryRewriter
from app.llm.rag_grader import RagGrader


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
        answer_generator: AnswerGenerator,
        max_rag_attempts: int,
    ):
        if max_rag_attempts < 1:
            raise ValueError("max_rag_attempts must be at least 1")
        self._retriever = retriever
        self._grader = grader
        self._query_preparer = query_preparer
        self._query_rewriter = query_rewriter
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
    ) -> Literal["answer", "rewrite", "no_answer"]:
        if state["rag_relevant"]:
            return "answer"
        if state["retrieval_attempts"] < self._max_rag_attempts:
            return "rewrite"
        return "no_answer"

    def rewrite_query(self, state: RagState) -> dict[str, object]:
        rewritten_query = self._query_rewriter.rewrite(
            state["prepared_query"],
            state["search_query"],
            state["rag_reason"],
        )
        return {"search_query": rewritten_query}

    def generate_answer(self, state: RagState) -> dict[str, object]:
        documents = state.get("documents", [])
        answer = self._answer_generator.generate(
            state["prepared_query"],
            documents,
        )
        return {
            "answer": answer,
            "source": "rag",
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
