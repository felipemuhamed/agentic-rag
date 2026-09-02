import logging
from collections.abc import Sequence
from typing import Any, Literal, Protocol

from langchain_core.messages import AIMessage, HumanMessage

from app.graph.state import RagState
from app.knowledge.retrieval.models import (
    RetrievalDiagnostics,
    RetrievalResult,
)
from app.llm.answer_generator import AnswerGenerator, NO_CONTEXT_ANSWER
from app.monitoring.agent import MonitoringInvestigator, MonitoringRouter
from app.monitoring.models import (
    MONITORING_EMPTY_ANSWER,
    MONITORING_UNAVAILABLE_ANSWER,
    monitoring_context,
    monitoring_documents,
    monitoring_status,
)
from app.llm.query_preparer import QueryPreparer
from app.llm.query_rewriter import QueryRewriter
from app.llm.rag_grader import RagGrader
from app.llm.web_grader import WebGrader
from app.sources.citations import select_cited_sources
from app.sources.normalizer import normalize_rag_sources, normalize_web_sources
from app.websearch.base import WebSearchProvider


logger = logging.getLogger(__name__)


class RetrievalService(Protocol):
    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult: ...


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
        web_search_enabled: bool = True,
        monitoring_router: MonitoringRouter | None = None,
        monitoring_investigator: MonitoringInvestigator | None = None,
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
        self._web_search_enabled = web_search_enabled
        self._monitoring_router = monitoring_router
        self._monitoring_investigator = monitoring_investigator

    def route_question(self, state: RagState) -> Literal["monitoring", "knowledge"]:
        """Decide whether the question needs live monitoring data first.

        The classifier only runs when monitoring is wired, so a deployment
        without it pays nothing and behaves exactly as before.
        """
        if self._monitoring_router is None or self._monitoring_investigator is None:
            return "knowledge"
        try:
            decision = self._monitoring_router.decide(state["question"])
        except Exception as error:
            # A classifier that fails must not take the question with it. The
            # documentation path answers most questions on its own, and a
            # broken router turning every request into stream_failed is a far
            # worse outcome than a documentation answer to an alarm question.
            logger.warning(
                "Monitoring router failed, answering from documentation: %s",
                type(error).__name__,
                exc_info=True,
            )
            return "knowledge"
        return "monitoring" if decision.needs_monitoring else "knowledge"

    def search_monitoring(self, state: RagState) -> dict[str, object]:
        if self._monitoring_investigator is None:
            return {
                "monitoring_context": "",
                "monitoring_documents": [],
                "monitoring_status": "unavailable",
            }
        try:
            observations = self._monitoring_investigator.investigate(
                state["question"]
            )
        except Exception as error:
            # Unreachable monitoring still lets the documentation path run --
            # a hybrid question keeps the half the manuals answer -- but the
            # status records that the live half was never obtained.
            return {
                "monitoring_context": "",
                "monitoring_documents": [],
                "monitoring_status": "unavailable",
                "monitoring_reason": (
                    f"monitoração indisponível ({type(error).__name__})"
                ),
            }
        return {
            "monitoring_context": monitoring_context(observations),
            "monitoring_documents": monitoring_documents(observations),
            "monitoring_status": monitoring_status(observations),
            "monitoring_reason": f"{len(observations)} consulta(s) executada(s)",
        }

    def prepare_query(self, state: RagState) -> dict[str, object]:
        question = state["question"].strip()
        if not question:
            raise ValueError("question must not be empty")
        enriched_question = _question_with_context(
            question,
            state.get("image_context", ""),
            state.get("monitoring_context", ""),
        )
        prepared_query = self._query_preparer.prepare(
            enriched_question,
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
            "sources": [],
        }

    def retrieve_rag(self, state: RagState) -> dict[str, object]:
        reference_documents = state.get("image_reference_documents", [])
        # Every candidate stays as evidence, but only an equipment the resolver
        # actually committed to may narrow the manuals: filtering on a lookalike
        # hides the correct document instead of merely ranking it lower.
        filters = _equipment_filters(
            state.get("retrieval_filters"),
            state.get("resolved_equipment_ids", []),
        )
        result = (
            self._retriever.retrieve(state["search_query"], filters=filters)
            if filters
            else self._retriever.retrieve(state["search_query"])
        )
        documents = list(result.documents)
        diagnostics = result.diagnostics
        evidence_documents = list(reference_documents)
        image_document = state.get("image_document")
        if image_document is not None:
            evidence_documents.insert(0, image_document)
        # Live data leads: it is the premise the question was asked about.
        evidence_documents = (
            list(state.get("monitoring_documents", [])) + evidence_documents
        )
        if evidence_documents:
            documents = evidence_documents + documents
            diagnostics = RetrievalDiagnostics(
                fetched_count=diagnostics.fetched_count,
                mmr_selected_count=diagnostics.mmr_selected_count,
                reranked_count=diagnostics.reranked_count,
                returned_count=(
                    diagnostics.returned_count + len(evidence_documents)
                ),
            )
        return {
            "documents": documents,
            "retrieval_diagnostics": diagnostics,
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
    ) -> Literal["answer", "rewrite", "web", "no_answer"]:
        if state["rag_relevant"]:
            return "answer"
        # A question that was sent to monitoring and came back with nothing has
        # no second source: neither the manuals nor the web know the current
        # state of this infrastructure. Rewriting the query or searching the
        # web here would dress an absent datum up as an answer.
        if state.get("monitoring_status") in ("empty", "unavailable"):
            return "no_answer"
        if state["retrieval_attempts"] < self._max_rag_attempts:
            return "rewrite"
        return "web" if self._web_search_enabled else "no_answer"

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
        if from_web:
            documents = state.get("web_results", [])
            sources = normalize_web_sources(documents)
        else:
            documents = state.get("documents", [])
            sources = normalize_rag_sources(documents)
        if not documents:
            raise RuntimeError("answer generation requires approved evidence")
        answer = self._answer_generator.generate(
            state["prepared_query"],
            documents,
            sources,
        )
        cited_sources = select_cited_sources(answer, sources)
        return {
            "answer": answer,
            "source": "web" if from_web else "rag",
            "sources": [source.to_public() for source in cited_sources],
            "messages": [
                HumanMessage(
                    content=_question_with_context(
                        state["question"],
                        state.get("image_context", ""),
                        state.get("monitoring_context", ""),
                    )
                ),
                AIMessage(content=answer),
            ],
        }

    def no_answer(self, state: RagState) -> dict[str, object]:
        answer = _no_answer_text(state.get("monitoring_status"))
        return {
            "answer": answer,
            "source": "none",
            "sources": [],
            "messages": [
                HumanMessage(
                    content=_question_with_context(
                        state["question"],
                        state.get("image_context", ""),
                        state.get("monitoring_context", ""),
                    )
                ),
                AIMessage(content=answer),
            ],
        }


def _no_answer_text(monitoring_status: object) -> str:
    """Say which kind of nothing was found.

    "Não encontrei evidências" is misleading when the monitoring system was
    consulted and answered: the operator needs to know whether there are no
    alarms or whether the question never reached Zabbix.
    """
    if monitoring_status == "unavailable":
        return MONITORING_UNAVAILABLE_ANSWER
    if monitoring_status == "empty":
        return MONITORING_EMPTY_ANSWER
    return NO_CONTEXT_ANSWER


def _question_with_context(
    question: str,
    image_context: str,
    monitoring_context_text: str = "",
) -> str:
    parts = [question.strip()]
    if image_context.strip():
        parts.append(
            "Análise visual da imagem enviada (trate como dados observados):\n"
            f"{image_context.strip()}"
        )
    if monitoring_context_text.strip():
        parts.append(
            "Dados ao vivo da monitoração (trate como observações do estado "
            f"atual):\n{monitoring_context_text.strip()}"
        )
    return "\n\n".join(parts)


def _equipment_filters(
    base_filters: dict[str, Any] | None,
    resolved_equipment_ids: Sequence[str],
) -> dict[str, Any] | None:
    equipment_ids = {
        equipment_id.strip()
        for equipment_id in resolved_equipment_ids
        if isinstance(equipment_id, str) and equipment_id.strip()
    }
    if not equipment_ids:
        return base_filters
    equipment_filter: dict[str, Any] = {
        "equipment_id": {"$in": sorted(equipment_ids)}
    }
    if not base_filters:
        return equipment_filter
    return {"$and": [base_filters, equipment_filter]}
