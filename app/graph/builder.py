from typing import Any

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import RagNodes, RetrievalService
from app.graph.state import RagInput, RagOutput, RagState
from app.llm.answer_generator import AnswerGenerator
from app.llm.query_preparer import QueryPreparer
from app.llm.query_rewriter import QueryRewriter
from app.llm.rag_grader import RagGrader
from app.llm.web_grader import WebGrader
from app.websearch.base import WebSearchProvider


def build_rag_graph(
    *,
    retriever: RetrievalService,
    grader: RagGrader,
    query_preparer: QueryPreparer,
    query_rewriter: QueryRewriter,
    web_search: WebSearchProvider,
    web_grader: WebGrader,
    answer_generator: AnswerGenerator,
    max_rag_attempts: int,
    checkpointer: Any = None,
):
    nodes = RagNodes(
        retriever=retriever,
        grader=grader,
        query_preparer=query_preparer,
        query_rewriter=query_rewriter,
        web_search=web_search,
        web_grader=web_grader,
        answer_generator=answer_generator,
        max_rag_attempts=max_rag_attempts,
    )
    builder = StateGraph(
        RagState,
        input_schema=RagInput,
        output_schema=RagOutput,
    )
    builder.add_node("prepare_query", nodes.prepare_query)
    builder.add_node("retrieve_rag", nodes.retrieve_rag)
    builder.add_node("grade_rag", nodes.grade_rag)
    builder.add_node("rewrite_query", nodes.rewrite_query)
    builder.add_node("search_web", nodes.search_web)
    builder.add_node("grade_web", nodes.grade_web)
    builder.add_node("generate_answer", nodes.generate_answer)
    builder.add_node("no_answer", nodes.no_answer)
    builder.add_edge(START, "prepare_query")
    builder.add_edge("prepare_query", "retrieve_rag")
    builder.add_edge("retrieve_rag", "grade_rag")
    builder.add_conditional_edges(
        "grade_rag",
        nodes.route_after_grade,
        {
            "answer": "generate_answer",
            "rewrite": "rewrite_query",
            "web": "search_web",
        },
    )
    builder.add_edge("rewrite_query", "retrieve_rag")
    builder.add_edge("search_web", "grade_web")
    builder.add_conditional_edges(
        "grade_web",
        nodes.route_after_web_grade,
        {
            "answer": "generate_answer",
            "no_answer": "no_answer",
        },
    )
    builder.add_edge("generate_answer", END)
    builder.add_edge("no_answer", END)
    return builder.compile(checkpointer=checkpointer)
