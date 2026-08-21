from langgraph.checkpoint.memory import InMemorySaver

from app.graph.builder import build_rag_graph
from app.knowledge.retrieval.factory import create_knowledge_retriever
from app.llm.answer_generator import LangChainAnswerGenerator
from app.llm.models import create_chat_model
from app.llm.query_preparer import LangChainQueryPreparer
from app.llm.query_rewriter import LangChainQueryRewriter
from app.llm.rag_grader import LangChainRagGrader
from app.llm.web_grader import LangChainWebGrader
from app.settings import Settings, settings
from app.websearch.factory import create_web_search_provider


def create_rag_graph(config: Settings = settings):
    model = create_chat_model(config.llm_model)
    answer_generator = LangChainAnswerGenerator(
        model.with_config({"tags": ["rag-answer"]})
    )
    query_preparer = LangChainQueryPreparer(
        model.with_config({"tags": ["rag-prepare", "nostream"]})
    )
    query_rewriter = LangChainQueryRewriter(
        model.with_config({"tags": ["rag-rewrite", "nostream"]})
    )
    return build_rag_graph(
        retriever=create_knowledge_retriever(config),
        grader=LangChainRagGrader(model),
        query_preparer=query_preparer,
        query_rewriter=query_rewriter,
        web_search=create_web_search_provider(config),
        web_grader=LangChainWebGrader(model),
        answer_generator=answer_generator,
        max_rag_attempts=config.max_rag_attempts,
        checkpointer=InMemorySaver(),
    )
