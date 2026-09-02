from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from app.graph.builder import build_rag_graph
from app.knowledge.embeddings.base import EmbeddingProvider
from app.knowledge.retrieval.factory import create_knowledge_retriever
from app.knowledge.vectorstore.base import KnowledgeVectorStore
from app.llm.answer_generator import LangChainAnswerGenerator
from app.llm.models import create_chat_model
from app.llm.query_preparer import LangChainQueryPreparer
from app.llm.query_rewriter import LangChainQueryRewriter
from app.llm.rag_grader import LangChainRagGrader
from app.llm.web_grader import LangChainWebGrader
from app.settings import Settings, settings
from app.websearch.factory import create_web_search_provider


def create_rag_graph(
    config: Settings = settings,
    *,
    checkpointer=None,
    vector_store: KnowledgeVectorStore | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    monitoring_router: Any = None,
    monitoring_investigator: Any = None,
):
    model = create_chat_model(
        config.llm_model,
        timeout=config.llm_timeout_seconds,
        # Retries are owned by the node-level runnables below. Disabling the
        # provider retry loop prevents multiplicative attempts and backoffs.
        max_retries=0,
    )
    answer_generator = LangChainAnswerGenerator(
        _with_retry(
            model.with_config({"tags": ["rag-answer"]}),
            config.llm_max_attempts,
        )
    )
    query_preparer = LangChainQueryPreparer(
        _with_retry(
            model.with_config({"tags": ["rag-prepare", "nostream"]}),
            config.llm_max_attempts,
        )
    )
    query_rewriter = LangChainQueryRewriter(
        _with_retry(
            model.with_config({"tags": ["rag-rewrite", "nostream"]}),
            config.llm_max_attempts,
        )
    )
    return build_rag_graph(
        retriever=create_knowledge_retriever(
            config,
            vector_store=vector_store,
            embedding_provider=embedding_provider,
        ),
        grader=LangChainRagGrader(
            model,
            max_attempts=config.llm_max_attempts,
        ),
        query_preparer=query_preparer,
        query_rewriter=query_rewriter,
        web_search=create_web_search_provider(config),
        web_grader=LangChainWebGrader(
            model,
            max_attempts=config.llm_max_attempts,
        ),
        answer_generator=answer_generator,
        monitoring_router=monitoring_router,
        monitoring_investigator=monitoring_investigator,
        max_rag_attempts=config.max_rag_attempts,
        web_search_enabled=config.web_search_enabled,
        checkpointer=(
            checkpointer if checkpointer is not None else InMemorySaver()
        ),
    )


def _with_retry(model, max_attempts: int):
    if max_attempts == 1:
        return model
    return model.with_retry(stop_after_attempt=max_attempts)
