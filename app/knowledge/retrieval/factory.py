from app.knowledge.embeddings import SentenceTransformerEmbeddingProvider
from app.knowledge.embeddings.base import EmbeddingProvider
from app.knowledge.retrieval.reranker import CrossEncoderReranker
from app.knowledge.retrieval.retriever import KnowledgeRetriever
from app.knowledge.vectorstore.base import KnowledgeVectorStore
from app.knowledge.vectorstore.factory import create_knowledge_vector_store
from app.settings import Settings, settings


def create_knowledge_retriever(
    config: Settings = settings,
    *,
    vector_store: KnowledgeVectorStore | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> KnowledgeRetriever:
    """Build the retriever, reusing an already loaded store or model if given."""
    return KnowledgeRetriever(
        vector_store=vector_store or create_knowledge_vector_store(config),
        embedding_provider=embedding_provider
        or SentenceTransformerEmbeddingProvider(
            config.embedding_model,
            device=config.embedding_device,
        ),
        reranker=CrossEncoderReranker(
            config.reranker_model,
            device=config.reranker_device,
        ),
        fetch_k=config.rag_fetch_k,
        candidate_k=config.rag_candidate_k,
        final_k=config.rag_final_k,
        lambda_mult=config.rag_lambda_mult,
    )
