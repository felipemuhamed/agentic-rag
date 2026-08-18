from app.knowledge.embeddings import SentenceTransformerEmbeddingProvider
from app.knowledge.retrieval.reranker import CrossEncoderReranker
from app.knowledge.retrieval.retriever import KnowledgeRetriever
from app.knowledge.vectorstore.factory import create_chroma_vector_store
from app.settings import Settings, settings


def create_knowledge_retriever(config: Settings = settings) -> KnowledgeRetriever:
    vector_store = create_chroma_vector_store(
        mode=config.chroma_mode,
        collection_name=config.chroma_collection,
        path=config.chroma_path,
        host=config.chroma_host,
        port=config.chroma_port,
        ssl=config.chroma_ssl,
        tenant=config.chroma_tenant,
        database=config.chroma_database,
    )
    return KnowledgeRetriever(
        vector_store=vector_store,
        embedding_provider=SentenceTransformerEmbeddingProvider(
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
