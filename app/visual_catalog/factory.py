from collections.abc import Callable

from app.knowledge.vectorstore.base import KnowledgeVectorStore
from app.knowledge.vectorstore.factory import create_chroma_vector_store
from app.settings import Settings, settings
from app.visual_catalog.embeddings import (
    SentenceTransformerVisualEmbeddingProvider,
    VisualEmbeddingProvider,
)
from app.visual_catalog.retrieval import VisualCatalogRetriever


def create_visual_vector_store(config: Settings = settings):
    return create_chroma_vector_store(
        mode=config.chroma_mode,
        collection_name=config.visual_collection,
        path=config.chroma_path,
        host=config.chroma_host,
        port=config.chroma_port,
        ssl=config.chroma_ssl,
        tenant=config.chroma_tenant,
        database=config.chroma_database,
    )


def create_visual_embedding_provider(
    config: Settings = settings,
) -> VisualEmbeddingProvider:
    return SentenceTransformerVisualEmbeddingProvider(
        config.visual_embedding_model,
        device=config.visual_embedding_device,
    )


def create_visual_catalog_retriever(
    config: Settings = settings,
    *,
    vector_store: KnowledgeVectorStore | None = None,
    embedding_provider_factory: Callable[[], VisualEmbeddingProvider] | None = None,
) -> VisualCatalogRetriever:
    """Build the retriever, reusing an already loaded store or model if given."""
    return VisualCatalogRetriever(
        vector_store=vector_store or create_visual_vector_store(config),
        embedding_provider_factory=(
            embedding_provider_factory
            or (lambda: create_visual_embedding_provider(config))
        ),
        max_results=config.visual_max_results,
        max_distance=config.visual_max_distance,
    )
