from app.knowledge.vectorstore.factory import create_chroma_vector_store
from app.settings import Settings, settings
from app.visual_catalog.embeddings import (
    SentenceTransformerVisualEmbeddingProvider,
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


def create_visual_catalog_retriever(
    config: Settings = settings,
) -> VisualCatalogRetriever:
    return VisualCatalogRetriever(
        vector_store=create_visual_vector_store(config),
        embedding_provider_factory=lambda: (
            SentenceTransformerVisualEmbeddingProvider(
                config.visual_embedding_model,
                device=config.visual_embedding_device,
            )
        ),
        max_results=config.visual_max_results,
        max_distance=config.visual_max_distance,
    )
