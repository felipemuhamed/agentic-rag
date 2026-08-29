from pathlib import Path
from typing import Literal

from app.knowledge.vectorstore.chroma import ChromaVectorStore
from app.settings import Settings, settings


def create_chroma_vector_store(
    *,
    mode: Literal["http", "persistent"],
    collection_name: str,
    path: str | Path,
    host: str,
    port: int,
    ssl: bool,
    tenant: str,
    database: str,
) -> ChromaVectorStore:
    if mode == "http":
        return ChromaVectorStore.http(
            collection_name,
            host=host,
            port=port,
            ssl=ssl,
            tenant=tenant,
            database=database,
        )
    return ChromaVectorStore(path, collection_name)


def create_knowledge_vector_store(
    config: Settings = settings,
) -> ChromaVectorStore:
    """Open the collection that stores the ingested text documents."""
    return create_chroma_vector_store(
        mode=config.chroma_mode,
        collection_name=config.chroma_collection,
        path=config.chroma_path,
        host=config.chroma_host,
        port=config.chroma_port,
        ssl=config.chroma_ssl,
        tenant=config.chroma_tenant,
        database=config.chroma_database,
    )
