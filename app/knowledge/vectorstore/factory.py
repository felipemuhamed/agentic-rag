from pathlib import Path
from typing import Literal

from app.knowledge.vectorstore.chroma import ChromaVectorStore


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
