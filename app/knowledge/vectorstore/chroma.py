from pathlib import Path
from typing import Any

from app.knowledge.models import Chunk, VectorCandidate


class ChromaVectorStore:
    def __init__(
        self,
        path: str | Path,
        collection_name: str,
        *,
        _client: Any | None = None,
    ):
        chromadb = _import_chromadb()
        self._client = _client or chromadb.PersistentClient(path=str(path))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
        )

    @classmethod
    def http(
        cls,
        collection_name: str,
        *,
        host: str,
        port: int,
        ssl: bool = False,
        tenant: str = "default_tenant",
        database: str = "default_database",
    ) -> "ChromaVectorStore":
        chromadb = _import_chromadb()
        client = chromadb.HttpClient(
            host=host,
            port=port,
            ssl=ssl,
            tenant=tenant,
            database=database,
        )
        return cls(".", collection_name, _client=client)

    def replace_document(
        self,
        document_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        self._collection.delete(where={"document_id": document_id})
        if chunks:
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                documents=[chunk.content for chunk in chunks],
                metadatas=[chunk.metadata for chunk in chunks],
                embeddings=embeddings,
            )
        return len(chunks)

    def count(self) -> int:
        return self._collection.count()

    def similarity_search(
        self,
        query_embedding: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorCandidate]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")

        query_arguments: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": limit,
            "include": ["documents", "metadatas", "distances", "embeddings"],
        }
        if filters:
            query_arguments["where"] = filters

        records = self._collection.query(**query_arguments)
        ids = _first_query_batch(records, "ids")
        documents = _first_query_batch(records, "documents")
        metadatas = _first_query_batch(records, "metadatas")
        distances = _first_query_batch(records, "distances")
        embeddings = _first_query_batch(records, "embeddings")

        candidates: list[VectorCandidate] = []
        for chunk_id, content, metadata, distance, embedding in zip(
            ids,
            documents,
            metadatas,
            distances,
            embeddings,
            strict=True,
        ):
            if (
                content is None
                or metadata is None
                or distance is None
                or embedding is None
            ):
                raise RuntimeError("Chroma returned an incomplete candidate record")
            vector = (
                embedding.tolist()
                if hasattr(embedding, "tolist")
                else list(embedding)
            )
            candidates.append(
                VectorCandidate(
                    chunk_id=chunk_id,
                    content=content,
                    metadata=dict(metadata),
                    embedding=[float(value) for value in vector],
                    distance=float(distance),
                )
            )
        return candidates

    def get_document_chunks(self, document_id: str) -> list[dict[str, Any]]:
        records = self._collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
        )
        documents = records.get("documents") or []
        metadatas = records.get("metadatas") or []
        result = [
            {"id": chunk_id, "content": content, "metadata": metadata}
            for chunk_id, content, metadata in zip(
                records["ids"],
                documents,
                metadatas,
                strict=True,
            )
        ]
        return sorted(result, key=lambda item: item["metadata"]["chunk_index"])


def _import_chromadb() -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise RuntimeError(
            "Install the knowledge dependencies with "
            "`python -m pip install -e '.[knowledge]'`"
        ) from exc
    return chromadb


def _first_query_batch(records: dict[str, Any], field: str) -> list[Any]:
    batches = records.get(field)
    if batches is None or len(batches) == 0:
        return []
    return list(batches[0])
