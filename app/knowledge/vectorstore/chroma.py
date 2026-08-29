from pathlib import Path
from typing import Any

from app.knowledge.models import (
    Chunk,
    DeletedDocument,
    IndexedDocument,
    VectorCandidate,
)


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

    def list_documents(
        self,
        filters: dict[str, Any] | None = None,
    ) -> list[IndexedDocument]:
        """Aggregate the stored chunks into one entry per source document."""
        records = self._collection.get(
            where=_where(filters),
            include=["metadatas"],
        )
        documents: dict[str, dict[str, Any]] = {}
        for metadata in records.get("metadatas") or []:
            if not metadata:
                continue
            document_id = str(metadata.get("document_id") or "")
            if not document_id:
                continue
            entry = documents.get(document_id)
            if entry is None:
                documents[document_id] = {
                    "metadata": dict(metadata),
                    "chunk_count": 1,
                }
                continue
            entry["chunk_count"] += 1
            if metadata.get("chunk_index") == 0:
                entry["metadata"] = dict(metadata)

        indexed = [
            IndexedDocument(
                document_id=document_id,
                title=str(entry["metadata"].get("title") or ""),
                source=str(entry["metadata"].get("source") or ""),
                media_type=str(entry["metadata"].get("media_type") or ""),
                chunk_count=entry["chunk_count"],
                created_at=str(entry["metadata"].get("created_at") or ""),
                updated_at=str(entry["metadata"].get("updated_at") or ""),
                tenant_id=str(entry["metadata"].get("tenant_id") or ""),
                equipment_id=str(entry["metadata"].get("equipment_id") or ""),
            )
            for document_id, entry in documents.items()
        ]
        return sorted(
            indexed,
            key=lambda document: (document.updated_at, document.title),
            reverse=True,
        )

    def delete_document(
        self,
        document_id: str,
        filters: dict[str, Any] | None = None,
    ) -> DeletedDocument:
        """Delete every chunk of a document, honouring the tenant filters."""
        conditions: dict[str, Any] = {"document_id": document_id}
        conditions.update(filters or {})
        records = self._collection.get(
            where=_where(conditions),
            include=["metadatas"],
        )
        chunk_ids = list(records.get("ids") or [])
        metadatas = records.get("metadatas") or []
        source = ""
        for metadata in metadatas:
            if metadata and metadata.get("source"):
                source = str(metadata["source"])
                break
        if chunk_ids:
            self._collection.delete(ids=chunk_ids)
        return DeletedDocument(
            document_id=document_id,
            chunks_deleted=len(chunk_ids),
            source=source,
        )

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


def _where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Build a Chroma `where` clause; several keys require an explicit $and."""
    if not filters:
        return None
    if len(filters) == 1:
        return dict(filters)
    return {"$and": [{key: value} for key, value in sorted(filters.items())]}


def _first_query_batch(records: dict[str, Any], field: str) -> list[Any]:
    batches = records.get(field)
    if batches is None or len(batches) == 0:
        return []
    return list(batches[0])
