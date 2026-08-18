from typing import Any, Protocol

from app.knowledge.models import Chunk, VectorCandidate


class KnowledgeVectorStore(Protocol):
    def replace_document(
        self,
        document_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        """Replace every stored chunk belonging to a document."""
        ...

    def count(self) -> int:
        ...

    def similarity_search(
        self,
        query_embedding: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorCandidate]:
        """Return nearest dense-vector candidates, ordered by distance."""
        ...
