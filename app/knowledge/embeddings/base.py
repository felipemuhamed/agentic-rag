from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Create one embedding for each document text."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Create an embedding optimized for a search query."""
        ...
