from dataclasses import dataclass
from pathlib import Path

from app.knowledge.embeddings.base import EmbeddingProvider
from app.knowledge.ingestion.chunker import TextChunker
from app.knowledge.ingestion.loaders import DocumentFileLoader
from app.knowledge.ingestion.normalizer import TextNormalizer
from app.knowledge.vectorstore.base import KnowledgeVectorStore


class EmptyDocumentError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class IngestionResult:
    document_id: str
    chunks_written: int


class IngestionService:
    def __init__(
        self,
        loader: DocumentFileLoader,
        normalizer: TextNormalizer,
        chunker: TextChunker,
        embedding_provider: EmbeddingProvider,
        vector_store: KnowledgeVectorStore,
    ):
        self._loader = loader
        self._normalizer = normalizer
        self._chunker = chunker
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def ingest_path(self, path: str | Path) -> IngestionResult:
        document = self._loader.load(path)
        normalized_content = self._normalizer.normalize(document.content)
        if not normalized_content:
            raise EmptyDocumentError(f"Document '{document.source}' is empty")

        document = document.with_content(normalized_content)
        chunks = self._chunker.split(document)
        embeddings = self._embedding_provider.embed_documents(
            [chunk.content for chunk in chunks]
        )
        chunks_written = self._vector_store.replace_document(
            document.document_id,
            chunks,
            embeddings,
        )
        return IngestionResult(
            document_id=document.document_id,
            chunks_written=chunks_written,
        )
