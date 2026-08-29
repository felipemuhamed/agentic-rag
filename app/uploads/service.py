import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from app.knowledge.embeddings.base import EmbeddingProvider
from app.knowledge.ingestion.chunker import TextChunker
from app.knowledge.ingestion.loaders import DocumentFileLoader
from app.knowledge.ingestion.metadata import FileMetadataExtractor
from app.knowledge.ingestion.normalizer import TextNormalizer
from app.knowledge.ingestion.service import IngestionService
from app.knowledge.models import DeletedDocument, IndexedDocument
from app.knowledge.vectorstore.base import ManagedKnowledgeVectorStore
from app.uploads.storage import UploadStorage
from app.visual_catalog.embeddings import VisualEmbeddingProvider
from app.visual_catalog.ingestion import (
    VisualReference,
    VisualReferenceIngestionService,
)


logger = logging.getLogger(__name__)


class DocumentNotFoundError(LookupError):
    pass


@dataclass(frozen=True, slots=True)
class DocumentRemoval:
    document_id: str
    chunks_deleted: int
    file_removed: bool


class DocumentLibraryService:
    """Ingest, list and remove the documents that feed the RAG collections.

    Text documents land in the knowledge collection and reference photos in the
    visual collection. Heavy dependencies arrive as factories so the embedding
    models are only loaded when the first upload is actually processed.
    """

    def __init__(
        self,
        *,
        storage: UploadStorage,
        knowledge_store_factory: Callable[[], ManagedKnowledgeVectorStore],
        embedding_provider_factory: Callable[[], EmbeddingProvider],
        visual_store_factory: Callable[[], ManagedKnowledgeVectorStore],
        visual_embedding_provider_factory: Callable[[], VisualEmbeddingProvider],
        chunk_size: int,
        chunk_overlap: int,
    ):
        self._storage = storage
        self._knowledge_store_factory = knowledge_store_factory
        self._embedding_provider_factory = embedding_provider_factory
        self._visual_store_factory = visual_store_factory
        self._visual_embedding_provider_factory = visual_embedding_provider_factory
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._knowledge_store: ManagedKnowledgeVectorStore | None = None
        self._embedding_provider: EmbeddingProvider | None = None
        self._visual_store: ManagedKnowledgeVectorStore | None = None
        self._visual_embedding_provider: VisualEmbeddingProvider | None = None
        self._lock = Lock()

    def save_document(
        self,
        *,
        filename: str,
        data: bytes,
        tenant_id: str | None,
    ) -> Path:
        """Persist the upload before it is queued, so the wait costs no memory."""
        return self._storage.save(
            self._storage.document_path(tenant_id=tenant_id, filename=filename),
            data,
        )

    def ingest_document(
        self,
        *,
        filename: str,
        data: bytes,
        tenant_id: str | None,
        equipment_id: str = "",
    ) -> dict[str, Any]:
        path = self.save_document(
            filename=filename,
            data=data,
            tenant_id=tenant_id,
        )
        return self.ingest_saved_document(
            path,
            tenant_id=tenant_id,
            equipment_id=equipment_id,
        )

    def ingest_saved_document(
        self,
        path: Path,
        *,
        tenant_id: str | None,
        equipment_id: str = "",
    ) -> dict[str, Any]:
        try:
            service = IngestionService(
                loader=DocumentFileLoader(
                    FileMetadataExtractor(
                        tenant_id=tenant_id or "",
                        equipment_id=equipment_id,
                    )
                ),
                normalizer=TextNormalizer(),
                chunker=TextChunker(self._chunk_size, self._chunk_overlap),
                embedding_provider=self._embeddings(),
                vector_store=self._knowledge(),
            )
            result = service.ingest_path(path)
        except Exception:
            # A file that could not be indexed would only leave an orphan on
            # disk, and keeping it would shadow a later valid upload.
            self._storage.remove(path)
            raise

        logger.info(
            "Document ingested from upload",
            extra={
                "document_id": result.document_id,
                "chunks_written": result.chunks_written,
            },
        )
        return {
            "documentId": result.document_id,
            "chunksWritten": result.chunks_written,
            "source": str(path),
            "filename": path.name,
            "equipmentId": equipment_id or None,
        }

    def save_reference_image(
        self,
        *,
        filename: str,
        data: bytes,
        reference: VisualReference,
    ) -> Path:
        return self._storage.save(
            self._storage.reference_image_path(
                tenant_id=reference.tenant_id or None,
                equipment_id=reference.equipment_id,
                filename=filename,
            ),
            data,
        )

    def ingest_reference_image(
        self,
        *,
        filename: str,
        data: bytes,
        reference: VisualReference,
    ) -> dict[str, Any]:
        path = self.save_reference_image(
            filename=filename,
            data=data,
            reference=reference,
        )
        return self.ingest_saved_reference_image(path, reference=reference)

    def ingest_saved_reference_image(
        self,
        path: Path,
        *,
        reference: VisualReference,
    ) -> dict[str, Any]:
        try:
            service = VisualReferenceIngestionService(
                embedding_provider=self._visual_embeddings(),
                vector_store=self._visual(),
            )
            result = service.ingest_path(path, reference)
        except Exception:
            self._storage.remove(path)
            raise

        logger.info(
            "Reference image indexed from upload",
            extra={
                "document_id": result.document_id,
                "equipment_id": result.equipment_id,
            },
        )
        return {
            "documentId": result.document_id,
            "referenceId": result.reference_id,
            "equipmentId": result.equipment_id,
            "source": str(path),
            "filename": path.name,
        }

    def list_documents(self, *, tenant_id: str | None) -> list[IndexedDocument]:
        return self._knowledge().list_documents(_tenant_filter(tenant_id))

    def list_reference_images(
        self,
        *,
        tenant_id: str | None,
    ) -> list[IndexedDocument]:
        return self._visual().list_documents(_tenant_filter(tenant_id))

    def delete_document(
        self,
        document_id: str,
        *,
        tenant_id: str | None,
    ) -> DocumentRemoval:
        return self._delete(
            self._knowledge(),
            document_id=document_id,
            tenant_id=tenant_id,
        )

    def delete_reference_image(
        self,
        document_id: str,
        *,
        tenant_id: str | None,
    ) -> DocumentRemoval:
        return self._delete(
            self._visual(),
            document_id=document_id,
            tenant_id=tenant_id,
        )

    def _delete(
        self,
        store: ManagedKnowledgeVectorStore,
        *,
        document_id: str,
        tenant_id: str | None,
    ) -> DocumentRemoval:
        deleted: DeletedDocument = store.delete_document(
            document_id,
            _tenant_filter(tenant_id),
        )
        if deleted.chunks_deleted == 0:
            raise DocumentNotFoundError(document_id)
        file_removed = bool(deleted.source) and self._storage.remove(
            Path(deleted.source)
        )
        logger.info(
            "Document removed from the index",
            extra={
                "document_id": document_id,
                "chunks_deleted": deleted.chunks_deleted,
                "file_removed": file_removed,
            },
        )
        return DocumentRemoval(
            document_id=document_id,
            chunks_deleted=deleted.chunks_deleted,
            file_removed=file_removed,
        )

    def _knowledge(self) -> ManagedKnowledgeVectorStore:
        with self._lock:
            if self._knowledge_store is None:
                self._knowledge_store = self._knowledge_store_factory()
            return self._knowledge_store

    def _embeddings(self) -> EmbeddingProvider:
        with self._lock:
            if self._embedding_provider is None:
                self._embedding_provider = self._embedding_provider_factory()
            return self._embedding_provider

    def _visual(self) -> ManagedKnowledgeVectorStore:
        with self._lock:
            if self._visual_store is None:
                self._visual_store = self._visual_store_factory()
            return self._visual_store

    def _visual_embeddings(self) -> VisualEmbeddingProvider:
        with self._lock:
            if self._visual_embedding_provider is None:
                self._visual_embedding_provider = (
                    self._visual_embedding_provider_factory()
                )
            return self._visual_embedding_provider


def _tenant_filter(tenant_id: str | None) -> dict[str, str] | None:
    normalized = (tenant_id or "").strip()
    return {"tenant_id": normalized} if normalized else None
