from pathlib import Path

import pytest

from app.knowledge.models import DeletedDocument, IndexedDocument
from app.uploads.service import DocumentLibraryService, DocumentNotFoundError
from app.uploads.storage import UploadStorage
from app.visual_catalog.ingestion import VisualReference


class DeterministicEmbeddingProvider:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0, 0.0]


class FakeVisualEmbeddingProvider:
    def embed_images(self, images: list[bytes]) -> list[list[float]]:
        return [[float(len(image)), 0.5, 0.25] for image in images]


class FakeStore:
    def __init__(self, *, indexed=None, deleted=None):
        self.documents: dict[str, list] = {}
        self.deletions: list[tuple[str, dict | None]] = []
        self.listings: list[dict | None] = []
        self._indexed = indexed or []
        self._deleted = deleted

    def replace_document(self, document_id, chunks, embeddings):
        self.documents[document_id] = list(chunks)
        return len(chunks)

    def count(self):
        return sum(len(chunks) for chunks in self.documents.values())

    def similarity_search(self, query_embedding, limit, filters=None):
        return []

    def list_documents(self, filters=None):
        self.listings.append(filters)
        return self._indexed

    def delete_document(self, document_id, filters=None):
        self.deletions.append((document_id, filters))
        return self._deleted or DeletedDocument(
            document_id=document_id,
            chunks_deleted=0,
            source="",
        )


def make_service(
    tmp_path: Path,
    *,
    knowledge: FakeStore | None = None,
    visual: FakeStore | None = None,
) -> tuple[DocumentLibraryService, FakeStore, FakeStore, UploadStorage]:
    knowledge_store = knowledge or FakeStore()
    visual_store = visual or FakeStore()
    storage = UploadStorage(tmp_path / "uploads")
    service = DocumentLibraryService(
        storage=storage,
        knowledge_store_factory=lambda: knowledge_store,
        embedding_provider_factory=DeterministicEmbeddingProvider,
        visual_store_factory=lambda: visual_store,
        visual_embedding_provider_factory=FakeVisualEmbeddingProvider,
        chunk_size=120,
        chunk_overlap=20,
    )
    return service, knowledge_store, visual_store, storage


def test_uploaded_document_is_stored_and_indexed_with_tenant_metadata(
    tmp_path: Path,
) -> None:
    service, knowledge, _, storage = make_service(tmp_path)

    result = service.ingest_document(
        filename="guia.md",
        data="# OAuth\n\nRenove o token pelo endpoint documentado.".encode(),
        tenant_id="public",
        equipment_id="OLT-9000",
    )

    stored = Path(result["source"])
    chunks = knowledge.documents[result["documentId"]]
    assert stored.exists()
    assert storage.contains(stored)
    assert result["chunksWritten"] == len(chunks)
    assert chunks[0].metadata["tenant_id"] == "public"
    assert chunks[0].metadata["equipment_id"] == "OLT-9000"
    assert chunks[0].metadata["media_type"] == "text/markdown"


def test_reuploading_the_same_name_replaces_the_previous_version(
    tmp_path: Path,
) -> None:
    service, knowledge, _, _ = make_service(tmp_path)

    first = service.ingest_document(
        filename="guia.md",
        data=b"Primeira versao do guia operacional.",
        tenant_id="public",
    )
    second = service.ingest_document(
        filename="guia.md",
        data=b"Segunda versao do guia operacional com outro conteudo.",
        tenant_id="public",
    )

    assert first["documentId"] == second["documentId"]
    assert len(knowledge.documents) == 1
    assert "Segunda versao" in knowledge.documents[second["documentId"]][0].content


def test_the_same_filename_from_two_tenants_stays_isolated(tmp_path: Path) -> None:
    service, knowledge, _, _ = make_service(tmp_path)

    first = service.ingest_document(
        filename="guia.md",
        data=b"Conteudo do tenant A para o guia.",
        tenant_id="tenant-a",
    )
    second = service.ingest_document(
        filename="guia.md",
        data=b"Conteudo do tenant B para o guia.",
        tenant_id="tenant-b",
    )

    assert first["documentId"] != second["documentId"]
    assert first["source"] != second["source"]
    assert len(knowledge.documents) == 2


def test_a_file_that_cannot_be_indexed_is_not_left_on_disk(tmp_path: Path) -> None:
    service, _, _, storage = make_service(tmp_path)

    with pytest.raises(ValueError):
        service.ingest_document(
            filename="vazio.txt",
            data=b"   \n\n  ",
            tenant_id="public",
        )

    expected = storage.document_path(tenant_id="public", filename="vazio.txt")
    assert not expected.exists()


def test_unsupported_extension_is_rejected_by_the_metadata_extractor(
    tmp_path: Path,
) -> None:
    service, _, _, _ = make_service(tmp_path)

    with pytest.raises(ValueError):
        service.ingest_document(
            filename="planilha.xlsx",
            data=b"conteudo binario",
            tenant_id="public",
        )


def test_reference_image_is_stored_and_indexed_in_the_visual_collection(
    tmp_path: Path,
) -> None:
    service, _, visual, storage = make_service(tmp_path)

    result = service.ingest_reference_image(
        filename="frontal.png",
        data=b"\x89PNG\r\n\x1a\nconteudo",
        reference=VisualReference(
            equipment_id="OLT-9000",
            equipment_name="OLT proprietaria",
            manufacturer="CPQD",
            tenant_id="public",
        ),
    )

    chunk = visual.documents[result["documentId"]][0]
    assert Path(result["source"]).exists()
    assert storage.contains(Path(result["source"]))
    assert chunk.metadata["equipment_id"] == "OLT-9000"
    assert chunk.metadata["tenant_id"] == "public"
    assert chunk.metadata["reference_type"] == "equipment_image"


def test_listing_filters_by_tenant_and_delegates_to_the_store(
    tmp_path: Path,
) -> None:
    indexed = [
        IndexedDocument(
            document_id="abc",
            title="Guia",
            source="/data/uploads/public/documents/guia.md",
            media_type="text/markdown",
            chunk_count=4,
            tenant_id="public",
        )
    ]
    service, knowledge, _, _ = make_service(
        tmp_path,
        knowledge=FakeStore(indexed=indexed),
    )

    with_tenant = service.list_documents(tenant_id="public")
    without_tenant = service.list_documents(tenant_id=None)

    assert with_tenant == indexed
    assert without_tenant == indexed
    assert knowledge.listings == [{"tenant_id": "public"}, None]


def test_deleting_a_document_also_removes_the_stored_upload(
    tmp_path: Path,
) -> None:
    service, knowledge, _, storage = make_service(tmp_path)
    result = service.ingest_document(
        filename="guia.md",
        data=b"Conteudo indexado do guia operacional.",
        tenant_id="public",
    )
    knowledge._deleted = DeletedDocument(
        document_id=result["documentId"],
        chunks_deleted=2,
        source=result["source"],
    )

    removal = service.delete_document(result["documentId"], tenant_id="public")

    assert removal.chunks_deleted == 2
    assert removal.file_removed is True
    assert not Path(result["source"]).exists()
    assert knowledge.deletions == [(result["documentId"], {"tenant_id": "public"})]


def test_deleting_an_unknown_document_raises(tmp_path: Path) -> None:
    service, _, _, _ = make_service(tmp_path)

    with pytest.raises(DocumentNotFoundError):
        service.delete_document("inexistente", tenant_id="public")
