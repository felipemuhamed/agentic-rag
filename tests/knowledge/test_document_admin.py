from pathlib import Path

from app.knowledge.ingestion.chunker import TextChunker
from app.knowledge.ingestion.loaders import DocumentFileLoader
from app.knowledge.ingestion.metadata import FileMetadataExtractor
from app.knowledge.ingestion.normalizer import TextNormalizer
from app.knowledge.ingestion.service import IngestionService
from app.knowledge.vectorstore.chroma import ChromaVectorStore


class DeterministicEmbeddingProvider:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]


def ingest(tmp_path: Path, store: ChromaVectorStore, name: str, tenant: str) -> str:
    path = tmp_path / name
    path.write_text(f"# {name}\n\n" + ("Conteudo operacional. " * 20), encoding="utf-8")
    service = IngestionService(
        loader=DocumentFileLoader(FileMetadataExtractor(tenant_id=tenant)),
        normalizer=TextNormalizer(),
        chunker=TextChunker(chunk_size=90, chunk_overlap=15),
        embedding_provider=DeterministicEmbeddingProvider(),
        vector_store=store,
    )
    return service.ingest_path(path).document_id


def test_documents_are_listed_once_with_their_chunk_count(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "admin_test")
    first = ingest(tmp_path, store, "guia.md", "public")
    second = ingest(tmp_path, store, "manual.md", "public")

    documents = store.list_documents()
    by_id = {document.document_id: document for document in documents}

    assert set(by_id) == {first, second}
    assert by_id[first].chunk_count > 1
    assert by_id[first].title == "guia"
    assert by_id[first].media_type == "text/markdown"
    assert by_id[first].tenant_id == "public"
    assert sum(document.chunk_count for document in documents) == store.count()


def test_listing_can_be_restricted_to_one_tenant(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "admin_test")
    ingest(tmp_path, store, "publico.md", "public")
    private_id = ingest(tmp_path, store, "privado.md", "privado")

    documents = store.list_documents({"tenant_id": "privado"})

    assert [document.document_id for document in documents] == [private_id]


def test_deleting_a_document_removes_only_its_chunks(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "admin_test")
    first = ingest(tmp_path, store, "guia.md", "public")
    ingest(tmp_path, store, "manual.md", "public")
    total_before = store.count()

    deleted = store.delete_document(first)

    assert deleted.chunks_deleted > 0
    assert deleted.source.endswith("guia.md")
    assert store.count() == total_before - deleted.chunks_deleted
    assert [document.document_id for document in store.list_documents()] != [first]


def test_deleting_across_tenants_is_refused_by_the_filter(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "admin_test")
    private_id = ingest(tmp_path, store, "privado.md", "privado")
    total_before = store.count()

    deleted = store.delete_document(private_id, {"tenant_id": "public"})

    assert deleted.chunks_deleted == 0
    assert store.count() == total_before


def test_deleting_an_unknown_document_reports_nothing_removed(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "admin_test")

    deleted = store.delete_document("inexistente")

    assert deleted.chunks_deleted == 0
    assert deleted.source == ""
