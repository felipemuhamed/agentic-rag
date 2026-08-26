from pathlib import Path

from app.knowledge.ingestion.chunker import TextChunker
from app.knowledge.ingestion.loaders import TextFileLoader
from app.knowledge.ingestion.metadata import FileMetadataExtractor
from app.knowledge.ingestion.normalizer import TextNormalizer
from app.knowledge.ingestion.service import EmptyDocumentError, IngestionService
from app.knowledge.vectorstore.chroma import ChromaVectorStore


class DeterministicEmbeddingProvider:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(len(text)), float(sum(map(ord, text)) % 101), 1.0]
            for text in texts
        ]


def make_service(tmp_path: Path) -> tuple[IngestionService, ChromaVectorStore]:
    vector_store = ChromaVectorStore(tmp_path / "chroma", "knowledge_test")
    service = IngestionService(
        loader=TextFileLoader(),
        normalizer=TextNormalizer(),
        chunker=TextChunker(chunk_size=90, chunk_overlap=15),
        embedding_provider=DeterministicEmbeddingProvider(),
        vector_store=vector_store,
    )
    return service, vector_store


def test_document_is_normalized_chunked_and_persisted(tmp_path: Path) -> None:
    path = tmp_path / "oauth.md"
    path.write_text(
        "# OAuth\r\n\r\n  Access   tokens authorize requests.  \r\n\r\n\r\n"
        + ("Refresh tokens renew access. " * 8),
        encoding="utf-8",
    )
    service, vector_store = make_service(tmp_path)

    first_result = service.ingest_path(path)
    first_records = vector_store.get_document_chunks(first_result.document_id)
    second_result = service.ingest_path(path)
    second_records = vector_store.get_document_chunks(second_result.document_id)
    reopened_store = ChromaVectorStore(tmp_path / "chroma", "knowledge_test")

    assert first_result.chunks_written > 1
    assert vector_store.count() == first_result.chunks_written
    assert reopened_store.count() == first_result.chunks_written
    assert [record["id"] for record in first_records] == [
        record["id"] for record in second_records
    ]
    assert "  " not in first_records[0]["content"]
    assert first_records[0]["metadata"]["section"] == "OAuth"
    assert first_records[0]["metadata"]["source"] == str(path.resolve())


def test_reindex_removes_stale_chunks(tmp_path: Path) -> None:
    path = tmp_path / "knowledge.txt"
    path.write_text("Long content. " * 80, encoding="utf-8")
    service, vector_store = make_service(tmp_path)
    first_result = service.ingest_path(path)

    path.write_text("Short replacement content.", encoding="utf-8")
    second_result = service.ingest_path(path)

    assert first_result.document_id == second_result.document_id
    assert first_result.chunks_written > second_result.chunks_written
    assert second_result.chunks_written == 1
    assert vector_store.count() == 1
    records = vector_store.get_document_chunks(second_result.document_id)
    assert records[0]["content"] == "Short replacement content."


def test_empty_document_is_rejected_before_embedding(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text(" \n\n\t ", encoding="utf-8")
    service, vector_store = make_service(tmp_path)

    try:
        service.ingest_path(path)
    except EmptyDocumentError as exc:
        assert str(path.resolve()) in str(exc)
    else:
        raise AssertionError("Expected EmptyDocumentError")

    assert vector_store.count() == 0


def test_ingestion_adds_tenant_metadata_when_configured(tmp_path: Path) -> None:
    path = tmp_path / "tenant.md"
    path.write_text("# Tenant\n\nScoped content.", encoding="utf-8")
    vector_store = ChromaVectorStore(tmp_path / "tenant-chroma", "tenant_test")
    service = IngestionService(
        loader=TextFileLoader(FileMetadataExtractor(tenant_id="tenant-a")),
        normalizer=TextNormalizer(),
        chunker=TextChunker(chunk_size=90, chunk_overlap=15),
        embedding_provider=DeterministicEmbeddingProvider(),
        vector_store=vector_store,
    )

    result = service.ingest_path(path)
    records = vector_store.get_document_chunks(result.document_id)

    second_service = IngestionService(
        loader=TextFileLoader(FileMetadataExtractor(tenant_id="tenant-b")),
        normalizer=TextNormalizer(),
        chunker=TextChunker(chunk_size=90, chunk_overlap=15),
        embedding_provider=DeterministicEmbeddingProvider(),
        vector_store=vector_store,
    )
    second_result = second_service.ingest_path(path)

    assert records[0]["metadata"]["tenant_id"] == "tenant-a"
    assert result.document_id != second_result.document_id
    assert vector_store.count() == 2


def test_ingestion_links_document_chunks_to_equipment(tmp_path: Path) -> None:
    path = tmp_path / "manual.md"
    path.write_text("# Manual\n\nProcedimento proprietário.", encoding="utf-8")
    vector_store = ChromaVectorStore(tmp_path / "equipment", "manuals")
    service = IngestionService(
        loader=TextFileLoader(
            FileMetadataExtractor(
                tenant_id="public",
                equipment_id="OLT-9000",
            )
        ),
        normalizer=TextNormalizer(),
        chunker=TextChunker(chunk_size=90, chunk_overlap=15),
        embedding_provider=DeterministicEmbeddingProvider(),
        vector_store=vector_store,
    )

    result = service.ingest_path(path)
    records = vector_store.get_document_chunks(result.document_id)

    assert records[0]["metadata"]["tenant_id"] == "public"
    assert records[0]["metadata"]["equipment_id"] == "OLT-9000"
