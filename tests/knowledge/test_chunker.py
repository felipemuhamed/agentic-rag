from app.knowledge.ingestion.chunker import TextChunker
from app.knowledge.models import SourceDocument


def make_document(content: str, media_type: str = "text/plain") -> SourceDocument:
    return SourceDocument(
        document_id="doc-1",
        title="Guide",
        source="/docs/guide.md",
        content=content,
        created_at="2026-08-17T00:00:00+00:00",
        updated_at="2026-08-17T00:00:00+00:00",
        media_type=media_type,
    )


def test_chunk_ids_are_stable_and_metadata_is_complete() -> None:
    document = make_document(
        "# OAuth\n\n" + ("Token validation and authorization. " * 8),
        media_type="text/markdown",
    )
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)

    first = chunker.split(document)
    second = chunker.split(document)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert len(first) > 1
    assert first[0].metadata == {
        "document_id": "doc-1",
        "title": "Guide",
        "source": "/docs/guide.md",
        "section": "OAuth",
        "chunk_index": 0,
        "start_index": 0,
        "created_at": "2026-08-17T00:00:00+00:00",
        "updated_at": "2026-08-17T00:00:00+00:00",
        "media_type": "text/markdown",
    }


def test_chunker_rejects_invalid_overlap() -> None:
    try:
        TextChunker(chunk_size=50, chunk_overlap=50)
    except ValueError as exc:
        assert "chunk_overlap" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
