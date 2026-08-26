from pathlib import Path

from app.knowledge.models import Chunk
from app.knowledge.vectorstore.chroma import ChromaVectorStore


def chunk(chunk_id: str, document_id: str, content: str, topic: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        content=content,
        metadata={
            "document_id": document_id,
            "title": topic,
            "source": f"/{document_id}.txt",
            "section": topic,
            "chunk_index": 0,
            "start_index": 0,
            "created_at": "2026-08-17T00:00:00+00:00",
            "updated_at": "2026-08-17T00:00:00+00:00",
            "media_type": "text/plain",
            "topic": topic,
        },
    )


def test_chroma_dense_search_returns_embeddings_and_honors_filters(
    tmp_path: Path,
) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "retrieval_test")
    store.replace_document(
        "oauth",
        [chunk("oauth-1", "oauth", "OAuth access token", "security")],
        [[1.0, 0.0, 0.0]],
    )
    store.replace_document(
        "cooking",
        [chunk("cooking-1", "cooking", "Bread fermentation", "cooking")],
        [[0.0, 1.0, 0.0]],
    )

    results = store.similarity_search([1.0, 0.0, 0.0], limit=2)
    filtered = store.similarity_search(
        [1.0, 0.0, 0.0],
        limit=2,
        filters={"topic": "cooking"},
    )

    assert [candidate.chunk_id for candidate in results] == ["oauth-1", "cooking-1"]
    assert results[0].embedding == [1.0, 0.0, 0.0]
    assert [candidate.chunk_id for candidate in filtered] == ["cooking-1"]


def test_chroma_combines_tenant_and_equipment_filters(tmp_path: Path) -> None:
    store = ChromaVectorStore(tmp_path / "chroma", "equipment_filter_test")
    first = chunk("olt-1", "olt", "OLT manual", "network")
    first.metadata.update(
        {"tenant_id": "public", "equipment_id": "OLT-9000"}
    )
    second = chunk("ont-1", "ont", "ONT manual", "network")
    second.metadata.update(
        {"tenant_id": "public", "equipment_id": "ONT-100"}
    )
    store.replace_document("olt", [first], [[1.0, 0.0]])
    store.replace_document("ont", [second], [[0.9, 0.1]])

    matches = store.similarity_search(
        [1.0, 0.0],
        limit=2,
        filters={
            "$and": [
                {"tenant_id": "public"},
                {"equipment_id": {"$in": ["OLT-9000"]}},
            ]
        },
    )

    assert [candidate.chunk_id for candidate in matches] == ["olt-1"]
