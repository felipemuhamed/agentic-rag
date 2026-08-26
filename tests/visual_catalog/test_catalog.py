import base64
from pathlib import Path

from app.knowledge.models import VectorCandidate
from app.multimodal.models import ImageInput
from app.visual_catalog.cli import build_parser
from app.visual_catalog.ingestion import (
    VisualReference,
    VisualReferenceIngestionService,
)
from app.visual_catalog.retrieval import VisualCatalogRetriever


class FakeVisualEmbedder:
    def __init__(self, embeddings: list[list[float]]):
        self.embeddings = embeddings
        self.calls: list[list[bytes]] = []

    def embed_images(self, images):
        self.calls.append(list(images))
        return self.embeddings


class FakeVisualStore:
    def __init__(self, *, candidates=None, count: int = 1):
        self.candidates = candidates or []
        self.stored = []
        self.searches = []
        self._count = count

    def replace_document(self, document_id, chunks, embeddings):
        self.stored.append((document_id, chunks, embeddings))
        return len(chunks)

    def count(self):
        return self._count

    def similarity_search(self, query_embedding, limit, filters=None):
        self.searches.append((query_embedding, limit, filters))
        return self.candidates


def test_reference_image_is_embedded_with_equipment_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "front.png"
    path.write_bytes(b"fake-image-for-test")
    embedder = FakeVisualEmbedder([[0.1, 0.2, 0.3]])
    store = FakeVisualStore()
    service = VisualReferenceIngestionService(
        embedding_provider=embedder,
        vector_store=store,
    )

    result = service.ingest_path(
        path,
        VisualReference(
            equipment_id="OLT-9000",
            equipment_name="OLT Proprietária",
            manufacturer="CPQD",
            model="X9000",
            description="Vista frontal com oito portas.",
            tenant_id="public",
        ),
    )

    assert result.equipment_id == "OLT-9000"
    assert embedder.calls == [[b"fake-image-for-test"]]
    document_id, chunks, embeddings = store.stored[0]
    assert document_id == result.document_id
    assert embeddings == [[0.1, 0.2, 0.3]]
    assert chunks[0].metadata["equipment_id"] == "OLT-9000"
    assert chunks[0].metadata["tenant_id"] == "public"
    assert chunks[0].metadata["model"] == "X9000"
    assert "image_sha256" in chunks[0].metadata


def test_visual_retrieval_applies_tenant_and_distance_threshold() -> None:
    near = VectorCandidate(
        chunk_id="near",
        content='{"equipment_id":"OLT-9000"}',
        metadata={
            "source": "/references/near.png",
            "title": "OLT Proprietária",
            "equipment_id": "OLT-9000",
            "equipment_name": "OLT Proprietária",
        },
        embedding=[1.0, 0.0],
        distance=0.2,
    )
    far = VectorCandidate(
        chunk_id="far",
        content='{"equipment_id":"OTHER"}',
        metadata={
            "source": "/references/far.png",
            "title": "Outro",
            "equipment_id": "OTHER",
        },
        embedding=[0.0, 1.0],
        distance=1.2,
    )
    store = FakeVisualStore(candidates=[near, far])
    embedder = FakeVisualEmbedder([[1.0, 0.0]])
    retriever = VisualCatalogRetriever(
        vector_store=store,
        embedding_provider_factory=lambda: embedder,
        max_results=3,
        max_distance=0.8,
    )
    image: ImageInput = {
        "name": "query.png",
        "mime_type": "image/png",
        "data": base64.b64encode(b"query-image").decode(),
        "sha256": "c" * 64,
    }

    matches = retriever.retrieve(
        [image],
        filters={"tenant_id": "public"},
    )

    assert [match.chunk_id for match in matches] == ["visual-ref:near"]
    assert matches[0].metadata["equipment_id"] == "OLT-9000"
    assert matches[0].metadata["visual_distance"] == 0.2
    assert store.searches == [
        ([1.0, 0.0], 3, {"tenant_id": "public"})
    ]


def test_empty_catalog_does_not_load_visual_model() -> None:
    created = []
    retriever = VisualCatalogRetriever(
        vector_store=FakeVisualStore(count=0),
        embedding_provider_factory=lambda: created.append(True),
        max_results=3,
        max_distance=0.8,
    )

    image: ImageInput = {
        "name": "query.png",
        "mime_type": "image/png",
        "data": base64.b64encode(b"query-image").decode(),
        "sha256": "d" * 64,
    }

    assert retriever.retrieve([image]) == []
    assert created == []


def test_visual_index_cli_requires_equipment_identity() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "/documents/front.png",
            "--equipment-id",
            "OLT-9000",
            "--equipment-name",
            "OLT Proprietária",
        ]
    )

    assert args.equipment_id == "OLT-9000"
    assert args.equipment_name == "OLT Proprietária"
