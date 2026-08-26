import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from app.knowledge.models import Chunk, stable_chunk_id, stable_document_id
from app.knowledge.vectorstore.base import KnowledgeVectorStore
from app.visual_catalog.embeddings import VisualEmbeddingProvider


SUPPORTED_REFERENCE_IMAGES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


@dataclass(frozen=True, slots=True)
class VisualReference:
    equipment_id: str
    equipment_name: str
    manufacturer: str = ""
    model: str = ""
    description: str = ""
    tenant_id: str = ""

    def __post_init__(self) -> None:
        for field_name in ("equipment_id", "equipment_name"):
            value = getattr(self, field_name).strip()
            if not value:
                raise ValueError(f"{field_name} must not be blank")
            object.__setattr__(self, field_name, value)
        for field_name in ("manufacturer", "model", "description", "tenant_id"):
            object.__setattr__(self, field_name, getattr(self, field_name).strip())


@dataclass(frozen=True, slots=True)
class VisualIngestionResult:
    document_id: str
    reference_id: str
    equipment_id: str


class VisualReferenceIngestionService:
    def __init__(
        self,
        *,
        embedding_provider: VisualEmbeddingProvider,
        vector_store: KnowledgeVectorStore,
    ):
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

    def ingest_path(
        self,
        path: str | Path,
        reference: VisualReference,
    ) -> VisualIngestionResult:
        resolved_path = Path(path).expanduser().resolve()
        try:
            media_type = SUPPORTED_REFERENCE_IMAGES[resolved_path.suffix.lower()]
        except KeyError as error:
            supported = ", ".join(sorted(SUPPORTED_REFERENCE_IMAGES))
            raise ValueError(
                f"unsupported reference image; expected one of: {supported}"
            ) from error
        image_bytes = resolved_path.read_bytes()
        if not image_bytes:
            raise ValueError("reference image must not be empty")

        identity = (
            f"{reference.tenant_id}\0{reference.equipment_id}\0{resolved_path}"
        )
        document_id = stable_document_id(identity)
        content = json.dumps(
            {
                "equipment_id": reference.equipment_id,
                "equipment_name": reference.equipment_name,
                "manufacturer": reference.manufacturer,
                "model": reference.model,
                "description": reference.description,
            },
            ensure_ascii=False,
        )
        reference_id = stable_chunk_id(document_id, 0, content)
        metadata = {
            "document_id": document_id,
            "equipment_id": reference.equipment_id,
            "equipment_name": reference.equipment_name,
            "source": str(resolved_path),
            "title": reference.equipment_name,
            "media_type": media_type,
            "image_sha256": sha256(image_bytes).hexdigest(),
            "chunk_index": 0,
            "reference_type": "equipment_image",
        }
        if reference.tenant_id:
            metadata["tenant_id"] = reference.tenant_id
        for field_name in ("manufacturer", "model", "description"):
            value = getattr(reference, field_name)
            if value:
                metadata[field_name] = value
        embeddings = self._embedding_provider.embed_images([image_bytes])
        if len(embeddings) != 1:
            raise RuntimeError("visual embedder returned an unexpected count")
        self._vector_store.replace_document(
            document_id,
            [Chunk(chunk_id=reference_id, content=content, metadata=metadata)],
            embeddings,
        )
        return VisualIngestionResult(
            document_id=document_id,
            reference_id=reference_id,
            equipment_id=reference.equipment_id,
        )
