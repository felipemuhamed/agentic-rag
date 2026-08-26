from dataclasses import dataclass, replace
from hashlib import sha256
from typing import TypeAlias


MetadataValue: TypeAlias = str | int | float | bool


@dataclass(frozen=True, slots=True)
class SourceDocument:
    document_id: str
    title: str
    source: str
    content: str
    created_at: str
    updated_at: str
    section: str = ""
    page: int | None = None
    url: str | None = None
    media_type: str = "text/plain"
    tenant_id: str = ""
    equipment_id: str = ""

    def with_content(self, content: str) -> "SourceDocument":
        return replace(self, content=content)


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    content: str
    metadata: dict[str, MetadataValue]


@dataclass(frozen=True, slots=True)
class VectorCandidate:
    chunk_id: str
    content: str
    metadata: dict[str, MetadataValue]
    embedding: list[float]
    distance: float


def stable_document_id(source: str) -> str:
    return sha256(source.encode("utf-8")).hexdigest()


def stable_chunk_id(document_id: str, chunk_index: int, content: str) -> str:
    identity = f"{document_id}:{chunk_index}:{content}"
    return sha256(identity.encode("utf-8")).hexdigest()
