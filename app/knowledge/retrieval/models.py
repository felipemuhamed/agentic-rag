from dataclasses import dataclass

from app.knowledge.models import MetadataValue


@dataclass(frozen=True, slots=True)
class RetrievedDocument:
    chunk_id: str
    content: str
    metadata: dict[str, MetadataValue]
    retrieval_score: float
    rerank_score: float


@dataclass(frozen=True, slots=True)
class RetrievalDiagnostics:
    fetched_count: int
    mmr_selected_count: int
    reranked_count: int
    returned_count: int


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    documents: list[RetrievedDocument]
    diagnostics: RetrievalDiagnostics
