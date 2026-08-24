from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ExpectedDocument(StrictModel):
    document_id: str | None = None
    source: str | None = None

    @field_validator("document_id", "source")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("expected document identifiers must not be blank")
        return normalized

    @model_validator(mode="after")
    def require_identifier(self) -> "ExpectedDocument":
        if self.document_id is None and self.source is None:
            raise ValueError("expected document needs document_id or source")
        return self


class EvaluationCase(StrictModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_documents: list[ExpectedDocument] = Field(min_length=1)
    reference_answer: str = Field(min_length=1)
    filters: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("id", "question", "reference_answer")
    @classmethod
    def normalize_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("evaluation text fields must not be blank")
        return normalized

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("evaluation tags must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("evaluation tags must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_unique_expected_documents(self) -> "EvaluationCase":
        identifiers = [
            (document.document_id, document.source)
            for document in self.expected_documents
        ]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("expected documents must be unique within a case")
        return self


class EvaluationDataset(StrictModel):
    schema_version: Literal["1.0"]
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    corpus: list[str] = Field(min_length=1)
    cases: list[EvaluationCase] = Field(min_length=1)

    @field_validator("name", "version", "description")
    @classmethod
    def normalize_dataset_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("dataset text fields must not be blank")
        return normalized

    @field_validator("corpus")
    @classmethod
    def normalize_corpus(cls, values: list[str]) -> list[str]:
        normalized = [value.strip().replace("\\", "/") for value in values]
        if any(not value for value in normalized):
            raise ValueError("corpus paths must not be blank")
        if any(
            value.startswith("/") or ".." in value.split("/")
            for value in normalized
        ):
            raise ValueError("corpus paths must be relative and remain below the root")
        if len(normalized) != len(set(normalized)):
            raise ValueError("corpus paths must be unique")
        return normalized

    @model_validator(mode="after")
    def validate_unique_cases_and_sources(self) -> "EvaluationDataset":
        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("evaluation case IDs must be unique")

        corpus = set(self.corpus)
        unknown_sources = {
            expected.source
            for case in self.cases
            for expected in case.expected_documents
            if expected.source is not None and expected.source not in corpus
        }
        if unknown_sources:
            unknown = ", ".join(sorted(unknown_sources))
            raise ValueError(f"expected sources are absent from corpus: {unknown}")
        return self


class Pricing(StrictModel):
    currency: str = "USD"
    input_per_million: float = Field(ge=0)
    output_per_million: float = Field(ge=0)


class EvaluationConfiguration(StrictModel):
    name: str = Field(min_length=1)
    mode: Literal["live", "offline"]
    app_version: str
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    embedding_model: str
    reranker_model: str
    fetch_k: int = Field(gt=0)
    candidate_k: int = Field(gt=0)
    final_k: int = Field(gt=0)
    lambda_mult: float = Field(ge=0, le=1)
    collection_name: str | None = None
    llm_model: str | None = None
    llm_temperature: float | None = None
    pricing: Pricing | None = None

    @model_validator(mode="after")
    def validate_retrieval_limits(self) -> "EvaluationConfiguration":
        if not self.fetch_k >= self.candidate_k >= self.final_k:
            raise ValueError("expected fetch_k >= candidate_k >= final_k")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class DatasetIdentity(StrictModel):
    name: str
    version: str
    sha256: str
    corpus_sha256: str
    cases: int = Field(gt=0)


class RetrievedHit(StrictModel):
    rank: int = Field(gt=0)
    chunk_id: str
    document_id: str | None
    source: str | None
    retrieval_score: float
    rerank_score: float


class RetrievalCaseMetrics(StrictModel):
    recall_at_k: dict[str, float]
    reciprocal_rank: float = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)


class GenerationUsage(StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class AnswerCaseMetrics(StrictModel):
    relevance: float = Field(ge=0, le=1)
    grounding: float = Field(ge=0, le=1)
    citation_validity: float = Field(ge=0, le=1)
    expected_source_recall: float = Field(ge=0, le=1)
    latency_ms: float = Field(ge=0)
    usage: GenerationUsage


class EvaluationCaseReport(StrictModel):
    id: str
    question: str
    retrieved: list[RetrievedHit]
    retrieval: RetrievalCaseMetrics
    answer: str | None = None
    answer_metrics: AnswerCaseMetrics | None = None


class LatencySummary(StrictModel):
    mean_ms: float = Field(ge=0)
    p50_ms: float = Field(ge=0)
    p95_ms: float = Field(ge=0)


class RetrievalSummary(StrictModel):
    recall_at_k: dict[str, float]
    mrr: float = Field(ge=0, le=1)
    latency: LatencySummary


class UsageSummary(StrictModel):
    calls: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    complete: bool
    estimated_cost: float | None = Field(default=None, ge=0)
    currency: str | None = None


class AnswerSummary(StrictModel):
    evaluated_cases: int = Field(ge=0)
    relevance: float = Field(ge=0, le=1)
    grounding: float = Field(ge=0, le=1)
    citation_validity: float = Field(ge=0, le=1)
    expected_source_recall: float = Field(ge=0, le=1)
    latency: LatencySummary
    usage: UsageSummary


class EvaluationSummary(StrictModel):
    retrieval: RetrievalSummary
    answers: AnswerSummary | None = None


class EvaluationReport(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    generated_at: str
    dataset: DatasetIdentity
    configuration: EvaluationConfiguration
    summary: EvaluationSummary
    cases: list[EvaluationCaseReport]
