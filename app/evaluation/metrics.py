import re
import unicodedata
from collections.abc import Sequence

from app.evaluation.models import AnswerCaseMetrics, ExpectedDocument, GenerationUsage
from app.knowledge.retrieval.models import RetrievedDocument
from app.sources.citations import select_cited_sources
from app.sources.models import PublicSource, Source


TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)
CITATION_PATTERN = re.compile(r"\[\d+\]")
STOP_WORDS = {
    "a",
    "ao",
    "aos",
    "as",
    "com",
    "como",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "entre",
    "esta",
    "no",
    "nos",
    "o",
    "os",
    "ou",
    "para",
    "por",
    "que",
    "se",
    "um",
    "uma",
}


def recall_at_k(
    documents: Sequence[RetrievedDocument],
    expected: Sequence[ExpectedDocument],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be at least one")
    if not expected:
        raise ValueError("expected documents must not be empty")
    retrieved = documents[:k]
    matched = sum(
        any(document_matches(document, target) for document in retrieved)
        for target in expected
    )
    return matched / len(expected)


def reciprocal_rank(
    documents: Sequence[RetrievedDocument],
    expected: Sequence[ExpectedDocument],
) -> float:
    if not expected:
        raise ValueError("expected documents must not be empty")
    for rank, document in enumerate(documents, start=1):
        if any(document_matches(document, target) for target in expected):
            return 1.0 / rank
    return 0.0


def document_matches(
    document: RetrievedDocument,
    expected: ExpectedDocument,
) -> bool:
    document_id = document.metadata.get("document_id")
    if expected.document_id is not None and document_id == expected.document_id:
        return True
    source = document.metadata.get("source")
    return (
        expected.source is not None
        and isinstance(source, str)
        and source_matches(source, expected.source)
    )


def source_matches(actual: str, expected: str) -> bool:
    normalized_actual = actual.strip().replace("\\", "/")
    normalized_expected = expected.strip().replace("\\", "/").lstrip("./")
    return normalized_actual == normalized_expected or normalized_actual.endswith(
        f"/{normalized_expected}"
    )


def evaluate_answer(
    *,
    answer: str,
    reference_answer: str,
    documents: Sequence[RetrievedDocument],
    public_sources: Sequence[PublicSource],
    expected_documents: Sequence[ExpectedDocument],
    latency_ms: float,
    usage: GenerationUsage,
) -> AnswerCaseMetrics:
    relevance = token_f1(answer, reference_answer)
    parsed_sources: list[Source] = []
    citations_valid = False
    try:
        parsed_sources = [Source.from_public(source) for source in public_sources]
        source_ids = [source.citation_id for source in parsed_sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("citation source IDs must be unique")
        cited_sources = select_cited_sources(answer, parsed_sources)
        documents_by_chunk = {
            document.chunk_id: document for document in documents
        }
        if any(
            source.chunk_id not in documents_by_chunk
            or not _source_maps_to_document(
                source,
                documents_by_chunk[source.chunk_id],
            )
            for source in parsed_sources
        ):
            raise ValueError("citation source does not map to retrieved evidence")
        citations_valid = (
            bool(cited_sources)
            and {source.citation_id for source in cited_sources} == set(source_ids)
        )
    except ValueError:
        citations_valid = False

    cited_chunk_ids = {
        source.chunk_id for source in parsed_sources if source.chunk_id is not None
    }
    cited_content = " ".join(
        document.content
        for document in documents
        if document.chunk_id in cited_chunk_ids
    )
    grounding = token_precision(answer, cited_content) if citations_valid else 0.0

    matched_expected = sum(
        any(source_matches(source.source, expected.source) for source in parsed_sources)
        if expected.source is not None
        else any(
            source.chunk_id == document.chunk_id
            and document.metadata.get("document_id") == expected.document_id
            for source in parsed_sources
            for document in documents
        )
        for expected in expected_documents
    )
    expected_source_recall = matched_expected / len(expected_documents)

    return AnswerCaseMetrics(
        relevance=relevance,
        grounding=grounding,
        citation_validity=1.0 if citations_valid else 0.0,
        expected_source_recall=expected_source_recall,
        latency_ms=latency_ms,
        usage=usage,
    )


def token_f1(candidate: str, reference: str) -> float:
    candidate_tokens = lexical_tokens(candidate)
    reference_tokens = lexical_tokens(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    candidate_counts = _counts(candidate_tokens)
    reference_counts = _counts(reference_tokens)
    overlap = sum(
        min(count, reference_counts.get(token, 0))
        for token, count in candidate_counts.items()
    )
    if overlap == 0:
        return 0.0
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return (2 * precision * recall) / (precision + recall)


def token_precision(candidate: str, evidence: str) -> float:
    candidate_tokens = lexical_tokens(candidate)
    evidence_tokens = set(lexical_tokens(evidence))
    if not candidate_tokens or not evidence_tokens:
        return 0.0
    supported = sum(token in evidence_tokens for token in candidate_tokens)
    return supported / len(candidate_tokens)


def lexical_tokens(value: str) -> list[str]:
    without_citations = CITATION_PATTERN.sub(" ", value)
    normalized = unicodedata.normalize("NFKD", without_citations.casefold())
    ascii_like = "".join(
        character for character in normalized if not unicodedata.combining(character)
    )
    return [
        token
        for token in TOKEN_PATTERN.findall(ascii_like)
        if len(token) > 2 and token not in STOP_WORDS
    ]


def _counts(tokens: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    return counts


def _source_maps_to_document(
    source: Source,
    document: RetrievedDocument,
) -> bool:
    document_source = document.metadata.get("source")
    if not isinstance(document_source, str):
        return source.source == f"chunk:{document.chunk_id}"
    return source_matches(document_source, source.source)
