from app.evaluation.metrics import (
    evaluate_answer,
    recall_at_k,
    reciprocal_rank,
    token_f1,
)
from app.evaluation.models import ExpectedDocument, GenerationUsage
from app.knowledge.retrieval.models import RetrievedDocument
from app.sources.normalizer import normalize_rag_sources


def make_document(chunk_id: str, source: str, content: str) -> RetrievedDocument:
    return RetrievedDocument(
        chunk_id=chunk_id,
        content=content,
        metadata={"document_id": source, "source": f"/repo/{source}"},
        retrieval_score=0.8,
        rerank_score=0.9,
    )


def test_recall_and_mrr_use_expected_documents_not_unstable_chunk_ids() -> None:
    documents = [
        make_document("chunk-a", "docs/distractor.md", "Distractor"),
        make_document("chunk-b", "docs/expected.md", "Expected answer"),
        make_document("chunk-c", "docs/expected.md", "Another chunk"),
    ]
    expected = [ExpectedDocument(source="docs/expected.md")]

    assert recall_at_k(documents, expected, 1) == 0.0
    assert recall_at_k(documents, expected, 2) == 1.0
    assert reciprocal_rank(documents, expected) == 0.5


def test_answer_metrics_measure_relevance_grounding_and_citations() -> None:
    document = make_document(
        "chunk-1",
        "docs/oauth.md",
        "O refresh token renova o access token.",
    )
    source = normalize_rag_sources([document])[0].to_public()

    metrics = evaluate_answer(
        answer="O refresh token renova o access token [1].",
        reference_answer="O refresh token renova o access token.",
        documents=[document],
        public_sources=[source],
        expected_documents=[ExpectedDocument(source="docs/oauth.md")],
        latency_ms=2.0,
        usage=GenerationUsage(input_tokens=10, output_tokens=5, total_tokens=15),
    )

    assert metrics.relevance == 1.0
    assert metrics.grounding == 1.0
    assert metrics.citation_validity == 1.0
    assert metrics.expected_source_recall == 1.0


def test_unknown_citation_fails_grounding_closed() -> None:
    document = make_document("chunk-1", "docs/oauth.md", "Supported answer")
    source = normalize_rag_sources([document])[0].to_public()

    metrics = evaluate_answer(
        answer="Unsupported citation [2].",
        reference_answer="Supported answer.",
        documents=[document],
        public_sources=[source],
        expected_documents=[ExpectedDocument(source="docs/oauth.md")],
        latency_ms=1.0,
        usage=GenerationUsage(),
    )

    assert metrics.citation_validity == 0.0
    assert metrics.grounding == 0.0


def test_citation_with_mismatched_source_fails_closed() -> None:
    document = make_document("chunk-1", "docs/oauth.md", "Supported answer")
    source = normalize_rag_sources([document])[0].to_public()
    source["source"] = "/repo/docs/another.md"

    metrics = evaluate_answer(
        answer="Supported answer [1].",
        reference_answer="Supported answer.",
        documents=[document],
        public_sources=[source],
        expected_documents=[ExpectedDocument(source="docs/oauth.md")],
        latency_ms=1.0,
        usage=GenerationUsage(),
    )

    assert metrics.citation_validity == 0.0
    assert metrics.grounding == 0.0


def test_lexical_relevance_is_accent_and_citation_insensitive() -> None:
    assert token_f1("A configuração está válida [1].", "Configuracao valida") == 1.0
