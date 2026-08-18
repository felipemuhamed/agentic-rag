from app.knowledge.models import VectorCandidate
from app.knowledge.retrieval.mmr import maximal_marginal_relevance


def candidate(chunk_id: str, embedding: list[float]) -> VectorCandidate:
    return VectorCandidate(
        chunk_id=chunk_id,
        content=chunk_id,
        metadata={"document_id": chunk_id},
        embedding=embedding,
        distance=0.0,
    )


def test_mmr_prefers_diversity_after_most_relevant_candidate() -> None:
    candidates = [
        candidate("oauth-primary", [1.0, 0.0]),
        candidate("oauth-duplicate", [0.999, 0.001]),
        candidate("oauth-diverse", [0.7, 0.7]),
    ]

    selected = maximal_marginal_relevance(
        [1.0, 0.0],
        candidates,
        limit=2,
        lambda_mult=0.4,
    )

    assert [item.chunk_id for item in selected] == [
        "oauth-primary",
        "oauth-diverse",
    ]
