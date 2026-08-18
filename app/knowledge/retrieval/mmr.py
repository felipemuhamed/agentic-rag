from math import sqrt

from app.knowledge.models import VectorCandidate


def maximal_marginal_relevance(
    query_embedding: list[float],
    candidates: list[VectorCandidate],
    limit: int,
    lambda_mult: float,
) -> list[VectorCandidate]:
    if limit <= 0 or not candidates:
        return []
    if not 0.0 <= lambda_mult <= 1.0:
        raise ValueError("lambda_mult must be between 0 and 1")

    selected_indexes: list[int] = []
    remaining_indexes = list(range(len(candidates)))

    while remaining_indexes and len(selected_indexes) < limit:
        best_index = max(
            remaining_indexes,
            key=lambda index: (
                _mmr_score(
                    query_embedding,
                    candidates,
                    index,
                    selected_indexes,
                    lambda_mult,
                ),
                -index,
            ),
        )
        selected_indexes.append(best_index)
        remaining_indexes.remove(best_index)

    return [candidates[index] for index in selected_indexes]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return sum(a * b for a, b in zip(left, right, strict=True)) / (
        left_norm * right_norm
    )


def _mmr_score(
    query_embedding: list[float],
    candidates: list[VectorCandidate],
    candidate_index: int,
    selected_indexes: list[int],
    lambda_mult: float,
) -> float:
    candidate_embedding = candidates[candidate_index].embedding
    relevance = cosine_similarity(query_embedding, candidate_embedding)
    if not selected_indexes:
        return relevance

    redundancy = max(
        cosine_similarity(candidate_embedding, candidates[index].embedding)
        for index in selected_indexes
    )
    return (lambda_mult * relevance) - ((1.0 - lambda_mult) * redundancy)
