from typing import Any

from app.knowledge.embeddings.base import EmbeddingProvider
from app.knowledge.retrieval.mmr import maximal_marginal_relevance
from app.knowledge.retrieval.models import (
    RetrievalDiagnostics,
    RetrievalResult,
    RetrievedDocument,
)
from app.knowledge.retrieval.reranker import Reranker
from app.knowledge.vectorstore.base import KnowledgeVectorStore


class KnowledgeRetriever:
    def __init__(
        self,
        *,
        vector_store: KnowledgeVectorStore,
        embedding_provider: EmbeddingProvider,
        reranker: Reranker,
        fetch_k: int,
        candidate_k: int,
        final_k: int,
        lambda_mult: float,
    ):
        _validate_configuration(fetch_k, candidate_k, final_k, lambda_mult)
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._reranker = reranker
        self.fetch_k = fetch_k
        self.candidate_k = candidate_k
        self.final_k = final_k
        self.lambda_mult = lambda_mult

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")

        query_embedding = self._embedding_provider.embed_query(normalized_query)
        fetched = self._vector_store.similarity_search(
            query_embedding,
            limit=self.fetch_k,
            filters=filters,
        )
        selected = maximal_marginal_relevance(
            query_embedding,
            fetched,
            limit=self.candidate_k,
            lambda_mult=self.lambda_mult,
        )

        scores = self._reranker.score(
            normalized_query,
            [candidate.content for candidate in selected],
        )
        if len(scores) != len(selected):
            raise RuntimeError("reranker returned an unexpected number of scores")

        ranked = sorted(
            zip(selected, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        documents = [
            RetrievedDocument(
                chunk_id=candidate.chunk_id,
                content=candidate.content,
                metadata=dict(candidate.metadata),
                retrieval_score=-candidate.distance,
                rerank_score=float(score),
            )
            for candidate, score in ranked[: self.final_k]
        ]
        return RetrievalResult(
            documents=documents,
            diagnostics=RetrievalDiagnostics(
                fetched_count=len(fetched),
                mmr_selected_count=len(selected),
                reranked_count=len(scores),
                returned_count=len(documents),
            ),
        )


def _validate_configuration(
    fetch_k: int,
    candidate_k: int,
    final_k: int,
    lambda_mult: float,
) -> None:
    if not fetch_k >= candidate_k >= final_k > 0:
        raise ValueError("expected fetch_k >= candidate_k >= final_k > 0")
    if not 0.0 <= lambda_mult <= 1.0:
        raise ValueError("lambda_mult must be between 0 and 1")
