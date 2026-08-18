from typing import Any

from app.knowledge.models import VectorCandidate
from app.knowledge.retrieval.retriever import KnowledgeRetriever


class FakeEmbeddingProvider:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise AssertionError("retrieval must not embed documents")

    def embed_query(self, text: str) -> list[float]:
        assert text == "How are OAuth tokens renewed?"
        return [1.0, 0.0]


class FakeVectorStore:
    def __init__(self):
        self.filters: dict[str, Any] | None = None

    def similarity_search(
        self,
        query_embedding: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorCandidate]:
        assert query_embedding == [1.0, 0.0]
        assert limit == 4
        self.filters = filters
        return [
            make_candidate("dense-first", "OAuth overview", [1.0, 0.0], 0.01),
            make_candidate("dense-second", "Refresh tokens renew access", [0.9, 0.1], 0.05),
            make_candidate("dense-third", "Token validation", [0.7, 0.7], 0.2),
        ]

    def replace_document(self, *args: object, **kwargs: object) -> int:
        raise AssertionError("retrieval must not write")

    def count(self) -> int:
        return 3


class FakeReranker:
    def score(self, query: str, documents: list[str]) -> list[float]:
        assert query == "How are OAuth tokens renewed?"
        return [
            0.1 if document == "OAuth overview" else 0.9
            for document in documents
        ]


def make_candidate(
    chunk_id: str,
    content: str,
    embedding: list[float],
    distance: float,
) -> VectorCandidate:
    return VectorCandidate(
        chunk_id=chunk_id,
        content=content,
        metadata={"document_id": "oauth", "section": "Tokens"},
        embedding=embedding,
        distance=distance,
    )


def test_retriever_applies_dense_mmr_and_reranking() -> None:
    store = FakeVectorStore()
    retriever = KnowledgeRetriever(
        vector_store=store,
        embedding_provider=FakeEmbeddingProvider(),
        reranker=FakeReranker(),
        fetch_k=4,
        candidate_k=3,
        final_k=2,
        lambda_mult=0.7,
    )

    result = retriever.retrieve(
        "  How are OAuth tokens renewed?  ",
        filters={"section": "Tokens"},
    )

    assert store.filters == {"section": "Tokens"}
    assert [document.chunk_id for document in result.documents] == [
        "dense-second",
        "dense-third",
    ]
    assert result.documents[0].rerank_score == 0.9
    assert result.documents[0].retrieval_score == -0.05
    assert result.diagnostics.fetched_count == 3
    assert result.diagnostics.mmr_selected_count == 3
    assert result.diagnostics.returned_count == 2


def test_retriever_rejects_invalid_k_configuration() -> None:
    try:
        KnowledgeRetriever(
            vector_store=FakeVectorStore(),
            embedding_provider=FakeEmbeddingProvider(),
            reranker=FakeReranker(),
            fetch_k=2,
            candidate_k=3,
            final_k=1,
            lambda_mult=0.5,
        )
    except ValueError as exc:
        assert "fetch_k" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
