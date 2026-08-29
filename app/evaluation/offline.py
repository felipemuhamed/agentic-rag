from hashlib import sha256
from math import sqrt
from pathlib import Path
from typing import Any

from app.evaluation.metrics import lexical_tokens
from app.knowledge.ingestion.chunker import TextChunker
from app.knowledge.ingestion.loaders import DocumentFileLoader
from app.knowledge.ingestion.normalizer import TextNormalizer
from app.knowledge.ingestion.service import IngestionService
from app.knowledge.models import Chunk, VectorCandidate
from app.knowledge.retrieval.mmr import cosine_similarity
from app.knowledge.retrieval.retriever import KnowledgeRetriever


OFFLINE_EMBEDDING_MODEL = "evaluation:hashing-384-v1"
OFFLINE_RERANKER_MODEL = "evaluation:token-overlap-v1"


class HashingEmbeddingProvider:
    def __init__(self, dimensions: int = 384):
        if dimensions < 1:
            raise ValueError("dimensions must be at least one")
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in lexical_tokens(text):
            digest = sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        norm = sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class InMemoryEvaluationVectorStore:
    def __init__(self):
        self._candidates: dict[str, VectorCandidate] = {}

    def replace_document(
        self,
        document_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self._candidates = {
            chunk_id: candidate
            for chunk_id, candidate in self._candidates.items()
            if candidate.metadata.get("document_id") != document_id
        }
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self._candidates[chunk.chunk_id] = VectorCandidate(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                metadata=dict(chunk.metadata),
                embedding=list(embedding),
                distance=0.0,
            )
        return len(chunks)

    def count(self) -> int:
        return len(self._candidates)

    def similarity_search(
        self,
        query_embedding: list[float],
        limit: int,
        filters: dict[str, Any] | None = None,
    ) -> list[VectorCandidate]:
        if limit < 1:
            raise ValueError("limit must be at least one")
        candidates = [
            candidate
            for candidate in self._candidates.values()
            if _matches_filters(candidate.metadata, filters)
        ]
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                -cosine_similarity(query_embedding, candidate.embedding),
                candidate.chunk_id,
            ),
        )
        return [
            VectorCandidate(
                chunk_id=candidate.chunk_id,
                content=candidate.content,
                metadata=dict(candidate.metadata),
                embedding=list(candidate.embedding),
                distance=1.0
                - cosine_similarity(query_embedding, candidate.embedding),
            )
            for candidate in ranked[:limit]
        ]


class TokenOverlapReranker:
    def score(self, query: str, documents: list[str]) -> list[float]:
        query_tokens = set(lexical_tokens(query))
        return [
            _overlap_score(query_tokens, set(lexical_tokens(document)))
            for document in documents
        ]


def create_offline_retriever(
    *,
    corpus_paths: list[Path],
    chunk_size: int,
    chunk_overlap: int,
    fetch_k: int,
    candidate_k: int,
    final_k: int,
    lambda_mult: float,
) -> KnowledgeRetriever:
    embeddings = HashingEmbeddingProvider()
    vector_store = InMemoryEvaluationVectorStore()
    ingestion = IngestionService(
        loader=DocumentFileLoader(),
        normalizer=TextNormalizer(),
        chunker=TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
        embedding_provider=embeddings,
        vector_store=vector_store,
    )
    for path in corpus_paths:
        ingestion.ingest_path(path)
    return KnowledgeRetriever(
        vector_store=vector_store,
        embedding_provider=embeddings,
        reranker=TokenOverlapReranker(),
        fetch_k=fetch_k,
        candidate_k=candidate_k,
        final_k=final_k,
        lambda_mult=lambda_mult,
    )


def _matches_filters(
    metadata: dict[str, object],
    filters: dict[str, Any] | None,
) -> bool:
    if not filters:
        return True
    return all(metadata.get(key) == value for key, value in filters.items())


def _overlap_score(query_tokens: set[str], document_tokens: set[str]) -> float:
    if not query_tokens or not document_tokens:
        return 0.0
    return len(query_tokens & document_tokens) / sqrt(
        len(query_tokens) * len(document_tokens)
    )
