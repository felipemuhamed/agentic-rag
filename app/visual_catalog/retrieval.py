import base64
import json
from collections.abc import Callable, Sequence
from threading import Lock
from typing import Any

from app.knowledge.retrieval.models import RetrievedDocument
from app.knowledge.vectorstore.base import KnowledgeVectorStore
from app.multimodal.models import ImageInput
from app.visual_catalog.embeddings import VisualEmbeddingProvider


class VisualCatalogRetriever:
    def __init__(
        self,
        *,
        vector_store: KnowledgeVectorStore,
        embedding_provider_factory: Callable[[], VisualEmbeddingProvider],
        max_results: int,
        max_distance: float,
    ):
        if max_results < 1:
            raise ValueError("max_results must be at least one")
        if max_distance < 0:
            raise ValueError("max_distance must not be negative")
        self._vector_store = vector_store
        self._embedding_provider_factory = embedding_provider_factory
        self._embedding_provider: VisualEmbeddingProvider | None = None
        self._max_results = max_results
        self._max_distance = max_distance
        self._lock = Lock()

    def retrieve(
        self,
        images: Sequence[ImageInput],
        *,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedDocument]:
        if not images or self._vector_store.count() == 0:
            return []
        provider = self._provider()
        raw_images = [
            base64.b64decode(image["data"], validate=True)
            for image in images
        ]
        embeddings = provider.embed_images(raw_images)
        if len(embeddings) != len(images):
            raise RuntimeError("visual embedder returned an unexpected count")

        candidates_by_id = {}
        for embedding in embeddings:
            for candidate in self._vector_store.similarity_search(
                embedding,
                limit=self._max_results,
                filters=filters,
            ):
                if candidate.distance > self._max_distance:
                    continue
                previous = candidates_by_id.get(candidate.chunk_id)
                if previous is None or candidate.distance < previous.distance:
                    candidates_by_id[candidate.chunk_id] = candidate

        candidates = sorted(
            candidates_by_id.values(),
            key=lambda candidate: candidate.distance,
        )[: self._max_results]
        return [
            RetrievedDocument(
                chunk_id=f"visual-ref:{candidate.chunk_id}",
                content=(
                    "Referência visual proprietária recuperada por similaridade.\n"
                    "Trate como candidato, não como identificação definitiva, "
                    "e compare as demais evidências.\n"
                    f"Distância visual: {candidate.distance:.6f}\n"
                    f"Metadados: {json.dumps(candidate.metadata, ensure_ascii=False)}\n"
                    f"Descrição cadastrada: {candidate.content}"
                ),
                metadata={
                    **candidate.metadata,
                    "visual_distance": float(candidate.distance),
                },
                retrieval_score=-float(candidate.distance),
                rerank_score=max(0.0, 1.0 - float(candidate.distance)),
            )
            for candidate in candidates
        ]

    def _provider(self) -> VisualEmbeddingProvider:
        with self._lock:
            if self._embedding_provider is None:
                self._embedding_provider = self._embedding_provider_factory()
            return self._embedding_provider


def visual_matches_context(matches: Sequence[RetrievedDocument]) -> str:
    return json.dumps(
        [
            {
                "equipment_id": match.metadata.get("equipment_id"),
                "equipment_name": match.metadata.get("equipment_name"),
                "manufacturer": match.metadata.get("manufacturer"),
                "model": match.metadata.get("model"),
                "description": match.metadata.get("description"),
                "visual_distance": match.metadata.get("visual_distance"),
            }
            for match in matches
        ],
        ensure_ascii=False,
        indent=2,
    )
