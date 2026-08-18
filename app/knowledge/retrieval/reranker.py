from typing import Any, Protocol


class Reranker(Protocol):
    def score(self, query: str, documents: list[str]) -> list[float]:
        """Return one relevance score per document, preserving input order."""
        ...


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str | None = None):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Install the knowledge dependencies with "
                "`python -m pip install -e '.[knowledge]'`"
            ) from exc

        self.model_name = model_name
        self._model: Any = CrossEncoder(model_name, device=device)

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        scores = self._model.predict(
            [(query, document) for document in documents],
            show_progress_bar=False,
        )
        return [float(score) for score in scores]
