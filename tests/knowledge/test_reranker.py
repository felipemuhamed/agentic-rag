import sys
from types import SimpleNamespace

from app.knowledge.retrieval.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    instance: "FakeCrossEncoder"

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self.pairs: list[tuple[str, str]] = []
        FakeCrossEncoder.instance = self

    def predict(
        self,
        pairs: list[tuple[str, str]],
        show_progress_bar: bool,
    ) -> list[float]:
        assert show_progress_bar is False
        self.pairs = pairs
        return [-2.0, 4.5]


def test_cross_encoder_scores_query_document_pairs(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(CrossEncoder=FakeCrossEncoder),
    )
    reranker = CrossEncoderReranker("reranker/test", device="cpu")

    scores = reranker.score("query", ["first", "second"])

    assert scores == [-2.0, 4.5]
    assert FakeCrossEncoder.instance.model_name == "reranker/test"
    assert FakeCrossEncoder.instance.device == "cpu"
    assert FakeCrossEncoder.instance.pairs == [
        ("query", "first"),
        ("query", "second"),
    ]
