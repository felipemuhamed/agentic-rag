import sys
from types import SimpleNamespace

from app.knowledge.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddingProvider,
)


class FakeArray:
    def tolist(self) -> list[list[float]]:
        return [[1.0, 0.0], [0.0, 1.0]]


class FakeSentenceTransformer:
    instance: "FakeSentenceTransformer"

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.device = device
        self.call: dict[str, object] = {}
        FakeSentenceTransformer.instance = self

    def encode_document(self, texts: list[str], **kwargs: object) -> FakeArray:
        self.call = {"texts": texts, **kwargs}
        return FakeArray()

    def encode_query(self, text: str, **kwargs: object) -> "FakeVector":
        self.call = {"text": text, **kwargs}
        return FakeVector()


class FakeVector:
    def tolist(self) -> list[float]:
        return [0.25, 0.75]


def test_sentence_transformer_provider_uses_document_embeddings(monkeypatch) -> None:
    fake_module = SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    provider = SentenceTransformerEmbeddingProvider("model/test", device="cpu")
    result = provider.embed_documents(["first", "second"])

    assert result == [[1.0, 0.0], [0.0, 1.0]]
    assert FakeSentenceTransformer.instance.model_name == "model/test"
    assert FakeSentenceTransformer.instance.device == "cpu"
    assert FakeSentenceTransformer.instance.call == {
        "texts": ["first", "second"],
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }


def test_sentence_transformer_provider_uses_query_embeddings(monkeypatch) -> None:
    fake_module = SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    provider = SentenceTransformerEmbeddingProvider("model/test")

    result = provider.embed_query("oauth token")

    assert result == [0.25, 0.75]
    assert FakeSentenceTransformer.instance.call == {
        "text": "oauth token",
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }
