from types import SimpleNamespace

import app.knowledge.vectorstore.chroma as chroma_module
from app.knowledge.vectorstore.chroma import ChromaVectorStore


class FakeCollection:
    pass


class FakeHttpClient:
    def __init__(self, **kwargs: object):
        self.kwargs = kwargs
        self.collection_call: dict[str, object] = {}

    def get_or_create_collection(self, **kwargs: object) -> FakeCollection:
        self.collection_call = kwargs
        return FakeCollection()


def test_http_client_uses_server_connection_settings(monkeypatch) -> None:
    client = FakeHttpClient()
    fake_chromadb = SimpleNamespace(HttpClient=lambda **kwargs: _configure(client, kwargs))
    monkeypatch.setattr(chroma_module, "_import_chromadb", lambda: fake_chromadb)

    store = ChromaVectorStore.http(
        "knowledge",
        host="chroma",
        port=8000,
        ssl=False,
        tenant="default_tenant",
        database="default_database",
    )

    assert store._client is client
    assert client.kwargs == {
        "host": "chroma",
        "port": 8000,
        "ssl": False,
        "tenant": "default_tenant",
        "database": "default_database",
    }
    assert client.collection_call == {
        "name": "knowledge",
        "embedding_function": None,
    }


def _configure(client: FakeHttpClient, kwargs: dict[str, object]) -> FakeHttpClient:
    client.kwargs = kwargs
    return client
