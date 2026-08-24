import asyncio
import json

import httpx

from app.api.chat import get_rag_stream_adapter
from app.main import create_app
from app.streaming.events import StreamEvent


class RecordingAdapter:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def stream(self, *, conversation_id: str, message: str):
        self.calls.append((conversation_id, message))
        yield StreamEvent(
            "status",
            {"stage": "rag_search", "message": "Buscando..."},
        )
        yield StreamEvent(
            "sources",
            {
                "sources": [
                    {
                        "id": 1,
                        "citation": "[1]",
                        "type": "rag",
                        "title": "Guia OAuth",
                        "source": "oauth.md",
                        "chunkId": "oauth-1",
                        "url": None,
                        "page": 3,
                        "section": "Tokens",
                    }
                ]
            },
        )
        yield StreamEvent("token", {"text": "Resposta [1]."})
        yield StreamEvent("done", {})


def parse_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    parsed = []
    for block in body.strip().split("\n\n"):
        lines = block.splitlines()
        event = lines[0].removeprefix("event: ")
        data = json.loads(lines[1].removeprefix("data: "))
        parsed.append((event, data))
    return parsed


def test_chat_stream_exposes_stable_sse_contract() -> None:
    adapter = RecordingAdapter()
    application = create_app()
    application.dependency_overrides[get_rag_stream_adapter] = lambda: adapter

    async def make_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/chat/stream",
                json={
                    "conversationId": " conversation-1 ",
                    "message": " Como renovar o token? ",
                },
            )

    response = asyncio.run(make_request())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert adapter.calls == [("conversation-1", "Como renovar o token?")]
    assert parse_sse(response.text) == [
        ("status", {"stage": "rag_search", "message": "Buscando..."}),
        (
            "sources",
            {
                "sources": [
                    {
                        "id": 1,
                        "citation": "[1]",
                        "type": "rag",
                        "title": "Guia OAuth",
                        "source": "oauth.md",
                        "chunkId": "oauth-1",
                        "url": None,
                        "page": 3,
                        "section": "Tokens",
                    }
                ]
            },
        ),
        ("token", {"text": "Resposta [1]."}),
        ("done", {}),
    ]


def test_chat_stream_rejects_blank_message_before_execution() -> None:
    adapter = RecordingAdapter()
    application = create_app()
    application.dependency_overrides[get_rag_stream_adapter] = lambda: adapter

    async def make_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/chat/stream",
                json={"conversationId": "conversation-1", "message": "   "},
            )

    response = asyncio.run(make_request())

    assert response.status_code == 422
    assert adapter.calls == []
