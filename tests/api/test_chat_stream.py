import asyncio
import base64
import json
from threading import Event
from types import SimpleNamespace

import httpx

from app.api.chat import _stream_sse, get_rag_stream_adapter
from app.main import create_app
from app.observability.metrics import MetricsRegistry
from app.security.access import AccessPolicy
from app.security.rate_limit import RateLimiter
from app.settings import Settings
from app.streaming.events import StreamEvent


class RecordingAdapter:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self.options: list[dict[str, object]] = []

    def stream(
        self,
        *,
        conversation_id: str,
        message: str,
        **options: object,
    ):
        self.calls.append((conversation_id, message))
        self.options.append(options)
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
    assert response.headers["x-request-id"]
    assert response.headers["x-ratelimit-limit"] == "60"
    assert adapter.calls == [("conversation-1", "Como renovar o token?")]
    assert len(adapter.options[0]["thread_id"]) == 64
    assert adapter.options[0]["retrieval_filters"] is None
    assert adapter.options[0]["images"] == []
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


def _png_base64(extra: bytes = b"") -> str:
    return base64.b64encode(b"\x89PNG\r\n\x1a\n" + extra).decode()


def test_chat_stream_accepts_valid_base64_image() -> None:
    adapter = RecordingAdapter()
    application = create_app(Settings(app_environment="test", _env_file=None))
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
                    "conversationId": "image-1",
                    "message": "Identifique este alarme.",
                    "images": [
                        {
                            "name": "../painel.png",
                            "mimeType": "image/png",
                            "data": _png_base64(),
                        }
                    ],
                },
            )

    response = asyncio.run(make_request())

    assert response.status_code == 200
    images = adapter.options[0]["images"]
    assert isinstance(images, list)
    assert images[0]["name"] == "painel.png"
    assert images[0]["mime_type"] == "image/png"
    assert len(images[0]["sha256"]) == 64


def test_chat_stream_rejects_invalid_or_disabled_image_input() -> None:
    adapter = RecordingAdapter()
    disabled_app = create_app(
        Settings(
            app_environment="test",
            multimodal_enabled=False,
            _env_file=None,
        )
    )
    disabled_app.dependency_overrides[get_rag_stream_adapter] = lambda: adapter
    payload = {
        "conversationId": "image-1",
        "message": "Analise.",
        "images": [
            {
                "name": "painel.png",
                "mimeType": "image/png",
                "data": _png_base64(),
            }
        ],
    }

    async def post(application, body) -> httpx.Response:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post("/chat/stream", json=body)

    disabled_response = asyncio.run(post(disabled_app, payload))
    assert disabled_response.status_code == 422

    enabled_app = create_app(Settings(app_environment="test", _env_file=None))
    enabled_app.dependency_overrides[get_rag_stream_adapter] = lambda: adapter
    invalid_payload = {
        **payload,
        "images": [{**payload["images"][0], "data": "not-base64"}],
    }
    invalid_response = asyncio.run(post(enabled_app, invalid_payload))

    assert invalid_response.status_code == 422
    assert adapter.calls == []


def test_chat_stream_rejects_image_larger_than_configured_limit() -> None:
    config = Settings(
        app_environment="test",
        multimodal_max_image_bytes=1024,
        _env_file=None,
    )
    adapter = RecordingAdapter()
    application = create_app(config)
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
                    "conversationId": "image-large",
                    "message": "Analise.",
                    "images": [
                        {
                            "name": "large.png",
                            "mimeType": "image/png",
                            "data": _png_base64(b"x" * 1024),
                        }
                    ],
                },
            )

    response = asyncio.run(make_request())

    assert response.status_code == 413
    assert adapter.calls == []


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


def test_chat_authenticates_key_and_applies_tenant_filter() -> None:
    policies = json.dumps(
        [
            {
                "key": "correct-key-with-32-characters-0001",
                "principal": "frontend",
                "tenants": ["tenant-a"],
            }
        ]
    )
    config = Settings(
        app_environment="test",
        auth_enabled=True,
        api_key_policies=policies,
        _env_file=None,
    )
    adapter = RecordingAdapter()
    application = create_app(config)
    application.dependency_overrides[get_rag_stream_adapter] = lambda: adapter

    async def make_request(api_key: str | None, tenant: str) -> httpx.Response:
        transport = httpx.ASGITransport(app=application)
        headers = {"X-API-Key": api_key} if api_key is not None else {}
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(
                "/chat/stream",
                headers=headers,
                json={
                    "conversationId": "conversation-1",
                    "tenantId": tenant,
                    "message": "Question",
                },
            )

    assert asyncio.run(make_request(None, "tenant-a")).status_code == 401
    assert (
        asyncio.run(make_request("wrong-key-with-32-characters-00000", "tenant-a"))
        .status_code
        == 401
    )
    assert (
        asyncio.run(
            make_request("correct-key-with-32-characters-0001", "tenant-b")
        ).status_code
        == 403
    )
    response = asyncio.run(
        make_request("correct-key-with-32-characters-0001", "tenant-a")
    )

    assert response.status_code == 200
    assert adapter.options[-1]["retrieval_filters"] == {"tenant_id": "tenant-a"}
    expected_access = AccessPolicy.from_settings(config).authorize(
        api_key="correct-key-with-32-characters-0001",
        tenant_id="tenant-a",
    )
    assert adapter.options[-1]["thread_id"] == expected_access.thread_id(
        "conversation-1"
    )


def test_rate_limit_rejects_second_request_in_window() -> None:
    config = Settings(
        app_environment="test",
        rate_limit_enabled=True,
        rate_limit_requests=1,
        rate_limit_window_seconds=60,
        _env_file=None,
    )
    adapter = RecordingAdapter()
    application = create_app(config)
    application.dependency_overrides[get_rag_stream_adapter] = lambda: adapter

    async def make_requests() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            payload = {"conversationId": "c-1", "message": "Question"}
            first = await client.post("/chat/stream", json=payload)
            second = await client.post("/chat/stream", json=payload)
            return first, second

    first, second = asyncio.run(make_requests())

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["retry-after"] == "60"


def test_metrics_endpoint_is_protected_when_key_is_configured() -> None:
    config = Settings(
        app_environment="test",
        metrics_api_key="metrics-secret",
        _env_file=None,
    )
    application = create_app(config)

    async def get_metrics(key: str | None) -> httpx.Response:
        transport = httpx.ASGITransport(app=application)
        headers = {"X-Metrics-Key": key} if key is not None else {}
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get("/metrics", headers=headers)

    assert asyncio.run(get_metrics(None)).status_code == 401
    response = asyncio.run(get_metrics("metrics-secret"))

    assert response.status_code == 200
    assert "agentic_rag_http_requests_total" in response.text
    assert "agentic_rag_streams_total" in response.text


class DisconnectedRequest:
    def __init__(self, metrics: MetricsRegistry):
        self.app = SimpleNamespace(state=SimpleNamespace(metrics=metrics))

    async def is_disconnected(self) -> bool:
        return True


class ClosableEvents:
    closed = False

    def __iter__(self):
        return self

    def __next__(self):
        raise AssertionError("disconnected stream must not request an event")

    def close(self) -> None:
        self.closed = True


def test_disconnected_stream_releases_concurrency_lease_and_closes_events() -> None:
    metrics = MetricsRegistry()
    limiter = RateLimiter(
        enabled=True,
        requests=2,
        window_seconds=60,
        max_concurrent=1,
    )
    lease = limiter.acquire("principal:tenant")
    events = ClosableEvents()

    async def consume_stream() -> None:
        stream = _stream_sse(
            request=DisconnectedRequest(metrics),  # type: ignore[arg-type]
            events=events,  # type: ignore[arg-type]
            cancellation=Event(),
            lease=lease,
        )
        assert [event async for event in stream] == []

    asyncio.run(consume_stream())
    second_lease = limiter.acquire("principal:tenant")
    second_lease.release()

    assert events.closed is True
    assert 'outcome="cancelled"} 1' in metrics.render_prometheus()
