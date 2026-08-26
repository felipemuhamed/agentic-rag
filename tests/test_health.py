import asyncio

import httpx

from app.main import app, create_app
from app.settings import Settings


def test_health_returns_service_status() -> None:
    async def request_health() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get("/health")

    response = asyncio.run(request_health())

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Agentic RAG",
        "version": "0.1.0",
    }


class UnreadyRuntime:
    ready = False

    def close(self) -> None:
        pass


def test_readiness_reports_runtime_initialization_failure() -> None:
    application = create_app(
        Settings(app_environment="test", _env_file=None),
        runtime=UnreadyRuntime(),  # type: ignore[arg-type]
    )

    async def request_readiness() -> httpx.Response:
        transport = httpx.ASGITransport(app=application)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get("/ready")

    response = asyncio.run(request_readiness())

    assert response.status_code == 503
    assert response.json() == {"detail": "service not ready"}
