from hmac import compare_digest
from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse


router = APIRouter(tags=["operations"])


@router.get("/ready")
async def readiness(request: Request) -> dict[str, str]:
    runtime = request.app.state.rag_runtime
    if not runtime.ready:
        raise HTTPException(status_code=503, detail="service not ready")
    return {"status": "ready"}


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics(
    request: Request,
    metrics_key: Annotated[str | None, Header(alias="X-Metrics-Key")] = None,
) -> PlainTextResponse:
    config = request.app.state.config
    if not config.metrics_enabled:
        raise HTTPException(status_code=404, detail="not found")
    expected = config.metrics_api_key.get_secret_value()
    if expected and (
        metrics_key is None or not compare_digest(metrics_key, expected)
    ):
        raise HTTPException(status_code=401, detail="invalid metrics credential")
    return PlainTextResponse(
        request.app.state.metrics.render_prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
