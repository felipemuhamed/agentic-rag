import logging
import re
from contextlib import asynccontextmanager
from threading import Thread
from time import monotonic
from uuid import uuid4

from fastapi import FastAPI, Request

from app.api.chat import router as chat_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.operations import router as operations_router
from app.observability.logging import (
    bind_request_id,
    configure_logging,
    reset_request_id,
)
from app.observability.metrics import MetricsRegistry
from app.runtime import RagRuntime
from app.security.access import AccessPolicy
from app.security.rate_limit import RateLimiter
from app.settings import Settings, settings
from app.uploads.jobs import UploadJobManager


logger = logging.getLogger(__name__)
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def create_app(
    config: Settings = settings,
    *,
    runtime: RagRuntime | None = None,
    metrics: MetricsRegistry | None = None,
) -> FastAPI:
    configure_logging(level=config.log_level, json_logs=config.json_logs)
    rag_runtime = runtime or RagRuntime(config)
    upload_jobs = UploadJobManager(
        workers=config.upload_workers,
        max_pending=config.upload_max_pending_jobs,
        max_jobs=config.upload_max_jobs,
        retention_seconds=config.upload_job_retention_seconds,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if config.warmup_on_startup:
            # Loading happens off the event loop so the server can answer
            # /ready with 503 while it runs, instead of accepting no
            # connections at all and looking like a crashed replica.
            Thread(
                target=_warmup,
                args=(rag_runtime,),
                name="rag-warmup",
                daemon=True,
            ).start()
        yield
        # Ingestion runs outside the request, so it has to drain before the
        # runtime releases the models and the vector store it is writing to.
        upload_jobs.shutdown()
        rag_runtime.close()

    application = FastAPI(
        title=config.app_name,
        version=config.app_version,
        lifespan=lifespan,
    )
    application.state.config = config
    application.state.rag_runtime = rag_runtime
    application.state.metrics = metrics or MetricsRegistry()
    application.state.upload_jobs = upload_jobs
    application.state.access_policy = AccessPolicy.from_settings(config)
    application.state.rate_limiter = RateLimiter(
        enabled=config.rate_limit_enabled,
        requests=config.rate_limit_requests,
        window_seconds=config.rate_limit_window_seconds,
        max_concurrent=config.max_concurrent_streams,
    )

    @application.middleware("http")
    async def observe_request(request: Request, call_next):
        supplied_request_id = request.headers.get("X-Request-ID", "")
        request_id = (
            supplied_request_id
            if REQUEST_ID_PATTERN.fullmatch(supplied_request_id)
            else str(uuid4())
        )
        token = bind_request_id(request_id)
        started_at = monotonic()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            duration = monotonic() - started_at
            application.state.metrics.observe_http(
                status_code=status_code,
                duration_seconds=duration,
            )
            logger.info(
                "HTTP request completed",
                extra={
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "status_code": status_code,
                    "duration_ms": round(duration * 1000, 3),
                },
            )
            reset_request_id(token)

    application.include_router(health_router)
    application.include_router(operations_router)
    application.include_router(chat_router)
    application.include_router(documents_router)
    return application


def _warmup(runtime: RagRuntime) -> None:
    started_at = monotonic()
    try:
        runtime.warmup()
    except Exception as error:
        # Readiness stays closed, which is the signal the orchestrator needs.
        logger.error("Model warmup failed: %s", type(error).__name__)
        return
    logger.info(
        "Model warmup completed",
        extra={"duration_ms": round((monotonic() - started_at) * 1000, 3)},
    )


app = create_app()
