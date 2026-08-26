import logging
from math import ceil
from threading import Event
from time import monotonic
from typing import Annotated, Iterator

import anyio
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.multimodal.models import (
    ABSOLUTE_MAX_IMAGES,
    ChatImage,
    ImageTooLargeError,
    InvalidImageError,
    validate_image_inputs,
)
from app.security.access import AuthenticationError, TenantAccessError
from app.security.rate_limit import RateLimitExceeded, RateLimitLease
from app.streaming.adapter import LangGraphStreamAdapter
from app.streaming.events import StreamEvent


router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)
_END_OF_STREAM = object()


class ChatStreamRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    conversation_id: str = Field(
        alias="conversationId",
        min_length=1,
        max_length=128,
    )
    message: str = Field(min_length=1, max_length=8_000)
    tenant_id: str | None = Field(
        default=None,
        alias="tenantId",
        min_length=1,
        max_length=128,
    )
    images: list[ChatImage] = Field(
        default_factory=list,
        max_length=ABSOLUTE_MAX_IMAGES,
    )

    @field_validator("conversation_id", "message", "tenant_id")
    @classmethod
    def values_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized


def get_rag_stream_adapter(request: Request) -> LangGraphStreamAdapter:
    return request.app.state.rag_runtime.adapter()


@router.post(
    "/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-Sent Events stream",
            "content": {"text/event-stream": {}},
        }
    },
)
async def stream_chat(
    payload: ChatStreamRequest,
    request: Request,
    adapter: Annotated[
        LangGraphStreamAdapter,
        Depends(get_rag_stream_adapter),
    ],
    api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> StreamingResponse:
    metrics = request.app.state.metrics
    try:
        access = request.app.state.access_policy.authorize(
            api_key=api_key,
            tenant_id=payload.tenant_id,
        )
    except AuthenticationError as error:
        metrics.auth_failed()
        raise HTTPException(
            status_code=401,
            detail="invalid API credential",
            headers={"WWW-Authenticate": "ApiKey"},
        ) from error
    except TenantAccessError as error:
        metrics.auth_failed()
        raise HTTPException(status_code=403, detail="tenant access denied") from error

    if payload.images and not request.app.state.config.multimodal_enabled:
        raise HTTPException(status_code=422, detail="multimodal input is disabled")
    try:
        images = validate_image_inputs(
            payload.images,
            max_images=request.app.state.config.multimodal_max_images,
            max_image_bytes=(
                request.app.state.config.multimodal_max_image_bytes
            ),
        )
    except ImageTooLargeError as error:
        raise HTTPException(status_code=413, detail=str(error)) from error
    except InvalidImageError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    try:
        lease = request.app.state.rate_limiter.acquire(access.quota_key)
    except RateLimitExceeded as error:
        metrics.rate_limit_rejected()
        raise HTTPException(
            status_code=429,
            detail="rate limit exceeded",
            headers={"Retry-After": str(max(ceil(error.retry_after), 1))},
        ) from error

    cancellation = Event()
    events = adapter.stream(
        conversation_id=payload.conversation_id,
        message=payload.message,
        thread_id=access.thread_id(payload.conversation_id),
        retrieval_filters=access.retrieval_filters,
        images=images,
        is_cancelled=cancellation.is_set,
    )
    return StreamingResponse(
        _stream_sse(
            request=request,
            events=events,
            cancellation=cancellation,
            lease=lease,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-RateLimit-Limit": str(lease.snapshot.limit),
            "X-RateLimit-Remaining": str(lease.snapshot.remaining),
        },
    )


async def _stream_sse(
    *,
    request: Request,
    events: Iterator[StreamEvent],
    cancellation: Event,
    lease: RateLimitLease,
):
    metrics = request.app.state.metrics
    started_at = monotonic()
    outcome = "cancelled"
    metrics.stream_started()
    try:
        while True:
            if await request.is_disconnected():
                cancellation.set()
                break
            event = await anyio.to_thread.run_sync(_next_event, events)
            if event is _END_OF_STREAM:
                break
            if event.event == "done":
                outcome = "completed"
            elif event.event == "error":
                outcome = (
                    "timeout"
                    if event.data.get("code") == "stream_timeout"
                    else "error"
                )
            yield event.to_sse()
    finally:
        cancellation.set()
        lease.release()
        duration = monotonic() - started_at
        metrics.stream_finished(outcome=outcome, duration_seconds=duration)
        logger.info(
            "Chat stream completed",
            extra={
                "stream_outcome": outcome,
                "duration_ms": round(duration * 1000, 3),
            },
        )
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(_close_iterator, events)


def _next_event(events: Iterator[StreamEvent]) -> StreamEvent | object:
    try:
        return next(events)
    except StopIteration:
        return _END_OF_STREAM


def _close_iterator(events: Iterator[StreamEvent]) -> None:
    close = getattr(events, "close", None)
    if callable(close):
        close()
