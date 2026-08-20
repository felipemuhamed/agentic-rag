from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.settings import settings
from app.streaming.adapter import LangGraphStreamAdapter, StreamableRagGraph


router = APIRouter(prefix="/chat", tags=["chat"])


class ChatStreamRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    conversation_id: str = Field(alias="conversationId", min_length=1)
    message: str = Field(min_length=1)

    @field_validator("conversation_id", "message")
    @classmethod
    def values_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value must not be blank")
        return normalized


@lru_cache(maxsize=1)
def _get_rag_graph() -> StreamableRagGraph:
    from app.graph.factory import create_rag_graph

    return create_rag_graph()


@lru_cache(maxsize=1)
def get_rag_stream_adapter() -> LangGraphStreamAdapter:
    return LangGraphStreamAdapter(
        graph_factory=_get_rag_graph,
        max_rag_attempts=settings.max_rag_attempts,
    )


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
def stream_chat(
    request: ChatStreamRequest,
    adapter: Annotated[
        LangGraphStreamAdapter,
        Depends(get_rag_stream_adapter),
    ],
) -> StreamingResponse:
    events = adapter.stream(
        conversation_id=request.conversation_id,
        message=request.message,
    )
    return StreamingResponse(
        (event.to_sse() for event in events),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
