from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from app.knowledge.retrieval.models import (
    RetrievalDiagnostics,
    RetrievedDocument,
)
from app.websearch.models import WebSearchResult


class RagInput(TypedDict):
    question: str


class RagOutput(TypedDict):
    answer: str
    source: Literal["rag", "web", "none"]


class RagState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    prepared_query: NotRequired[str]
    search_query: NotRequired[str]
    retrieval_attempts: NotRequired[int]
    documents: NotRequired[list[RetrievedDocument]]
    retrieval_diagnostics: NotRequired[RetrievalDiagnostics]
    rag_relevant: NotRequired[bool]
    rag_confidence: NotRequired[float]
    rag_reason: NotRequired[str]
    web_results: NotRequired[list[WebSearchResult]]
    web_relevant: NotRequired[bool]
    web_confidence: NotRequired[float]
    web_reason: NotRequired[str]
    answer: NotRequired[str]
    source: NotRequired[Literal["rag", "web", "none"]]
