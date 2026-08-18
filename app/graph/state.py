from typing import Literal

from langchain_core.documents import Document
from langgraph.graph import MessagesState


class RagState(MessagesState):

    question: str

    search_query: str

    documents: list[Document]

    retrieval_attempts: int

    rag_relevant: bool
    rag_confidence: float

    source: Literal[
        "rag",
        "web",
        "none",
    ]