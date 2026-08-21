import json
from collections.abc import Mapping, Sequence
from typing import Protocol

from app.knowledge.retrieval.models import RetrievedDocument


class GroundingDocument(Protocol):
    content: str

    @property
    def metadata(self) -> Mapping[str, object]: ...


def serialize_grounding_documents(
    documents: Sequence[GroundingDocument],
) -> str:
    serialized = []
    for document in documents:
        item: dict[str, object] = {
            "content": document.content,
            "metadata": dict(document.metadata),
        }
        for field_name in ("chunk_id", "title", "url", "score"):
            field_value = getattr(document, field_name, None)
            if field_value is not None:
                item[field_name] = field_value
        serialized.append(item)
    return json.dumps(serialized, ensure_ascii=False, indent=2)


def serialize_retrieved_documents(
    documents: Sequence[RetrievedDocument],
) -> str:
    return serialize_grounding_documents(documents)
