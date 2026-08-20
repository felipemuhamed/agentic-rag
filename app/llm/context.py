import json
from typing import Sequence

from app.knowledge.retrieval.models import RetrievedDocument


def serialize_retrieved_documents(
    documents: Sequence[RetrievedDocument],
) -> str:
    return json.dumps(
        [
            {
                "chunk_id": document.chunk_id,
                "content": document.content,
                "metadata": document.metadata,
            }
            for document in documents
        ],
        ensure_ascii=False,
        indent=2,
    )
