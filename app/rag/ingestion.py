from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from app.rag.vector_store import get_vector_store
from app.settings import settings


tokenizer = AutoTokenizer.from_pretrained(
    settings.embedding_model,
)


splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=700,
    chunk_overlap=100,
    add_start_index=True,
)


def ingest_documents(
    documents: list[Document],
) -> int:

    chunks = splitter.split_documents(documents)

    ids = []

    for chunk_index, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = chunk_index

        ids.append(str(uuid4()))

    get_vector_store().add_documents(
        documents=chunks,
        ids=ids,
    )

    return len(chunks)