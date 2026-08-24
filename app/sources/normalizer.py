from pathlib import Path

from app.knowledge.models import MetadataValue
from app.knowledge.retrieval.models import RetrievedDocument
from app.sources.models import Source
from app.websearch.models import WebSearchResult


def normalize_rag_sources(
    documents: list[RetrievedDocument],
) -> list[Source]:
    return [
        _rag_source(citation_id, document)
        for citation_id, document in enumerate(documents, start=1)
    ]


def normalize_web_sources(
    results: list[WebSearchResult],
) -> list[Source]:
    return [
        Source(
            citation_id=citation_id,
            source_type="web",
            title=result.title,
            source=result.url,
            url=result.url,
        )
        for citation_id, result in enumerate(results, start=1)
    ]


def _rag_source(citation_id: int, document: RetrievedDocument) -> Source:
    metadata = document.metadata
    raw_source = _metadata_text(metadata, "source")
    url = _metadata_text(metadata, "url")
    source = raw_source or url or f"chunk:{document.chunk_id}"
    title = (
        _metadata_text(metadata, "title")
        or _title_from_locator(source)
        or document.chunk_id
    )
    return Source(
        citation_id=citation_id,
        source_type="rag",
        title=title,
        source=source,
        chunk_id=document.chunk_id,
        url=url,
        page=_metadata_page(metadata),
        section=_metadata_text(metadata, "section"),
    )


def _metadata_text(
    metadata: dict[str, MetadataValue],
    field: str,
) -> str | None:
    value = metadata.get(field)
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _metadata_page(metadata: dict[str, MetadataValue]) -> int | None:
    value = metadata.get("page")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _title_from_locator(locator: str) -> str:
    name = Path(locator).name
    return Path(name).stem if name else ""
