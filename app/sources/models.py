from dataclasses import dataclass
from typing import Literal, TypedDict
from urllib.parse import urlparse


SourceType = Literal["rag", "web"]


class PublicSource(TypedDict):
    id: int
    citation: str
    type: SourceType
    title: str
    source: str
    chunkId: str | None
    url: str | None
    page: int | None
    section: str | None


@dataclass(frozen=True, slots=True)
class Source:
    citation_id: int
    source_type: SourceType
    title: str
    source: str
    chunk_id: str | None = None
    url: str | None = None
    page: int | None = None
    section: str | None = None

    def __post_init__(self) -> None:
        title = self.title.strip()
        source = self.source.strip()
        chunk_id = _optional_text(self.chunk_id)
        url = _optional_text(self.url)
        section = _optional_text(self.section)

        if (
            isinstance(self.citation_id, bool)
            or not isinstance(self.citation_id, int)
            or self.citation_id < 1
        ):
            raise ValueError("citation_id must be at least one")
        if self.source_type not in {"rag", "web"}:
            raise ValueError("source_type must be rag or web")
        if not title:
            raise ValueError("source title must not be empty")
        if not source:
            raise ValueError("source locator must not be empty")
        if url is not None:
            parsed_url = urlparse(url)
            if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
                raise ValueError("source URL must be absolute HTTP(S)")
        if self.page is not None and (
            isinstance(self.page, bool) or not isinstance(self.page, int)
        ):
            raise ValueError("source page must be an integer")
        if self.source_type == "rag" and chunk_id is None:
            raise ValueError("RAG source must map to a chunk_id")
        if self.source_type == "web" and url is None:
            raise ValueError("web source must map to a URL")

        object.__setattr__(self, "title", title)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "chunk_id", chunk_id)
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "section", section)

    @property
    def citation(self) -> str:
        return f"[{self.citation_id}]"

    def to_public(self) -> PublicSource:
        return {
            "id": self.citation_id,
            "citation": self.citation,
            "type": self.source_type,
            "title": self.title,
            "source": self.source,
            "chunkId": self.chunk_id,
            "url": self.url,
            "page": self.page,
            "section": self.section,
        }

    @classmethod
    def from_public(cls, value: object) -> "Source":
        if not isinstance(value, dict):
            raise ValueError("public source must be an object")
        expected_fields = {
            "id",
            "citation",
            "type",
            "title",
            "source",
            "chunkId",
            "url",
            "page",
            "section",
        }
        if set(value) != expected_fields:
            raise ValueError("public source fields are invalid")

        citation_id = value["id"]
        source_type = value["type"]
        title = value["title"]
        source = value["source"]
        if isinstance(citation_id, bool) or not isinstance(citation_id, int):
            raise ValueError("public source id must be an integer")
        if source_type not in {"rag", "web"}:
            raise ValueError("public source type is invalid")
        if not isinstance(title, str) or not isinstance(source, str):
            raise ValueError("public source text fields are invalid")

        parsed = cls(
            citation_id=citation_id,
            source_type=source_type,
            title=title,
            source=source,
            chunk_id=_public_optional_text(value["chunkId"], "chunkId"),
            url=_public_optional_text(value["url"], "url"),
            page=_public_page(value["page"]),
            section=_public_optional_text(value["section"], "section"),
        )
        if value["citation"] != parsed.citation:
            raise ValueError("public source citation does not match its id")
        return parsed


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _public_optional_text(value: object, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"public source {field} must be text or null")
    return value


def _public_page(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("public source page must be an integer or null")
    return value
