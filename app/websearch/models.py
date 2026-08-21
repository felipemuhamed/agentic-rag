from dataclasses import dataclass
from urllib.parse import urlparse

from app.knowledge.models import MetadataValue


@dataclass(frozen=True, slots=True)
class WebSearchResult:
    title: str
    url: str
    content: str
    score: float
    provider: str

    def __post_init__(self) -> None:
        title = self.title.strip()
        url = self.url.strip()
        content = self.content.strip()
        provider = self.provider.strip()
        parsed_url = urlparse(url)

        if not title:
            raise ValueError("web result title must not be empty")
        if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
            raise ValueError("web result URL must be absolute HTTP(S)")
        if not content:
            raise ValueError("web result content must not be empty")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("web result score must be between zero and one")
        if not provider:
            raise ValueError("web result provider must not be empty")

        object.__setattr__(self, "title", title)
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "content", content)
        object.__setattr__(self, "provider", provider)

    @property
    def metadata(self) -> dict[str, MetadataValue]:
        return {
            "source": self.url,
            "url": self.url,
            "title": self.title,
            "provider": self.provider,
        }
