from typing import Protocol

from app.websearch.models import WebSearchResult


class WebSearchProvider(Protocol):
    def search(self, query: str) -> list[WebSearchResult]: ...
