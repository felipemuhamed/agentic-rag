from app.websearch.models import WebSearchResult


class DisabledWebSearchProvider:
    """No-op provider used when external search is explicitly disabled."""

    def search(self, query: str) -> list[WebSearchResult]:
        return []
