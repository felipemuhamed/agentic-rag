from collections.abc import Mapping
from typing import Protocol
from urllib.parse import urlparse

import httpx

from app.websearch.models import WebSearchResult


class HttpResponse(Protocol):
    def raise_for_status(self) -> None: ...

    def json(self) -> object: ...


class HttpClient(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: float,
    ) -> HttpResponse: ...


class TavilyWebSearchProvider:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        max_results: int,
        search_depth: str,
        timeout_seconds: float,
        client: HttpClient | None = None,
    ):
        if max_results < 1 or max_results > 20:
            raise ValueError("max_results must be between 1 and 20")
        if search_depth not in {"basic", "advanced", "fast", "ultra-fast"}:
            raise ValueError("unsupported Tavily search depth")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")

        normalized_base_url = base_url.strip().rstrip("/")
        parsed_base_url = urlparse(normalized_base_url)
        if (
            parsed_base_url.scheme not in {"http", "https"}
            or not parsed_base_url.netloc
        ):
            raise ValueError("base_url must be an absolute HTTP(S) URL")

        self._api_key = api_key.strip()
        self._base_url = normalized_base_url
        self._max_results = max_results
        self._search_depth = search_depth
        self._timeout_seconds = timeout_seconds
        self._client = client or httpx.Client()

    def search(self, query: str) -> list[WebSearchResult]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("web search query must not be empty")
        if not self._api_key:
            raise RuntimeError(
                "TAVILY_API_KEY must be configured before web fallback"
            )

        response = self._client.post(
            f"{self._base_url}/search",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "query": normalized_query,
                "search_depth": self._search_depth,
                "max_results": self._max_results,
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False,
            },
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()
        return _parse_results(response.json())


def _parse_results(payload: object) -> list[WebSearchResult]:
    if not isinstance(payload, Mapping):
        raise RuntimeError("Tavily returned a non-object response")
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise RuntimeError("Tavily response does not contain a results list")

    results: list[WebSearchResult] = []
    for raw_result in raw_results:
        parsed = _parse_result(raw_result)
        if parsed is not None:
            results.append(parsed)
    return results


def _parse_result(raw_result: object) -> WebSearchResult | None:
    if not isinstance(raw_result, Mapping):
        return None

    title = raw_result.get("title")
    url = raw_result.get("url")
    content = raw_result.get("content")
    score = raw_result.get("score")
    if (
        not isinstance(title, str)
        or not isinstance(url, str)
        or not isinstance(content, str)
        or isinstance(score, bool)
        or not isinstance(score, (int, float))
    ):
        return None

    try:
        return WebSearchResult(
            title=title,
            url=url,
            content=content,
            score=float(score),
            provider="tavily",
        )
    except ValueError:
        return None
