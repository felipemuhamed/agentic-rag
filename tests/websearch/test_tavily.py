from dataclasses import dataclass

import pytest

from app.websearch.tavily import TavilyWebSearchProvider


@dataclass
class FakeResponse:
    payload: object
    raised: bool = False

    def raise_for_status(self) -> None:
        self.raised = True

    def json(self) -> object:
        return self.payload


class RecordingHttpClient:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.calls: list[dict[str, object]] = []

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
        timeout: float,
    ) -> FakeResponse:
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return self.response


def make_provider(client: RecordingHttpClient, api_key: str = "secret-key"):
    return TavilyWebSearchProvider(
        api_key=api_key,
        base_url="https://api.tavily.com/",
        max_results=3,
        search_depth="basic",
        timeout_seconds=4.5,
        client=client,
    )


def test_tavily_adapter_maps_results_and_source_metadata() -> None:
    response = FakeResponse(
        {
            "results": [
                {
                    "title": "OAuth in Keycloak",
                    "url": "https://www.keycloak.org/docs/oauth",
                    "content": "Keycloak supports OAuth 2.0.",
                    "score": 0.91,
                },
                {
                    "title": "Invalid result",
                    "url": "javascript:alert(1)",
                    "content": "This URL must be discarded.",
                    "score": 0.8,
                },
            ]
        }
    )
    client = RecordingHttpClient(response)

    results = make_provider(client).search("  OAuth Keycloak  ")

    assert response.raised is True
    assert len(results) == 1
    assert results[0].title == "OAuth in Keycloak"
    assert results[0].metadata == {
        "source": "https://www.keycloak.org/docs/oauth",
        "url": "https://www.keycloak.org/docs/oauth",
        "title": "OAuth in Keycloak",
        "provider": "tavily",
    }
    assert client.calls == [
        {
            "url": "https://api.tavily.com/search",
            "headers": {
                "Authorization": "Bearer secret-key",
                "Content-Type": "application/json",
            },
            "json": {
                "query": "OAuth Keycloak",
                "search_depth": "basic",
                "max_results": 3,
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False,
            },
            "timeout": 4.5,
        }
    ]


def test_missing_api_key_fails_only_when_fallback_is_called() -> None:
    client = RecordingHttpClient(FakeResponse({"results": []}))
    provider = make_provider(client, api_key="   ")

    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        provider.search("consulta")

    assert client.calls == []


def test_malformed_provider_response_is_not_treated_as_evidence() -> None:
    provider = make_provider(RecordingHttpClient(FakeResponse({"answer": "x"})))

    with pytest.raises(RuntimeError, match="results list"):
        provider.search("consulta")


def test_provider_rejects_invalid_base_url() -> None:
    client = RecordingHttpClient(FakeResponse({"results": []}))

    with pytest.raises(ValueError, match="absolute HTTP"):
        TavilyWebSearchProvider(
            api_key="secret-key",
            base_url="not-a-url",
            max_results=3,
            search_depth="basic",
            timeout_seconds=4.5,
            client=client,
        )
