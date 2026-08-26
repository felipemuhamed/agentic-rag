from app.resilience.web import ResilientWebSearchProvider
from app.settings import Settings
from app.websearch.disabled import DisabledWebSearchProvider
from app.websearch.factory import create_web_search_provider


def test_factory_returns_disabled_provider_when_web_search_is_off() -> None:
    provider = create_web_search_provider(
        Settings(web_search_enabled=False, _env_file=None)
    )

    assert isinstance(provider, DisabledWebSearchProvider)
    assert provider.search("consulta") == []


def test_factory_builds_tavily_provider_only_when_enabled() -> None:
    provider = create_web_search_provider(
        Settings(
            web_search_enabled=True,
            tavily_api_key="test-tavily-key",
            _env_file=None,
        )
    )

    assert isinstance(provider, ResilientWebSearchProvider)
