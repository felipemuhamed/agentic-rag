from app.settings import Settings
from app.resilience.web import ResilientWebSearchProvider
from app.websearch.base import WebSearchProvider
from app.websearch.disabled import DisabledWebSearchProvider
from app.websearch.tavily import TavilyWebSearchProvider


def create_web_search_provider(config: Settings) -> WebSearchProvider:
    if not config.web_search_enabled:
        return DisabledWebSearchProvider()
    if config.web_search_provider == "tavily":
        provider = TavilyWebSearchProvider(
            api_key=config.tavily_api_key.get_secret_value(),
            base_url=config.tavily_base_url,
            max_results=config.web_search_max_results,
            search_depth=config.web_search_depth,
            timeout_seconds=config.web_search_timeout_seconds,
        )
        return ResilientWebSearchProvider(
            provider,
            max_attempts=config.external_max_attempts,
            backoff_seconds=config.external_backoff_seconds,
            circuit_failure_threshold=config.circuit_failure_threshold,
            circuit_reset_seconds=config.circuit_reset_seconds,
        )
    raise ValueError(
        f"unsupported web search provider: {config.web_search_provider}"
    )
