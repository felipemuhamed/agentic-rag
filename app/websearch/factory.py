from app.settings import Settings
from app.websearch.base import WebSearchProvider
from app.websearch.tavily import TavilyWebSearchProvider


def create_web_search_provider(config: Settings) -> WebSearchProvider:
    if config.web_search_provider == "tavily":
        return TavilyWebSearchProvider(
            api_key=config.tavily_api_key,
            base_url=config.tavily_base_url,
            max_results=config.web_search_max_results,
            search_depth=config.web_search_depth,
            timeout_seconds=config.web_search_timeout_seconds,
        )
    raise ValueError(
        f"unsupported web search provider: {config.web_search_provider}"
    )
