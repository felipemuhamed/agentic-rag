from typing import Any

from app.llm.models import create_chat_model
from app.monitoring.agent import MonitoringInvestigator, MonitoringRouter
from app.monitoring.bridge import AsyncBridge
from app.monitoring.client import MonitoringToolset
from app.settings import Settings, settings


def allowed_tools(config: Settings = settings) -> list[str]:
    return [
        name.strip()
        for name in config.monitoring_allowed_tools.split(",")
        if name.strip()
    ]


def create_connections(config: Settings = settings) -> dict[str, dict[str, Any]]:
    url = config.monitoring_mcp_url.strip()
    if not url:
        raise ValueError("MONITORING_MCP_URL must be configured")
    connection: dict[str, Any] = {"transport": "streamable_http", "url": url}
    token = config.monitoring_mcp_token.get_secret_value().strip()
    if token:
        connection["headers"] = {"Authorization": f"Bearer {token}"}
    return {"monitoring": connection}


def create_toolset(
    config: Settings = settings,
    *,
    bridge: AsyncBridge,
) -> MonitoringToolset:
    return MonitoringToolset(
        connections=create_connections(config),
        bridge=bridge,
        allowed_tools=allowed_tools(config),
        timeout_seconds=config.monitoring_timeout_seconds,
    )


def create_monitoring_router(config: Settings = settings) -> MonitoringRouter:
    return MonitoringRouter(
        create_chat_model(
            config.llm_model,
            timeout=config.llm_timeout_seconds,
            max_retries=0,
        ),
        max_attempts=config.llm_max_attempts,
    )


def create_monitoring_investigator(
    config: Settings = settings,
    *,
    toolset: MonitoringToolset,
) -> MonitoringInvestigator:
    return MonitoringInvestigator(
        create_chat_model(
            config.llm_model,
            timeout=config.llm_timeout_seconds,
            max_retries=0,
        ),
        toolset=toolset,
        max_iterations=config.monitoring_max_iterations,
    )
