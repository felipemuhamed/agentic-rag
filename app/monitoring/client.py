"""Load and call the monitoring MCP server's tools from synchronous code.

The allowlist is the safety boundary. A full Zabbix MCP server exposes the whole
API, including calls that change hosts, items and templates; a question typed
into a chat box must not be able to reach those. Only the tools named in the
configuration are ever handed to the model, so widening what the agent can do is
an explicit configuration change rather than a consequence of the server it
happens to be pointed at.
"""

import logging
from collections.abc import Sequence
from threading import Lock
from typing import Any

from app.monitoring.bridge import AsyncBridge


logger = logging.getLogger(__name__)


class MonitoringUnavailable(RuntimeError):
    pass


class MonitoringToolset:
    def __init__(
        self,
        *,
        connections: dict[str, dict[str, Any]],
        bridge: AsyncBridge,
        allowed_tools: Sequence[str],
        timeout_seconds: float,
    ):
        if not connections:
            raise ValueError("at least one MCP connection is required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._connections = connections
        self._bridge = bridge
        self._allowed = {name.strip() for name in allowed_tools if name.strip()}
        self._timeout_seconds = timeout_seconds
        self._tools: list[Any] | None = None
        self._lock = Lock()

    def tools(self) -> list[Any]:
        """Return the allowed tools, loading the catalog once."""
        with self._lock:
            if self._tools is not None:
                return self._tools
            self._tools = self._load()
            return self._tools

    def call(self, name: str, arguments: dict[str, Any]) -> str:
        tool = next((item for item in self.tools() if item.name == name), None)
        if tool is None:
            # Reached only if the model invents a name; the allowlist already
            # decided what it was allowed to see.
            raise MonitoringUnavailable(f"tool '{name}' is not available")
        result = self._bridge.run(
            tool.ainvoke(arguments),
            timeout=self._timeout_seconds,
        )
        return as_text(result)

    def close(self) -> None:
        with self._lock:
            self._tools = None

    def _load(self) -> list[Any]:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(self._connections)
        try:
            discovered = self._bridge.run(
                client.get_tools(),
                timeout=self._timeout_seconds,
            )
        except Exception as error:
            raise MonitoringUnavailable(
                f"could not load monitoring tools: {type(error).__name__}"
            ) from error

        if not self._allowed:
            logger.warning(
                "No monitoring tool allowlist configured; exposing none",
            )
            return []
        allowed = [tool for tool in discovered if tool.name in self._allowed]
        missing = self._allowed - {tool.name for tool in discovered}
        if missing:
            logger.warning(
                "Configured monitoring tools are absent from the server: %s",
                ", ".join(sorted(missing)),
            )
        logger.info(
            "Monitoring tools loaded",
            extra={"tool_count": len(allowed)},
        )
        return allowed


def as_text(result: Any) -> str:
    """Flatten an MCP tool result into text.

    Tool results arrive as a list of content blocks, one per returned item, so
    the useful payload has to be gathered rather than read from the first block.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return str(result.get("text", result))
    if isinstance(result, (list, tuple)):
        parts = [as_text(item) for item in result]
        return "\n".join(part for part in parts if part)
    return str(result)
