"""List what the configured monitoring MCP server actually exposes.

The allowlist has to name tools exactly as the server does, and a full Zabbix
MCP server publishes hundreds of them. Guessing the names produces a silent
empty allowlist, so this prints the catalog to build it from.
"""

import argparse
import json
from typing import Sequence

from app.monitoring.bridge import AsyncBridge
from app.monitoring.factory import create_connections
from app.settings import settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="List the tools published by the monitoring MCP server."
    )
    parser.add_argument(
        "--url",
        default=settings.monitoring_mcp_url,
        help="Streamable HTTP endpoint of the MCP server",
    )
    parser.add_argument(
        "--filter",
        default="",
        help="Only show tools whose name contains this text",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the catalog as JSON instead of a table",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=settings.monitoring_timeout_seconds,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = settings.model_copy(update={"monitoring_mcp_url": args.url})

    from langchain_mcp_adapters.client import MultiServerMCPClient

    bridge = AsyncBridge()
    try:
        client = MultiServerMCPClient(create_connections(config))
        tools = bridge.run(client.get_tools(), timeout=args.timeout)
    finally:
        bridge.close()

    needle = args.filter.strip().lower()
    selected = [
        tool
        for tool in tools
        if not needle or needle in tool.name.lower()
    ]

    if args.json:
        print(
            json.dumps(
                [
                    {"name": tool.name, "description": tool.description}
                    for tool in selected
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        for tool in selected:
            summary = (tool.description or "").strip().splitlines()
            print(f"{tool.name}\t{summary[0] if summary else ''}")
        print(f"\n{len(selected)} de {len(tools)} ferramentas")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
