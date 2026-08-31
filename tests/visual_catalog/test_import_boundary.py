import subprocess
import sys

import pytest


# The knowledge image installs only the `knowledge` extra, so nothing reachable
# from these entry points may import LangChain. A re-export added to any
# `__init__.py` on the path is enough to break them at runtime, which is why
# this runs as a subprocess: it has to prove a cold interpreter can do it.
KNOWLEDGE_TIER_ENTRY_POINTS = (
    "app.visual_catalog.cli",
    "app.visual_catalog.evaluation_cli",
    "app.visual_catalog.resolver",
    "app.knowledge.ingestion.cli",
    "app.knowledge.retrieval.cli",
)

BLOCKED_PACKAGES = ("langchain", "langchain_core", "langgraph")

PROGRAM = """
import sys


class BlockedFinder:
    def find_module(self, name, path=None):
        root = name.split(".")[0]
        return self if root in {blocked!r} else None

    def load_module(self, name):
        raise ImportError(f"{{name}} is not installed in the knowledge image")


sys.meta_path.insert(0, BlockedFinder())
import {module}
"""


@pytest.mark.parametrize("module", KNOWLEDGE_TIER_ENTRY_POINTS)
def test_knowledge_entry_points_import_without_langchain(module: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            PROGRAM.format(module=module, blocked=set(BLOCKED_PACKAGES)),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
