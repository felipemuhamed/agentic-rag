import argparse
import json
from typing import Sequence
from uuid import uuid4

from app.graph.factory import create_rag_graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Answer one question with the graded, retry-limited RAG flow."
    )
    parser.add_argument("question")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = create_rag_graph().invoke(
        {"question": args.question},
        config={"configurable": {"thread_id": f"cli-{uuid4()}"}},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
