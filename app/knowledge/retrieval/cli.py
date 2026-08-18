import argparse
import json
from dataclasses import asdict
from typing import Sequence

from app.knowledge.retrieval.factory import create_knowledge_retriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrieve and rerank knowledge without LangGraph."
    )
    parser.add_argument("query")
    parser.add_argument(
        "--where",
        help='Chroma metadata filter as JSON, e.g. {"title":"Guide"}',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    filters = json.loads(args.where) if args.where else None
    if filters is not None and not isinstance(filters, dict):
        raise ValueError("--where must decode to a JSON object")

    result = create_knowledge_retriever().retrieve(args.query, filters=filters)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
