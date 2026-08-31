import argparse
import json
from pathlib import Path
from typing import Sequence

from app.settings import settings
from app.visual_catalog.evaluation import (
    DEFAULT_THRESHOLDS,
    VisualEvaluationRunner,
    load_visual_dataset,
)
from app.visual_catalog.factory import (
    create_equipment_resolver,
    create_visual_catalog_retriever,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure equipment identification from photos and recommend a "
            "visual distance threshold."
        )
    )
    parser.add_argument("dataset", help="Path to the visual evaluation dataset")
    parser.add_argument(
        "--images-root",
        default=".",
        help="Directory the dataset image paths are relative to",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
        help="Candidate values for VISUAL_MAX_DISTANCE",
    )
    parser.add_argument(
        "--output",
        help="Write the complete JSON report to this path",
    )
    parser.add_argument(
        "--identifier-weight",
        type=float,
        default=settings.equipment_identifier_weight,
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=settings.equipment_min_confidence,
    )
    parser.add_argument(
        "--ambiguity-margin",
        type=float,
        default=settings.equipment_ambiguity_margin,
    )
    parser.add_argument("--collection", default=settings.visual_collection)
    parser.add_argument(
        "--embedding-model",
        default=settings.visual_embedding_model,
    )
    parser.add_argument("--device", default=settings.visual_embedding_device)
    parser.add_argument(
        "--chroma-mode",
        choices=("http", "persistent"),
        default=settings.chroma_mode,
    )
    parser.add_argument("--chroma-path", default=settings.chroma_path)
    parser.add_argument("--chroma-host", default=settings.chroma_host)
    parser.add_argument("--chroma-port", type=int, default=settings.chroma_port)
    parser.add_argument(
        "--chroma-ssl",
        action=argparse.BooleanOptionalAction,
        default=settings.chroma_ssl,
    )
    parser.add_argument("--chroma-tenant", default=settings.chroma_tenant)
    parser.add_argument("--chroma-database", default=settings.chroma_database)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.thresholds:
        raise ValueError("at least one threshold is required")

    config = settings.model_copy(
        update={
            "chroma_mode": args.chroma_mode,
            "chroma_path": args.chroma_path,
            "chroma_host": args.chroma_host,
            "chroma_port": args.chroma_port,
            "chroma_ssl": args.chroma_ssl,
            "chroma_tenant": args.chroma_tenant,
            "chroma_database": args.chroma_database,
            "visual_collection": args.collection,
            "visual_embedding_model": args.embedding_model,
            "visual_embedding_device": args.device,
            # Query once with the gate wide open; the sweep narrows it in
            # memory, so every threshold is measured on the same distances.
            "visual_max_distance": max(args.thresholds),
            "visual_max_results": 20,
            "equipment_identifier_weight": args.identifier_weight,
            "equipment_min_confidence": args.min_confidence,
            "equipment_ambiguity_margin": args.ambiguity_margin,
        }
    )

    runner = VisualEvaluationRunner(
        retriever=create_visual_catalog_retriever(config),
        resolver_factory=lambda: create_equipment_resolver(config),
        identifier_weight=config.equipment_identifier_weight,
        min_confidence=config.equipment_min_confidence,
        ambiguity_margin=config.equipment_ambiguity_margin,
    )
    report = runner.run(
        load_visual_dataset(args.dataset),
        images_root=args.images_root,
        thresholds=args.thresholds,
    )

    payload = report.model_dump()
    if args.output:
        Path(args.output).expanduser().write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
