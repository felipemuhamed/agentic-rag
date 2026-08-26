import argparse
import json
from pathlib import Path
from typing import Sequence

from app.settings import settings
from app.visual_catalog.embeddings import (
    SentenceTransformerVisualEmbeddingProvider,
)
from app.visual_catalog.factory import create_visual_vector_store
from app.visual_catalog.ingestion import (
    VisualReference,
    VisualReferenceIngestionService,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Index proprietary equipment reference images in Chroma."
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--equipment-id", required=True)
    parser.add_argument("--equipment-name", required=True)
    parser.add_argument("--manufacturer", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--description", default="")
    parser.add_argument("--tenant-id", default=settings.knowledge_tenant_id)
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
        }
    )
    service = VisualReferenceIngestionService(
        embedding_provider=SentenceTransformerVisualEmbeddingProvider(
            args.embedding_model,
            device=args.device,
        ),
        vector_store=create_visual_vector_store(config),
    )
    reference = VisualReference(
        equipment_id=args.equipment_id,
        equipment_name=args.equipment_name,
        manufacturer=args.manufacturer,
        model=args.model,
        description=args.description,
        tenant_id=args.tenant_id,
    )
    for path in args.paths:
        result = service.ingest_path(path, reference)
        print(
            json.dumps(
                {
                    "source": str(path),
                    "document_id": result.document_id,
                    "reference_id": result.reference_id,
                    "equipment_id": result.equipment_id,
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
