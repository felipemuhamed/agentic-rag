import argparse
import json
from pathlib import Path
from typing import Sequence

from app.knowledge.embeddings import SentenceTransformerEmbeddingProvider
from app.knowledge.ingestion.chunker import TextChunker
from app.knowledge.ingestion.loaders import TextFileLoader
from app.knowledge.ingestion.metadata import FileMetadataExtractor
from app.knowledge.ingestion.normalizer import TextNormalizer
from app.knowledge.ingestion.service import IngestionService
from app.knowledge.vectorstore.factory import create_chroma_vector_store
from app.settings import settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load TXT or Markdown documents into Chroma."
    )
    parser.add_argument("paths", nargs="+", type=Path)
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
    parser.add_argument("--collection", default=settings.chroma_collection)
    parser.add_argument("--embedding-model", default=settings.embedding_model)
    parser.add_argument("--device", default=settings.embedding_device)
    parser.add_argument("--tenant-id", default=settings.knowledge_tenant_id)
    parser.add_argument("--chunk-size", type=int, default=settings.ingestion_chunk_size)
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.ingestion_chunk_overlap,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    vector_store = create_chroma_vector_store(
        mode=args.chroma_mode,
        collection_name=args.collection,
        path=args.chroma_path,
        host=args.chroma_host,
        port=args.chroma_port,
        ssl=args.chroma_ssl,
        tenant=args.chroma_tenant,
        database=args.chroma_database,
    )

    service = IngestionService(
        loader=TextFileLoader(
            FileMetadataExtractor(tenant_id=args.tenant_id)
        ),
        normalizer=TextNormalizer(),
        chunker=TextChunker(args.chunk_size, args.chunk_overlap),
        embedding_provider=SentenceTransformerEmbeddingProvider(
            args.embedding_model,
            device=args.device,
        ),
        vector_store=vector_store,
    )

    for path in args.paths:
        result = service.ingest_path(path)
        print(
            json.dumps(
                {
                    "source": str(path),
                    "document_id": result.document_id,
                    "chunks_written": result.chunks_written,
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
