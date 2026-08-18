from datetime import UTC, datetime
from pathlib import Path

from app.knowledge.models import SourceDocument, stable_document_id


SUPPORTED_MEDIA_TYPES = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
}


class UnsupportedDocumentTypeError(ValueError):
    pass


class FileMetadataExtractor:
    def extract(self, path: Path, content: str) -> SourceDocument:
        resolved_path = path.expanduser().resolve()
        suffix = resolved_path.suffix.lower()

        try:
            media_type = SUPPORTED_MEDIA_TYPES[suffix]
        except KeyError as exc:
            supported = ", ".join(sorted(SUPPORTED_MEDIA_TYPES))
            raise UnsupportedDocumentTypeError(
                f"Unsupported document type '{suffix}'. Supported: {supported}"
            ) from exc

        stat = resolved_path.stat()
        source = str(resolved_path)
        return SourceDocument(
            document_id=stable_document_id(source),
            title=resolved_path.stem,
            source=source,
            content=content,
            created_at=_as_utc_iso(stat.st_ctime),
            updated_at=_as_utc_iso(stat.st_mtime),
            media_type=media_type,
        )


def _as_utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()
