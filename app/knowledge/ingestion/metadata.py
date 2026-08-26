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
    def __init__(self, *, tenant_id: str = ""):
        self._tenant_id = tenant_id.strip()

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
        document_identity = (
            f"{self._tenant_id}\0{source}" if self._tenant_id else source
        )
        return SourceDocument(
            document_id=stable_document_id(document_identity),
            title=resolved_path.stem,
            source=source,
            content=content,
            created_at=_as_utc_iso(stat.st_ctime),
            updated_at=_as_utc_iso(stat.st_mtime),
            media_type=media_type,
            tenant_id=self._tenant_id,
        )


def _as_utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()
