from pathlib import Path

from app.knowledge.ingestion.metadata import FileMetadataExtractor
from app.knowledge.models import SourceDocument


class TextFileLoader:
    def __init__(self, metadata_extractor: FileMetadataExtractor | None = None):
        self._metadata_extractor = metadata_extractor or FileMetadataExtractor()

    def load(self, path: str | Path) -> SourceDocument:
        document_path = Path(path)
        content = document_path.read_text(encoding="utf-8")
        return self._metadata_extractor.extract(document_path, content)
