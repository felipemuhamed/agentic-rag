import re

from app.knowledge.models import Chunk, SourceDocument, stable_chunk_id


class TextChunker:
    _heading = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 100):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be between zero and chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: SourceDocument) -> list[Chunk]:
        if not document.content:
            return []

        headings = list(self._heading.finditer(document.content))
        spans = self._split_spans(document.content)
        chunks: list[Chunk] = []

        for chunk_index, (start, end) in enumerate(spans):
            content = document.content[start:end].strip()
            section = self._section_at(start, headings) or document.section
            metadata = {
                "document_id": document.document_id,
                "title": document.title,
                "source": document.source,
                "section": section,
                "chunk_index": chunk_index,
                "start_index": start,
                "created_at": document.created_at,
                "updated_at": document.updated_at,
                "media_type": document.media_type,
            }
            if document.page is not None:
                metadata["page"] = document.page
            if document.url is not None:
                metadata["url"] = document.url

            chunks.append(
                Chunk(
                    chunk_id=stable_chunk_id(
                        document.document_id,
                        chunk_index,
                        content,
                    ),
                    content=content,
                    metadata=metadata,
                )
            )

        return chunks

    def _split_spans(self, content: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        start = 0
        content_length = len(content)

        while start < content_length:
            maximum_end = min(start + self.chunk_size, content_length)
            end = self._find_boundary(content, start, maximum_end)
            spans.append((start, end))

            if end == content_length:
                break

            next_start = end - self.chunk_overlap
            while next_start < end and content[next_start].isspace():
                next_start += 1
            start = max(start + 1, next_start)

        return spans

    def _find_boundary(self, content: str, start: int, maximum_end: int) -> int:
        if maximum_end == len(content):
            return maximum_end

        minimum_end = start + (self.chunk_size // 2)
        for separator in ("\n\n", "\n", " "):
            boundary = content.rfind(separator, minimum_end, maximum_end)
            if boundary != -1:
                return boundary + len(separator)
        return maximum_end

    @staticmethod
    def _section_at(start: int, headings: list[re.Match[str]]) -> str:
        section = ""
        for heading in headings:
            if heading.start() > start:
                break
            section = heading.group(2).strip()
        return section
