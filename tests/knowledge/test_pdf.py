from collections.abc import Callable
from pathlib import Path

import pytest

from app.knowledge.ingestion.chunker import TextChunker
from app.knowledge.ingestion.loaders import DocumentFileLoader
from app.knowledge.ingestion.metadata import (
    FileMetadataExtractor,
    UnsupportedDocumentTypeError,
)
from app.knowledge.ingestion.normalizer import TextNormalizer
from app.knowledge.ingestion.pdf import PdfExtractionError, extract_pdf_text


def test_every_page_becomes_a_heading_that_carries_its_number(
    build_pdf: Callable[[list[str]], bytes],
) -> None:
    text = extract_pdf_text(build_pdf(["Alarme critico na porta 3", "Procedimento"]))

    assert "## Pagina 1" in text
    assert "## Pagina 2" in text
    assert "Alarme critico na porta 3" in text
    assert text.index("## Pagina 1") < text.index("## Pagina 2")


def test_page_numbers_reach_the_chunk_metadata_as_sections(
    tmp_path: Path,
    build_pdf: Callable[[list[str]], bytes],
) -> None:
    path = tmp_path / "manual.pdf"
    path.write_bytes(build_pdf(["Instalacao do chassi", "Troca de modulo optico"]))
    document = DocumentFileLoader(FileMetadataExtractor(tenant_id="public")).load(path)
    normalized = document.with_content(TextNormalizer().normalize(document.content))

    chunks = TextChunker(chunk_size=40, chunk_overlap=10).split(normalized)
    sections = {chunk.metadata["section"] for chunk in chunks}

    assert document.media_type == "application/pdf"
    assert sections == {"Pagina 1", "Pagina 2"}


def test_a_pdf_without_a_text_layer_is_rejected() -> None:
    with pytest.raises(PdfExtractionError, match="OCR"):
        extract_pdf_text(_build_empty_pdf())


def test_a_file_that_is_not_a_pdf_is_rejected() -> None:
    with pytest.raises(PdfExtractionError, match="invalid PDF"):
        extract_pdf_text(b"nao e um PDF")


def test_other_extensions_remain_unsupported(tmp_path: Path) -> None:
    path = tmp_path / "planilha.xlsx"
    path.write_bytes(b"conteudo")

    with pytest.raises(UnsupportedDocumentTypeError):
        DocumentFileLoader().load(path)


def _build_empty_pdf() -> bytes:
    from pypdf import PdfWriter

    from io import BytesIO

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
