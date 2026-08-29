from typing import Any


class PdfExtractionError(ValueError):
    pass


def extract_pdf_text(data: bytes) -> str:
    """Extract the text layer of a PDF as Markdown with one heading per page.

    The chunker derives ``section`` from Markdown headings, so writing the page
    number as a heading is what carries page provenance into chunk metadata and,
    from there, into the citations returned by the API.
    """
    reader = _read_pdf(data)
    pages: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as error:  # pragma: no cover - depends on the file
            raise PdfExtractionError(
                f"could not extract text from page {page_number}"
            ) from error
        if not text.strip():
            continue
        pages.append(f"## Pagina {page_number}\n\n{text.strip()}")

    if not pages:
        raise PdfExtractionError(
            "the PDF has no extractable text layer; scanned files need OCR "
            "before ingestion"
        )
    return "\n\n".join(pages)


def _read_pdf(data: bytes) -> Any:
    from io import BytesIO

    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as error:
        raise RuntimeError(
            "Install the knowledge dependencies with "
            "`python -m pip install -e '.[knowledge]'`"
        ) from error

    try:
        reader = PdfReader(BytesIO(data))
    except PdfReadError as error:
        raise PdfExtractionError(f"invalid PDF file: {error}") from error
    if reader.is_encrypted:
        raise PdfExtractionError("encrypted PDF files are not supported")
    return reader
