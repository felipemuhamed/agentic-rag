from collections.abc import Callable

import pytest


@pytest.fixture
def build_pdf() -> Callable[[list[str]], bytes]:
    """Build a minimal PDF with one text line per page, without extra tools."""
    return _build_pdf


def _build_pdf(pages: list[str]) -> bytes:
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    catalog_number = 1
    pages_number = 2
    objects.append(b"")  # placeholder catalog
    objects.append(b"")  # placeholder pages
    font_number = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_numbers = []
    for text in pages:
        stream = (
            f"BT /F1 12 Tf 40 700 Td ({text}) Tj ET".encode("latin-1")
        )
        content_number = add(
            b"<< /Length "
            + str(len(stream)).encode()
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
        page_number = add(
            b"<< /Type /Page /Parent "
            + str(pages_number).encode()
            + b" 0 R /MediaBox [0 0 595 842] /Contents "
            + str(content_number).encode()
            + b" 0 R /Resources << /Font << /F1 "
            + str(font_number).encode()
            + b" 0 R >> >> >>"
        )
        page_numbers.append(page_number)

    kids = b" ".join(f"{number} 0 R".encode() for number in page_numbers)
    objects[pages_number - 1] = (
        b"<< /Type /Pages /Kids [" + kids + b"] /Count "
        + str(len(page_numbers)).encode() + b" >>"
    )
    objects[catalog_number - 1] = (
        b"<< /Type /Catalog /Pages " + str(pages_number).encode() + b" 0 R >>"
    )

    output = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(output))
        output += str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n"

    xref_offset = len(output)
    output += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n"
    output += b"0000000000 65535 f \n"
    for offset in offsets:
        output += f"{offset:010d} 00000 n \n".encode()
    output += (
        b"trailer\n<< /Size " + str(len(objects) + 1).encode()
        + b" /Root " + str(catalog_number).encode() + b" 0 R >>\nstartxref\n"
        + str(xref_offset).encode() + b"\n%%EOF\n"
    )
    return bytes(output)
