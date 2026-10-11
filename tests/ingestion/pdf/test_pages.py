"""Tests for extracting observable data from physical PDF pages."""

from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from src.ingestion.pdf import extract_pdf_page


def write_text_pdf(path: Path) -> None:
    """Create a synthetic two-page PDF with extractable text."""
    writer = PdfWriter()
    _add_text_page(writer, "First page text", width=300, height=400)
    _add_text_page(writer, "Second page text", width=500, height=600)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        writer.write(stream)


def _add_text_page(
    writer: PdfWriter,
    text: str,
    *,
    width: float,
    height: float,
) -> None:
    """Append one minimal text page to a synthetic PDF writer."""
    page = writer.add_blank_page(width=width, height=height)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}
            )
        }
    )

    content = DecodedStreamObject()
    content.set_data(
        f"BT /F1 12 Tf 20 50 Td ({text}) Tj ET".encode("ascii")
    )
    page[NameObject("/Contents")] = writer._add_object(content)


def test_extract_pdf_page_returns_text_dimensions_and_provenance(
    tmp_path: Path,
) -> None:
    """Extraction preserves the physical page facts needed by chunking."""
    library_root = tmp_path / "library"
    book_path = "books/synthetic.pdf"
    write_text_pdf(library_root / book_path)

    page = extract_pdf_page(library_root, book_path, page_number=2)

    assert page.book_path == book_path
    assert page.page_number == 2
    assert page.width_points == 500
    assert page.height_points == 600
    assert page.image_count == 0
    assert page.drawing_operator_count == 0
    assert page.text.strip() == "Second page text"
    assert page.extractor.startswith("pypdf-")


@pytest.mark.parametrize("page_number", [0, -1])
def test_extract_pdf_page_rejects_non_positive_page_number(
    tmp_path: Path,
    page_number: int,
) -> None:
    """The public page coordinate remains one-based."""
    library_root = tmp_path / "library"
    book_path = "books/synthetic.pdf"
    write_text_pdf(library_root / book_path)

    with pytest.raises(ValueError, match="must be at least 1"):
        extract_pdf_page(library_root, book_path, page_number)


def test_extract_pdf_page_rejects_page_after_document_end(
    tmp_path: Path,
) -> None:
    """Extraction reports a page coordinate outside the PDF."""
    library_root = tmp_path / "library"
    book_path = "books/synthetic.pdf"
    write_text_pdf(library_root / book_path)

    with pytest.raises(ValueError, match="exceeds the document length"):
        extract_pdf_page(library_root, book_path, page_number=3)


def test_extract_pdf_page_rejects_book_outside_library(
    tmp_path: Path,
) -> None:
    """A source locator cannot escape the configured private library."""
    library_root = tmp_path / "library"
    outside_book = tmp_path / "outside.pdf"
    library_root.mkdir()
    write_text_pdf(outside_book)

    with pytest.raises(ValueError, match="must be inside the library root"):
        extract_pdf_page(library_root, "../outside.pdf", page_number=1)
