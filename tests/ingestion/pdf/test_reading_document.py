"""Integration tests for deterministic PDF reading preparation."""

from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from src.ingestion.pdf import prepare_pdf_reading_document


def _write_book_with_repeated_footers(path: Path) -> None:
    writer = PdfWriter()
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)

    for page_number in range(1, 4):
        page = writer.add_blank_page(width=300, height=400)
        page[NameObject("/Resources")] = DictionaryObject(
            {
                NameObject("/Font"): DictionaryObject(
                    {NameObject("/F1"): font_reference}
                )
            }
        )
        content = DecodedStreamObject()
        content.set_data(
            (
                "BT /F1 12 Tf 20 200 Td "
                f"(Body {page_number}.) Tj ET "
                "BT /F1 9 Tf 20 20 Td "
                f"({page_number} | Guide) Tj ET"
            ).encode("ascii")
        )
        page[NameObject("/Contents")] = writer._add_object(content)

    path.parent.mkdir(parents=True)
    with path.open("wb") as stream:
        writer.write(stream)


def test_prepares_one_auditable_document_for_chunking(
    tmp_path: Path,
) -> None:
    library_root = tmp_path / "library"
    book_path = "books/example.pdf"
    _write_book_with_repeated_footers(library_root / book_path)

    document = prepare_pdf_reading_document(library_root, book_path)

    assert document.book_path == book_path
    assert len(document.physical_pages) == 3
    assert len(document.margin_patterns) == 1
    assert tuple(
        page.reading_text.strip() for page in document.reading_pages
    ) == ("Body 1.", "Body 2.", "Body 3.")
    assert len(document.reading_flows) == 1
    assert document.reading_flows[0].text == (
        "Body 1. Body 2. Body 3."
    )
    assert tuple(
        span.page_number
        for span in document.reading_flows[0].page_spans
    ) == (1, 2, 3)
