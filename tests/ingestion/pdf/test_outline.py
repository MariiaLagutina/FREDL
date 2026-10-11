"""Check embedded navigation using synthetic PDFs only."""

from pathlib import Path

import pytest
from pypdf import PdfWriter

from src.ingestion.pdf.outline import read_pdf_outline


def test_preserves_hierarchy_order_and_shared_page(tmp_path: Path) -> None:
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=300, height=400)
    chapter = writer.add_outline_item("Chapter", 1)
    writer.add_outline_item("First topic", 1, parent=chapter)
    writer.add_outline_item("Second topic", 2, parent=chapter)
    writer.add_outline_item("Appendix", 2)
    writer.write(tmp_path / "book.pdf")

    entries = read_pdf_outline(tmp_path, "book.pdf")

    assert tuple(entry.page_number for entry in entries) == (2, 2, 3, 3)
    assert tuple(entry.section_path for entry in entries) == (
        ("Chapter",),
        ("Chapter", "First topic"),
        ("Chapter", "Second topic"),
        ("Appendix",),
    )
    assert entries[1].title == "First topic"


def test_missing_outline_is_empty(tmp_path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=400)
    writer.write(tmp_path / "book.pdf")

    assert read_pdf_outline(tmp_path, "book.pdf") == ()


def test_unresolved_destination_remains_explicit(tmp_path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=400)
    writer.add_outline_item("Unresolved", None)
    writer.write(tmp_path / "book.pdf")

    entries = read_pdf_outline(tmp_path, "book.pdf")

    assert len(entries) == 1
    assert entries[0].page_number is None


def test_rejects_path_outside_library(tmp_path: Path) -> None:
    library = tmp_path / "library"
    library.mkdir()
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=400)
    writer.write(tmp_path / "outside.pdf")

    with pytest.raises(ValueError, match="inside the library"):
        read_pdf_outline(library, "../outside.pdf")
