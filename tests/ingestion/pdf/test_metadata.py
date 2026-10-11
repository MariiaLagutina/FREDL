"""Tests for reading identity metadata from PDF books."""

from pathlib import Path

from pypdf import PdfWriter

from src.ingestion.pdf import read_pdf_identity


def write_pdf(
    path: Path,
    *,
    title: str | None = None,
    author: str | None = None,
) -> None:
    """Create a minimal synthetic PDF without copyrighted source content."""
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=400)
    writer.add_blank_page(width=300, height=400)

    metadata = {}
    if title is not None:
        metadata["/Title"] = title
    if author is not None:
        metadata["/Author"] = author
    if metadata:
        writer.add_metadata(metadata)

    with path.open("wb") as stream:
        writer.write(stream)


def test_read_pdf_identity_returns_metadata_without_page_content(
    tmp_path: Path,
) -> None:
    """Identity exposes fields needed by a generated book card."""
    pdf_path = tmp_path / "book.pdf"
    write_pdf(
        pdf_path,
        title="  A Synthetic Book  ",
        author="  Example Author  ",
    )

    identity = read_pdf_identity(pdf_path)

    assert identity.title == "A Synthetic Book"
    assert identity.author == "Example Author"
    assert identity.page_count == 2


def test_read_pdf_identity_preserves_missing_metadata(tmp_path: Path) -> None:
    """Missing optional metadata remains explicit instead of invented."""
    pdf_path = tmp_path / "untitled.pdf"
    write_pdf(pdf_path, title="  ")

    identity = read_pdf_identity(pdf_path)

    assert identity.title is None
    assert identity.author is None
    assert identity.page_count == 2
