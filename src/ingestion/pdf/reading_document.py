"""Prepare one private PDF book for later semantic chunking."""

from dataclasses import dataclass
from pathlib import Path

from src.ingestion.pdf.margins import (
    find_repeated_margin_patterns,
    PdfMarginPattern,
)
from src.ingestion.pdf.pages import extract_pdf_pages, PdfPage
from src.ingestion.pdf.reading_flow import (
    build_pdf_reading_flows,
    PdfReadingFlow,
)
from src.ingestion.pdf.reading_text import (
    build_pdf_reading_pages,
    PdfReadingPage,
)


@dataclass(frozen=True, slots=True)
class PdfReadingDocument:
    """Keep every auditable stage of PDF reading-text preparation."""

    book_path: str
    physical_pages: tuple[PdfPage, ...]
    margin_patterns: tuple[PdfMarginPattern, ...]
    reading_pages: tuple[PdfReadingPage, ...]
    reading_flows: tuple[PdfReadingFlow, ...]


def prepare_pdf_reading_document(
    library_root: Path,
    book_path: str,
) -> PdfReadingDocument:
    """Run deterministic PDF text preparation before chunking."""
    physical_pages = extract_pdf_pages(library_root, book_path)
    margin_patterns = find_repeated_margin_patterns(physical_pages)
    reading_pages = build_pdf_reading_pages(
        physical_pages,
        margin_patterns,
    )
    reading_flows = build_pdf_reading_flows(reading_pages)
    return PdfReadingDocument(
        book_path=book_path,
        physical_pages=physical_pages,
        margin_patterns=margin_patterns,
        reading_pages=reading_pages,
        reading_flows=reading_flows,
    )
