"""Build auditable page text without repeated PDF margins."""

from dataclasses import dataclass

from src.ingestion.pdf.margins import (
    find_margin_fragment_indexes,
    find_repeated_margin_patterns,
    PdfMarginPattern,
)
from src.ingestion.pdf.pages import PdfPage


@dataclass(frozen=True, slots=True)
class PdfReadingPage:
    """Keep source and reading text for one physical PDF page."""

    book_path: str
    page_number: int
    raw_text: str
    reading_text: str
    removed_margin_signatures: tuple[str, ...]


def build_pdf_reading_pages(
    pages: tuple[PdfPage, ...],
    patterns: tuple[PdfMarginPattern, ...] | None = None,
) -> tuple[PdfReadingPage, ...]:
    """Remove confirmed repeated margins while preserving raw text."""
    confirmed_patterns = (
        find_repeated_margin_patterns(pages)
        if patterns is None
        else patterns
    )
    return tuple(
        _build_reading_page(page, confirmed_patterns) for page in pages
    )


def _build_reading_page(
    page: PdfPage,
    patterns: tuple[PdfMarginPattern, ...],
) -> PdfReadingPage:
    removed_indexes = set(find_margin_fragment_indexes(page, patterns))
    if removed_indexes:
        reading_text = "".join(
            fragment.text
            for index, fragment in enumerate(page.text_fragments)
            if index not in removed_indexes
        )
    else:
        reading_text = page.text

    removed_signatures = tuple(
        sorted(
            pattern.signature
            for pattern in patterns
            if page.page_number in pattern.page_numbers
        )
    )
    return PdfReadingPage(
        book_path=page.book_path,
        page_number=page.page_number,
        raw_text=page.text,
        reading_text=reading_text,
        removed_margin_signatures=removed_signatures,
    )
