"""Plan how extracted PDF pages enter later ingestion stages."""

from dataclasses import dataclass

from src.ingestion.pdf.assessment import (
    assess_pdf_page,
    PdfPageContentKind,
)
from src.ingestion.pdf.pages import PdfPage


VISUAL_CONTENT_KINDS: frozenset[PdfPageContentKind] = frozenset(
    {
        PdfPageContentKind.IMAGE_ONLY,
        PdfPageContentKind.POSSIBLE_VECTOR_GRAPHICS,
        PdfPageContentKind.TEXT_WITH_IMAGES,
    }
)


@dataclass(frozen=True, slots=True)
class PdfBookPagePlan:
    """Separate page routes without changing physical page coordinates."""

    text_pages: tuple[PdfPage, ...]
    empty_page_numbers: tuple[int, ...]
    visual_page_numbers: tuple[int, ...]


def plan_pdf_pages(pages: tuple[PdfPage, ...]) -> PdfBookPagePlan:
    """Route pages to text, visual, or empty handling in physical order."""
    text_pages: list[PdfPage] = []
    empty_page_numbers: list[int] = []
    visual_page_numbers: list[int] = []

    for page in pages:
        assessment = assess_pdf_page(page)

        if assessment.content_kind is PdfPageContentKind.EMPTY:
            empty_page_numbers.append(page.page_number)
            continue

        if assessment.content_kind in VISUAL_CONTENT_KINDS:
            visual_page_numbers.append(page.page_number)

        if assessment.text_char_count:
            text_pages.append(page)

    return PdfBookPagePlan(
        text_pages=tuple(text_pages),
        empty_page_numbers=tuple(empty_page_numbers),
        visual_page_numbers=tuple(visual_page_numbers),
    )
