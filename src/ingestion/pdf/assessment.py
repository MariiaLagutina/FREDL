"""Classify observable PDF page content before semantic analysis."""

from dataclasses import dataclass
from enum import Enum

from src.ingestion.pdf.pages import PdfPage


class PdfPageContentKind(str, Enum):
    """Describe observable content without assigning a semantic page role."""

    EMPTY = "empty"
    IMAGE_ONLY = "image_only"
    POSSIBLE_VECTOR_GRAPHICS = "possible_vector_graphics"
    TEXT_ONLY = "text_only"
    TEXT_WITH_IMAGES = "text_with_images"


@dataclass(frozen=True, slots=True)
class PdfPageAssessment:
    """Summarize the signals used to route later page processing."""

    content_kind: PdfPageContentKind
    text_char_count: int
    image_count: int
    drawing_operator_count: int


def assess_pdf_page(page: PdfPage) -> PdfPageAssessment:
    """Classify a page from observable extraction signals only."""
    text_char_count = len(page.text.strip())

    if text_char_count and page.image_count:
        content_kind = PdfPageContentKind.TEXT_WITH_IMAGES
    elif page.image_count:
        content_kind = PdfPageContentKind.IMAGE_ONLY
    elif text_char_count:
        content_kind = PdfPageContentKind.TEXT_ONLY
    elif page.drawing_operator_count:
        content_kind = PdfPageContentKind.POSSIBLE_VECTOR_GRAPHICS
    else:
        content_kind = PdfPageContentKind.EMPTY

    return PdfPageAssessment(
        content_kind=content_kind,
        text_char_count=text_char_count,
        image_count=page.image_count,
        drawing_operator_count=page.drawing_operator_count,
    )
