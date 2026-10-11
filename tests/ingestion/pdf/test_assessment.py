"""Tests for routing PDF pages from observable content signals."""

import pytest

from src.ingestion.pdf import (
    assess_pdf_page,
    PdfPage,
    PdfPageContentKind,
)


def pdf_page(
    *,
    text: str = "",
    image_count: int = 0,
    drawing_operator_count: int = 0,
) -> PdfPage:
    """Create a page observation without reading a real book."""
    return PdfPage(
        book_path="books/synthetic.pdf",
        page_number=1,
        width_points=300,
        height_points=400,
        image_count=image_count,
        drawing_operator_count=drawing_operator_count,
        text=text,
        text_fragments=(),
        extractor="pypdf-test",
    )


@pytest.mark.parametrize(
    ("page", "expected_kind"),
    [
        (pdf_page(), PdfPageContentKind.EMPTY),
        (pdf_page(text="Readable text"), PdfPageContentKind.TEXT_ONLY),
        (pdf_page(image_count=1), PdfPageContentKind.IMAGE_ONLY),
        (
            pdf_page(text="Caption", image_count=1),
            PdfPageContentKind.TEXT_WITH_IMAGES,
        ),
        (
            pdf_page(drawing_operator_count=4),
            PdfPageContentKind.POSSIBLE_VECTOR_GRAPHICS,
        ),
        (
            pdf_page(text="Text", drawing_operator_count=4),
            PdfPageContentKind.TEXT_ONLY,
        ),
    ],
)
def test_assess_pdf_page_routes_observable_content(
    page: PdfPage,
    expected_kind: PdfPageContentKind,
) -> None:
    """Each signal combination receives an explicit processing route."""
    assessment = assess_pdf_page(page)

    assert assessment.content_kind is expected_kind
    assert assessment.text_char_count == len(page.text.strip())
    assert assessment.image_count == page.image_count
    assert assessment.drawing_operator_count == page.drawing_operator_count
