"""Tests for auditable PDF reading-text preparation."""

from src.ingestion.pdf.layout import PdfTextFragment
from src.ingestion.pdf.margins import find_repeated_margin_patterns
from src.ingestion.pdf.pages import PdfPage
from src.ingestion.pdf.reading_text import build_pdf_reading_pages


def _fragment(text: str, page_y: float) -> PdfTextFragment:
    return PdfTextFragment(
        text=text,
        container_x=0.0,
        container_y=0.0,
        text_x=0.0,
        text_y=0.0,
        page_x=40.0,
        page_y=page_y,
        font_size=9.0,
    )


def _page(page_number: int) -> PdfPage:
    body = _fragment(f"Body {page_number}.\n", 400.0)
    footer = _fragment(f"{page_number} | Chapter\n", 42.0)
    raw_text = body.text + footer.text
    return PdfPage(
        book_path="books/example.pdf",
        page_number=page_number,
        width_points=600.0,
        height_points=800.0,
        image_count=0,
        drawing_operator_count=0,
        text=raw_text,
        text_fragments=(body, footer),
        extractor="test",
    )


def test_removes_confirmed_margins_and_preserves_raw_text() -> None:
    pages = tuple(_page(page_number) for page_number in range(1, 4))
    patterns = find_repeated_margin_patterns(pages)

    reading_pages = build_pdf_reading_pages(pages, patterns)

    assert tuple(page.raw_text for page in reading_pages) == tuple(
        page.text for page in pages
    )
    assert tuple(page.reading_text for page in reading_pages) == (
        "Body 1.\n",
        "Body 2.\n",
        "Body 3.\n",
    )
    assert all(
        page.removed_margin_signatures == (patterns[0].signature,)
        for page in reading_pages
    )


def test_preserves_page_text_when_no_pattern_is_confirmed() -> None:
    page = _page(1)

    reading_page = build_pdf_reading_pages((page,))[0]

    assert reading_page.reading_text == page.text
    assert reading_page.removed_margin_signatures == ()
