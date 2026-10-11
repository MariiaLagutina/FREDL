"""Tests for repeated PDF margin-pattern detection."""

import pytest

from src.ingestion.pdf.layout import PdfTextFragment
from src.ingestion.pdf.margins import (
    find_repeated_margin_patterns,
    PdfMarginLocation,
)
from src.ingestion.pdf.pages import PdfPage


def _fragment(text: str, page_x: float, page_y: float) -> PdfTextFragment:
    return PdfTextFragment(
        text=text,
        container_x=0.0,
        container_y=0.0,
        text_x=0.0,
        text_y=0.0,
        page_x=page_x,
        page_y=page_y,
        font_size=9.0,
    )


def _page(
    page_number: int,
    *fragments: PdfTextFragment,
) -> PdfPage:
    return PdfPage(
        book_path="books/example.pdf",
        page_number=page_number,
        width_points=600.0,
        height_points=800.0,
        image_count=0,
        drawing_operator_count=0,
        text="",
        text_fragments=fragments,
        extractor="test",
    )


def test_finds_numbered_footer_repeated_across_pages() -> None:
    pages = tuple(
        _page(
            page_number,
            _fragment(str(page_number), 40.0, 42.0),
            _fragment("| Chapter", 55.0, 42.0),
            _fragment("Body text", 40.0, 400.0),
        )
        for page_number in range(1, 4)
    )

    patterns = find_repeated_margin_patterns(pages)

    assert len(patterns) == 1
    assert patterns[0].location is PdfMarginLocation.BOTTOM
    assert patterns[0].page_numbers == (1, 2, 3)
    assert patterns[0].y_ratio == 0.052
    assert patterns[0].font_size == 9.0


def test_ignores_unique_edge_text_and_body_text() -> None:
    pages = (
        _page(1, _fragment("Unique note", 40.0, 42.0)),
        _page(2, _fragment("Repeated body", 40.0, 400.0)),
        _page(3, _fragment("Repeated body", 40.0, 400.0)),
        _page(4, _fragment("Repeated body", 40.0, 400.0)),
    )

    assert find_repeated_margin_patterns(pages) == ()


@pytest.mark.parametrize("edge_ratio", [0.0, 0.5])
def test_rejects_invalid_edge_ratio(edge_ratio: float) -> None:
    with pytest.raises(ValueError, match="edge ratio"):
        find_repeated_margin_patterns((), edge_ratio=edge_ratio)


def test_requires_patterns_to_repeat() -> None:
    with pytest.raises(ValueError, match="at least two pages"):
        find_repeated_margin_patterns((), min_occurrences=1)
