"""Tests for reconstructed text across physical PDF pages."""

import pytest

from src.ingestion.pdf.reading_flow import (
    build_pdf_reading_flows,
    PdfPageJoinKind,
)
from src.ingestion.pdf.reading_text import PdfReadingPage


def _page(
    page_number: int,
    text: str,
    *,
    book_path: str = "books/example.pdf",
) -> PdfReadingPage:
    return PdfReadingPage(
        book_path=book_path,
        page_number=page_number,
        raw_text=text,
        reading_text=text,
        removed_margin_signatures=(),
    )


def test_joins_adjacent_pages_with_a_space() -> None:
    pages = (
        _page(1, "A sentence continues\n"),
        _page(2, "on the next page."),
    )

    flow = build_pdf_reading_flows(pages)[0]

    assert flow.text == "A sentence continues on the next page."
    assert flow.page_joins[0].kind is PdfPageJoinKind.SPACE
    assert flow.page_joins[0].flow_offset == len("A sentence continues")


def test_reconstructs_a_word_split_between_pages() -> None:
    pages = (
        _page(10, "A reconstructed docu-\n"),
        _page(11, "  ment keeps its provenance."),
    )

    flow = build_pdf_reading_flows(pages)[0]

    assert flow.text == "A reconstructed document keeps its provenance."
    assert flow.page_joins[0].kind is PdfPageJoinKind.WORD_CONTINUATION
    assert flow.page_joins[0].flow_offset == len("A reconstructed docu")
    assert flow.page_spans[0].source_end == len(
        "A reconstructed docu"
    )
    assert flow.page_spans[1].source_start == 2


def test_blank_reading_page_starts_a_new_flow() -> None:
    pages = (
        _page(1, "Before."),
        _page(2, " \n"),
        _page(3, "After."),
    )

    flows = build_pdf_reading_flows(pages)

    assert tuple(flow.text for flow in flows) == ("Before.", "After.")
    assert all(flow.page_joins == () for flow in flows)


def test_rejects_pages_from_multiple_books() -> None:
    pages = (
        _page(1, "First."),
        _page(2, "Second.", book_path="books/other.pdf"),
    )

    with pytest.raises(ValueError, match="pages from one book"):
        build_pdf_reading_flows(pages)
