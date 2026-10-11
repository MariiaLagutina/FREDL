"""Tests for continuous PDF text with physical-page provenance."""

import pytest

from src.ingestion.pdf import PdfPage, build_pdf_text_flows


def pdf_page(
    page_number: int,
    text: str,
    *,
    book_path: str = "books/synthetic.pdf",
) -> PdfPage:
    """Create one synthetic page for text-flow tests."""
    return PdfPage(
        book_path=book_path,
        page_number=page_number,
        width_points=300,
        height_points=400,
        image_count=0,
        drawing_operator_count=0,
        text=text,
        extractor="pypdf-test",
    )


def test_build_pdf_text_flows_joins_adjacent_text_pages() -> None:
    """Adjacent page text becomes one flow without losing provenance."""
    pages = (
        pdf_page(1, "First page text."),
        pdf_page(2, "Second page text."),
    )

    flows = build_pdf_text_flows(pages)

    assert len(flows) == 1
    assert flows[0].text == "First page text.\nSecond page text."
    assert tuple(span.page_number for span in flows[0].page_spans) == (1, 2)

    for page, span in zip(pages, flows[0].page_spans, strict=True):
        assert flows[0].text[span.flow_start:span.flow_end] == page.text


def test_build_pdf_text_flows_stops_at_a_non_text_page() -> None:
    """A blank or visual-only page separates independent text flows."""
    pages = (
        pdf_page(1, "Before the blank page."),
        pdf_page(2, ""),
        pdf_page(3, "After the blank page."),
    )

    flows = build_pdf_text_flows(pages)

    assert tuple(flow.text for flow in flows) == (
        "Before the blank page.",
        "After the blank page.",
    )
    assert tuple(flow.page_spans[0].page_number for flow in flows) == (1, 3)


def test_build_pdf_text_flows_accepts_an_empty_document() -> None:
    """A document without physical pages has no text flows."""
    assert build_pdf_text_flows(()) == ()


def test_build_pdf_text_flows_rejects_multiple_books() -> None:
    """One flow cannot combine provenance from different source books."""
    pages = (
        pdf_page(1, "First book."),
        pdf_page(2, "Second book.", book_path="books/other.pdf"),
    )

    with pytest.raises(ValueError, match="pages from one book"):
        build_pdf_text_flows(pages)


def test_build_pdf_text_flows_rejects_non_increasing_pages() -> None:
    """Page order must match the physical order in the PDF."""
    pages = (
        pdf_page(2, "Second page."),
        pdf_page(1, "First page."),
    )

    with pytest.raises(ValueError, match="in increasing physical order"):
        build_pdf_text_flows(pages)
