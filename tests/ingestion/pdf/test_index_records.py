"""Tests for index-neutral PDF retrieval records."""

import pytest

from src.ingestion.pdf.chunks import make_pdf_chunk
from src.ingestion.pdf.index_records import build_pdf_index_records
from src.ingestion.pdf.reading_flow import build_pdf_reading_flows
from src.ingestion.pdf.reading_text import PdfReadingPage


def _page(page_number: int, text: str) -> PdfReadingPage:
    return PdfReadingPage(
        book_path="books/example.pdf",
        page_number=page_number,
        raw_text=text,
        reading_text=text,
        removed_margin_signatures=(),
    )


def test_builds_index_record_without_changing_verified_evidence() -> None:
    flow = build_pdf_reading_flows(
        (
            _page(7, "First page."),
            _page(8, "Second page."),
        )
    )[0]
    chunk = make_pdf_chunk(
        flow,
        flow_index=0,
        start=6,
        end=18,
        section_path=("Chapter 1", "Section"),
    )

    record = build_pdf_index_records((chunk,))[0]

    assert record.record_id == chunk.chunk_id
    assert record.text == "page. Second"
    assert record.book_path == "books/example.pdf"
    assert record.section_path == ("Chapter 1", "Section")
    assert (record.flow_start, record.flow_end) == (6, 18)
    assert record.page_spans == chunk.page_spans
    assert record.first_page_number == 7
    assert record.last_page_number == 8
    assert record.source_type == "pdf"


def test_preserves_chunk_order_for_deterministic_index_builds() -> None:
    flow = build_pdf_reading_flows((_page(1, "First. Second."),))[0]
    first = make_pdf_chunk(flow, 0, 0, 6)
    second = make_pdf_chunk(flow, 0, 7, len(flow.text))

    records = build_pdf_index_records((first, second))

    assert tuple(record.record_id for record in records) == (
        first.chunk_id,
        second.chunk_id,
    )


def test_rejects_duplicate_chunk_identifiers() -> None:
    flow = build_pdf_reading_flows((_page(1, "Stable text."),))[0]
    chunk = make_pdf_chunk(flow, 0, 0, len(flow.text))

    with pytest.raises(ValueError, match="unique chunk identifiers"):
        build_pdf_index_records((chunk, chunk))
