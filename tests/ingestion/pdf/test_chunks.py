"""Tests for PDF chunks with physical-page provenance."""

import pytest

from src.ingestion.pdf.chunks import make_pdf_chunk
from src.ingestion.pdf.reading_flow import (
    build_pdf_reading_flows,
    PdfPageJoinKind,
)
from src.ingestion.pdf.reading_text import PdfReadingPage


def _page(page_number: int, text: str) -> PdfReadingPage:
    return PdfReadingPage(
        book_path="books/example.pdf",
        page_number=page_number,
        raw_text=text,
        reading_text=text,
        removed_margin_signatures=(),
    )


def test_maps_cross_page_chunk_to_physical_page_text() -> None:
    pages = (
        _page(7, "First page."),
        _page(8, "Second page."),
    )
    flow = build_pdf_reading_flows(pages)[0]

    chunk = make_pdf_chunk(flow, flow_index=0, start=6, end=18)

    assert chunk.text == "page. Second"
    assert chunk.first_page_number == 7
    assert chunk.last_page_number == 8
    assert tuple(span.page_number for span in chunk.page_spans) == (7, 8)
    assert chunk.page_spans[0].page_source_start == 6
    assert chunk.page_spans[0].page_source_end == len("First page.")
    assert chunk.page_spans[1].page_source_start == 0
    assert chunk.page_spans[1].page_source_end == len("Second")
    assert chunk.page_joins[0].kind is PdfPageJoinKind.SPACE
    assert chunk.page_joins[0].chunk_offset == len("page.")


def test_preserves_word_continuation_inside_chunk() -> None:
    pages = (
        _page(10, "docu-"),
        _page(11, "ment"),
    )
    flow = build_pdf_reading_flows(pages)[0]

    chunk = make_pdf_chunk(flow, flow_index=2, start=0, end=8)

    assert chunk.text == "document"
    assert chunk.page_joins[0].kind is PdfPageJoinKind.WORD_CONTINUATION
    assert chunk.page_joins[0].chunk_offset == 4
    assert chunk.page_spans[0].page_source_end == 4
    assert chunk.page_spans[1].page_source_start == 0


def test_chunk_identifier_is_stable_for_the_same_source_range() -> None:
    flow = build_pdf_reading_flows((_page(1, "Stable text."),))[0]

    first = make_pdf_chunk(flow, 0, 0, len(flow.text))
    second = make_pdf_chunk(flow, 0, 0, len(flow.text))

    assert first.chunk_id == second.chunk_id
    assert first.chunk_id.startswith("pdf:")


def test_chunk_preserves_verified_section_path() -> None:
    flow = build_pdf_reading_flows((_page(1, "Section text."),))[0]

    chunk = make_pdf_chunk(
        flow,
        0,
        0,
        len(flow.text),
        section_path=("Chapter 1", "Section"),
    )

    assert chunk.section_path == ("Chapter 1", "Section")


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (-1, 2, "range must be positive"),
        (2, 2, "range must be positive"),
        (0, 20, "must not exceed"),
    ],
)
def test_rejects_invalid_chunk_range(
    start: int,
    end: int,
    message: str,
) -> None:
    flow = build_pdf_reading_flows((_page(1, "Short."),))[0]

    with pytest.raises(ValueError, match=message):
        make_pdf_chunk(flow, 0, start, end)
