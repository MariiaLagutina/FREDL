"""Tests for verified PDF section boundaries in reading flows."""

from dataclasses import replace

import pytest

from src.ingestion.pdf.headings import PdfHeadingMatch, PdfHeadingStatus
from src.ingestion.pdf.outline import PdfOutlineEntry
from src.ingestion.pdf.reading_flow import build_pdf_reading_flows
from src.ingestion.pdf.reading_text import PdfReadingPage
from src.ingestion.pdf.sections import (
    build_pdf_section_intervals,
    map_pdf_section_boundaries,
)


def _page(page_number: int, text: str) -> PdfReadingPage:
    return PdfReadingPage(
        book_path="books/example.pdf",
        page_number=page_number,
        raw_text=text,
        reading_text=text,
        removed_margin_signatures=(),
    )


def _match(
    title: str,
    page_number: int,
    start: int | None,
    end: int | None,
    status: PdfHeadingStatus = PdfHeadingStatus.MATCHED,
) -> PdfHeadingMatch:
    entry = PdfOutlineEntry(
        title=title,
        section_path=("Chapter", title),
        page_number=page_number,
    )
    return PdfHeadingMatch(entry, status, start, end)


def test_maps_page_offsets_into_reading_flow() -> None:
    pages = (
        _page(1, "First page."),
        _page(2, "\nHeading\nBody."),
    )
    flow = build_pdf_reading_flows(pages)[0]
    match = _match("Heading", 2, 1, 8)

    boundary = map_pdf_section_boundaries((match,), (flow,))[0]

    second_page = flow.page_spans[1]
    assert boundary.section_path == ("Chapter", "Heading")
    assert boundary.book_path == "books/example.pdf"
    assert boundary.page_number == 2
    assert boundary.page_source_start == 1
    assert boundary.page_source_end == 8
    assert boundary.flow_index == 0
    assert boundary.flow_start == second_page.flow_start
    assert boundary.flow_end == second_page.flow_start + len("Heading")
    assert flow.text[boundary.flow_start:boundary.flow_end] == "Heading"


def test_returns_boundaries_in_source_order() -> None:
    page = _page(1, "First\nBody\nSecond")
    flow = build_pdf_reading_flows((page,))[0]
    matches = (
        _match("Second", 1, 11, 17),
        _match("First", 1, 0, 5),
    )

    boundaries = map_pdf_section_boundaries(matches, (flow,))

    assert tuple(boundary.title for boundary in boundaries) == (
        "First",
        "Second",
    )


def test_ignores_unverified_and_non_content_entries() -> None:
    page = _page(1, "Heading")
    flow = build_pdf_reading_flows((page,))[0]
    matches = (
        _match("Missing", 1, None, None, PdfHeadingStatus.NOT_FOUND),
        _match("Cover", 1, None, None, PdfHeadingStatus.NON_CONTENT),
    )

    assert map_pdf_section_boundaries(matches, (flow,)) == ()


def test_rejects_heading_outside_page_text() -> None:
    flow = build_pdf_reading_flows((_page(1, "Heading"),))[0]

    with pytest.raises(ValueError, match="inside its reading page span"):
        map_pdf_section_boundaries((_match("Heading", 1, 0, 20),), (flow,))


def test_rejects_heading_page_without_reading_flow() -> None:
    flow = build_pdf_reading_flows((_page(1, "Text"),))[0]

    with pytest.raises(ValueError, match="must belong to a reading flow"):
        map_pdf_section_boundaries(
            (_match("Heading", 2, 0, 7),),
            (flow,),
        )


def test_builds_intervals_between_headings_and_document_end() -> None:
    page = _page(1, "First\nBody\nSecond\nMore")
    flow = build_pdf_reading_flows((page,))[0]
    boundaries = map_pdf_section_boundaries(
        (
            _match("First", 1, 0, 5),
            _match("Second", 1, 11, 17),
        ),
        (flow,),
    )

    intervals = build_pdf_section_intervals(boundaries, (flow,))

    assert len(intervals) == 2
    assert intervals[0].start_flow_offset == 0
    assert intervals[0].heading_end_flow_offset == 5
    assert intervals[0].end_flow_offset == 11
    assert intervals[1].start_flow_offset == 11
    assert intervals[1].end_flow_offset == len(flow.text)
    assert intervals[1].section_path == ("Chapter", "Second")


def test_section_can_span_empty_page_and_multiple_flows() -> None:
    pages = (
        _page(1, "First section"),
        _page(2, ""),
        _page(3, "Second section"),
    )
    flows = build_pdf_reading_flows(pages)
    boundaries = map_pdf_section_boundaries(
        (
            _match("First section", 1, 0, 13),
            _match("Second section", 3, 0, 14),
        ),
        flows,
    )

    intervals = build_pdf_section_intervals(boundaries, flows)

    assert intervals[0].spans_multiple_flows is True
    assert intervals[0].start_flow_index == 0
    assert intervals[0].end_flow_index == 1
    assert intervals[0].end_flow_offset == 0
    assert intervals[1].spans_multiple_flows is False
    assert intervals[1].end_flow_offset == len(flows[1].text)


def test_no_verified_boundaries_produce_no_intervals() -> None:
    flow = build_pdf_reading_flows((_page(1, "Front matter"),))[0]

    assert build_pdf_section_intervals((), (flow,)) == ()


def test_rejects_duplicate_section_positions() -> None:
    flow = build_pdf_reading_flows((_page(1, "Heading"),))[0]
    boundary = map_pdf_section_boundaries(
        (_match("Heading", 1, 0, 7),),
        (flow,),
    )[0]
    duplicate = replace(boundary, title="Duplicate")

    with pytest.raises(ValueError, match="unique positions"):
        build_pdf_section_intervals((boundary, duplicate), (flow,))


def test_rejects_section_boundary_outside_flow() -> None:
    flow = build_pdf_reading_flows((_page(1, "Heading"),))[0]
    boundary = map_pdf_section_boundaries(
        (_match("Heading", 1, 0, 7),),
        (flow,),
    )[0]
    invalid = replace(boundary, flow_end=100)

    with pytest.raises(ValueError, match="outside its reading flow"):
        build_pdf_section_intervals((invalid,), (flow,))
