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
    PdfSectionResolution,
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


def test_ignores_non_content_and_unavailable_entries() -> None:
    page = _page(1, "Heading")
    flow = build_pdf_reading_flows((page,))[0]
    matches = (
        _match("Cover", 1, None, None, PdfHeadingStatus.NON_CONTENT),
        PdfHeadingMatch(
            PdfOutlineEntry("Lost", ("Lost",), None),
            PdfHeadingStatus.PAGE_UNAVAILABLE,
        ),
    )

    assert map_pdf_section_boundaries(matches, (flow,)) == ()


def _entry_match(
    path: tuple[str, ...],
    page_number: int,
    span: tuple[int, int] | None,
    status: PdfHeadingStatus = PdfHeadingStatus.MATCHED,
) -> PdfHeadingMatch:
    entry = PdfOutlineEntry(path[-1], path, page_number)
    if span is None:
        return PdfHeadingMatch(entry, status)
    return PdfHeadingMatch(entry, status, span[0], span[1])


def test_unmatched_chapter_closes_the_previous_verified_section() -> None:
    pages = (
        _page(1, "Summary\nEnd of chapter one."),
        _page(2, "Chapter Two\nOpening of chapter two."),
    )
    flow = build_pdf_reading_flows(pages)[0]
    matches = (
        _entry_match(("One", "Summary"), 1, (0, 7)),
        _entry_match(("Two",), 2, None, PdfHeadingStatus.NOT_FOUND),
    )

    intervals = build_pdf_section_intervals(
        map_pdf_section_boundaries(matches, (flow,)), (flow,),
    )

    verified, unresolved = intervals
    second_page = flow.page_spans[1]
    assert verified.section_path == ("One", "Summary")
    assert verified.resolution is PdfSectionResolution.VERIFIED
    assert verified.end_flow_offset == second_page.flow_start
    assert unresolved.resolution is PdfSectionResolution.UNRESOLVED
    assert unresolved.section_path == ()
    assert unresolved.outline_hint == ("Two",)
    assert unresolved.unresolved_reason is PdfHeadingStatus.NOT_FOUND
    assert unresolved.start_flow_offset == second_page.flow_start
    assert unresolved.heading_end_flow_offset == second_page.flow_start


def test_unmatched_sibling_keeps_only_shared_verified_ancestors() -> None:
    pages = (
        _page(1, "Tokenizers\nText about tokenizers."),
        _page(2, "Caching text."),
    )
    flow = build_pdf_reading_flows(pages)[0]
    matches = (
        _entry_match(("Chapter 3", "Indexing", "Tokenizers"), 1, (0, 10)),
        _entry_match(
            ("Chapter 3", "Caching"), 2, None, PdfHeadingStatus.AMBIGUOUS,
        ),
    )

    boundaries = map_pdf_section_boundaries(matches, (flow,))

    assert boundaries[1].section_path == ("Chapter 3",)
    assert boundaries[1].outline_hint == ("Chapter 3", "Caching")
    assert boundaries[1].unresolved_reason is PdfHeadingStatus.AMBIGUOUS


def test_unmatched_heading_on_same_page_starts_after_verified_heading(
) -> None:
    page = _page(1, "First\nBody text with a hidden second heading.")
    flow = build_pdf_reading_flows((page,))[0]
    matches = (
        _entry_match(("First",), 1, (0, 5)),
        _entry_match(("Second",), 1, None, PdfHeadingStatus.NOT_FOUND),
    )

    boundaries = map_pdf_section_boundaries(matches, (flow,))

    assert tuple(b.resolution for b in boundaries) == (
        PdfSectionResolution.VERIFIED,
        PdfSectionResolution.UNRESOLVED,
    )
    assert boundaries[1].flow_start == 5
    assert flow.text[boundaries[0].flow_start:boundaries[0].flow_end] == (
        "First"
    )


def test_verified_heading_supersedes_unresolved_at_same_position() -> None:
    pages = (_page(1, "Intro text."), _page(2, "Next\nBody."))
    flow = build_pdf_reading_flows(pages)[0]
    matches = (
        _entry_match(("Missing",), 2, None, PdfHeadingStatus.NOT_FOUND),
        _entry_match(("Next",), 2, (0, 4)),
    )

    boundaries = map_pdf_section_boundaries(matches, (flow,))

    assert len(boundaries) == 1
    assert boundaries[0].section_path == ("Next",)


def test_no_unresolved_boundary_against_page_order() -> None:
    pages = (_page(1, "Early text."), _page(2, "Late\nBody."))
    flow = build_pdf_reading_flows(pages)[0]
    matches = (
        _entry_match(("Late",), 2, (0, 4)),
        _entry_match(("Earlier",), 1, None, PdfHeadingStatus.NOT_FOUND),
    )

    boundaries = map_pdf_section_boundaries(matches, (flow,))

    assert tuple(b.section_path for b in boundaries) == (("Late",),)


def test_displaced_heading_uses_its_verified_page() -> None:
    pages = (_page(30, "Previous text."), _page(31, "Heading\nBody."))
    flow = build_pdf_reading_flows(pages)[0]
    match = PdfHeadingMatch(
        PdfOutlineEntry("Heading", ("Heading",), 30),
        PdfHeadingStatus.MATCHED,
        0,
        7,
        page_number=31,
    )

    boundary = map_pdf_section_boundaries((match,), (flow,))[0]

    assert boundary.page_number == 31
    assert flow.text[boundary.flow_start:boundary.flow_end] == "Heading"


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
