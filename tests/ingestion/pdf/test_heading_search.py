"""Verify the text-search fallback for outlines without destinations."""

from src.ingestion.pdf.headings import (
    match_pdf_headings,
    PdfHeadingMatch,
    PdfHeadingPageSource,
    PdfHeadingStatus,
)
from src.ingestion.pdf.layout import PdfTextFragment
from src.ingestion.pdf.outline import PdfOutlineEntry
from src.ingestion.pdf.pages import PdfPage
from src.ingestion.pdf.reading_flow import build_pdf_reading_flows
from src.ingestion.pdf.reading_text import PdfReadingPage
from src.ingestion.pdf.sections import (
    build_pdf_section_intervals,
    map_pdf_section_boundaries,
    PdfSectionResolution,
)


BOOK = "books/test.pdf"
BODY = 10.0
HEADING = 18.0
Part = tuple[str, float] | str


def _page(page_number: int, *parts: Part) -> tuple[PdfReadingPage, PdfPage]:
    """Build a page from (text, font size) fragments and separators."""
    text = "".join(
        part if isinstance(part, str) else part[0] for part in parts
    )
    fragments = tuple(
        PdfTextFragment(part[0], 0, 0, 0, 0, 40, 700 - index * 20, part[1])
        for index, part in enumerate(
            part for part in parts if not isinstance(part, str)
        )
    )
    return (
        PdfReadingPage(BOOK, page_number, text, text, ()),
        PdfPage(BOOK, page_number, 600, 800, 0, 0, text, fragments, "test"),
    )


def _body(page_number: int, *lines: str) -> tuple[PdfReadingPage, PdfPage]:
    parts: list[Part] = []
    for line in lines or ("Body text line.",) * 4:
        parts.extend(((line, BODY), "\n"))
    return _page(page_number, *parts)


def _entries(*paths: tuple[str, ...]) -> tuple[PdfOutlineEntry, ...]:
    return tuple(PdfOutlineEntry(path[-1], path, None) for path in paths)


def _match(
    entries: tuple[PdfOutlineEntry, ...],
    *pages: tuple[PdfReadingPage, PdfPage],
) -> tuple[PdfHeadingMatch, ...]:
    return match_pdf_headings(
        entries,
        tuple(reading for reading, _ in pages),
        physical_pages=tuple(physical for _, physical in pages),
    )


def test_locates_display_headings_without_destinations() -> None:
    pages = (
        _page(1, ("Basics", HEADING), "\n", ("Body text line.", BODY)),
        _page(2, ("Advanced", HEADING), "\n", ("Body text line.", BODY)),
        _body(3),
    )

    matches = _match(_entries(("Basics",), ("Advanced",)), *pages)

    assert tuple(m.status for m in matches) == (
        PdfHeadingStatus.MATCHED, PdfHeadingStatus.MATCHED,
    )
    assert tuple(m.heading_page_number for m in matches) == (1, 2)
    assert all(
        m.page_source is PdfHeadingPageSource.TEXT_SEARCH for m in matches
    )
    assert all(m.entry.page_number is None for m in matches)


def test_title_in_body_font_is_not_a_candidate() -> None:
    pages = (
        _body(1, "Basics", "A paragraph about basics."),
        _body(2),
    )

    match = _match(_entries(("Basics",)), *pages)[0]

    assert match.status is PdfHeadingStatus.NOT_FOUND
    assert match.heading_page_number is None


def test_table_of_contents_line_is_not_a_candidate() -> None:
    pages = (
        _page(1, ("Basics . . . . . . 3", HEADING), "\n"),
        _page(3, ("Basics", HEADING), "\n", ("Body text line.", BODY)),
        _body(4),
    )

    match = _match(_entries(("Basics",)), *pages)[0]

    assert match.status is PdfHeadingStatus.MATCHED
    assert match.heading_page_number == 3


def test_repeated_title_without_anchors_stays_ambiguous() -> None:
    pages = (
        _page(1, ("Summary", HEADING), "\n", ("Body text line.", BODY)),
        _page(2, ("Summary", HEADING), "\n", ("Body text line.", BODY)),
        _body(3),
    )

    match = _match(_entries(("One", "Summary")), *pages)[0]

    assert match.status is PdfHeadingStatus.AMBIGUOUS
    assert match.heading_page_number is None


def test_repeated_title_is_chosen_only_between_located_headings() -> None:
    pages = (
        _page(1, ("Chapter One", HEADING), "\n", ("Text.", BODY), "\n",
              ("Summary", HEADING), "\n", ("Text.", BODY)),
        _page(2, ("Chapter Two", HEADING), "\n", ("Text.", BODY), "\n",
              ("Summary", HEADING), "\n", ("Text.", BODY)),
        _body(3),
    )
    entries = _entries(
        ("Chapter One",),
        ("Chapter One", "Summary"),
        ("Chapter Two",),
        ("Chapter Two", "Summary"),
    )

    matches = _match(entries, *pages)

    assert all(m.status is PdfHeadingStatus.MATCHED for m in matches)
    assert tuple(m.heading_page_number for m in matches) == (1, 1, 2, 2)


def test_unique_candidate_against_outline_order_is_rejected() -> None:
    pages = (
        _page(1, ("Later Topic", HEADING), "\n", ("Text.", BODY)),
        _page(2, ("Early Topic", HEADING), "\n", ("Text.", BODY)),
        _body(3),
    )

    matches = _match(_entries(("Early Topic",), ("Later Topic",)), *pages)

    assert tuple(m.status for m in matches) == (
        PdfHeadingStatus.AMBIGUOUS, PdfHeadingStatus.AMBIGUOUS,
    )


def test_search_needs_physical_layout_evidence() -> None:
    reading, _ = _page(1, ("Basics", HEADING), "\n", ("Body.", BODY))

    match = match_pdf_headings(_entries(("Basics",)), (reading,))[0]

    assert match.status is PdfHeadingStatus.PAGE_UNAVAILABLE


def test_partial_structure_marks_unlocated_heading_unresolved() -> None:
    pages = (
        _page(1, ("Chapter One", HEADING), "\n",
              ("Intro text for chapter one.", BODY)),
        _body(2, "A subsection title that was never typeset as a heading."),
        _page(3, ("Chapter Two", HEADING), "\n", ("Text.", BODY)),
    )
    entries = _entries(
        ("Chapter One",),
        ("Chapter One", "Missing Subsection"),
        ("Chapter Two",),
    )
    matches = _match(entries, *pages)
    flows = build_pdf_reading_flows(tuple(reading for reading, _ in pages))

    intervals = build_pdf_section_intervals(
        map_pdf_section_boundaries(matches, flows), flows,
    )

    assert matches[1].status is PdfHeadingStatus.NOT_FOUND
    assert tuple(i.resolution for i in intervals) == (
        PdfSectionResolution.VERIFIED,
        PdfSectionResolution.UNRESOLVED,
        PdfSectionResolution.VERIFIED,
    )
    unresolved = intervals[1]
    assert unresolved.section_path == ("Chapter One",)
    assert unresolved.outline_hint == ("Chapter One", "Missing Subsection")
    assert unresolved.start_flow_offset == len("Chapter One")
    assert intervals[0].end_flow_offset == len("Chapter One")
    assert intervals[2].section_path == ("Chapter Two",)


def _heading_page(page_number: int, title: str) -> tuple[
    PdfReadingPage, PdfPage,
]:
    return _page(
        page_number, (title, HEADING), "\n", ("Body text line.", BODY),
    )


def test_conflicting_choices_in_one_round_all_stay_ambiguous() -> None:
    # Within the anchors on pages 1 and 5, Alpha can only be page 3 and
    # Beta only page 2, which contradicts the outline order Alpha < Beta.
    pages = (
        _heading_page(1, "Start"),
        _heading_page(2, "Beta"),
        _heading_page(3, "Alpha"),
        _heading_page(5, "End"),
        _heading_page(8, "Alpha"),
        _heading_page(9, "Beta"),
    )
    entries = _entries(("Start",), ("Alpha",), ("Beta",), ("End",))

    matches = _match(entries, *pages)

    assert tuple(m.status for m in matches) == (
        PdfHeadingStatus.MATCHED,
        PdfHeadingStatus.AMBIGUOUS,
        PdfHeadingStatus.AMBIGUOUS,
        PdfHeadingStatus.MATCHED,
    )
    assert matches[1].heading_page_number is None
    assert matches[2].heading_page_number is None


def test_repeated_titles_with_a_missing_neighbor_anchor_stay_ambiguous(
) -> None:
    pages = (
        _page(1, ("Chapter One", HEADING), "\n", ("Text.", BODY), "\n",
              ("Summary", HEADING), "\n", ("Text.", BODY)),
        _page(2, ("Chapter Two", BODY), "\n", ("Text.", BODY), "\n",
              ("Summary", HEADING), "\n", ("Text.", BODY)),
        _body(3),
    )
    entries = _entries(
        ("Chapter One",),
        ("Chapter One", "Summary"),
        ("Chapter Two",),
        ("Chapter Two", "Summary"),
    )

    matches = _match(entries, *pages)

    assert tuple(m.status for m in matches) == (
        PdfHeadingStatus.MATCHED,
        PdfHeadingStatus.AMBIGUOUS,
        PdfHeadingStatus.NOT_FOUND,
        PdfHeadingStatus.AMBIGUOUS,
    )


def test_search_result_does_not_depend_on_page_input_order() -> None:
    pages = (
        _page(1, ("Chapter One", HEADING), "\n", ("Text.", BODY), "\n",
              ("Summary", HEADING), "\n", ("Text.", BODY)),
        _page(2, ("Chapter Two", HEADING), "\n", ("Text.", BODY), "\n",
              ("Summary", HEADING), "\n", ("Text.", BODY)),
        _body(3),
    )
    entries = _entries(
        ("Chapter One",),
        ("Chapter One", "Summary"),
        ("Chapter Two",),
        ("Chapter Two", "Summary"),
    )

    forward = _match(entries, *pages)
    backward = _match(entries, *reversed(pages))

    assert forward == backward
    assert _match(entries, *pages) == forward
