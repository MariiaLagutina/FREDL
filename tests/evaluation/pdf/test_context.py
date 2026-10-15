"""Tests for expanding PDF hits with adjacent reading context."""

import pytest

from src.evaluation.pdf.context import find_pdf_neighbor_context
from src.ingestion.pdf.chunker import chunk_pdf_reading_document
from src.ingestion.pdf.chunks import PdfChunkPageSpan
from src.ingestion.pdf.headings import PdfHeadingMatch, PdfHeadingStatus
from src.ingestion.pdf.index_records import (
    build_pdf_index_records,
    PdfIndexRecord,
)
from src.ingestion.pdf.outline import PdfOutlineEntry
from src.ingestion.pdf.reading_document import PdfReadingDocument
from src.ingestion.pdf.reading_flow import build_pdf_reading_flows
from src.ingestion.pdf.reading_text import PdfReadingPage
from src.ingestion.pdf.sections import (
    build_pdf_section_intervals,
    map_pdf_section_boundaries,
    PdfSectionResolution,
)


BOOK = "books/example.pdf"


def _record(
    record_id: str,
    start: int,
    end: int,
    *,
    section: str = "Section",
    interval: str | None = "interval-a",
) -> PdfIndexRecord:
    return PdfIndexRecord(
        record_id=record_id,
        text=record_id,
        book_path=BOOK,
        section_path=(section,),
        flow_index=0,
        flow_start=start,
        flow_end=end,
        page_spans=(PdfChunkPageSpan(1, 0, 1, start, end),),
        section_interval_id=interval,
    )


def test_returns_touching_neighbors_in_the_same_section() -> None:
    records = (
        _record("before", 0, 10),
        _record("hit", 10, 20),
        _record("after", 20, 30),
        _record("next-section", 30, 40, section="Next", interval="b"),
    )

    context = find_pdf_neighbor_context(("hit",), records, top_k=1)

    assert context == ("before", "after")


def test_deduplicates_neighbors_and_excludes_ranked_records() -> None:
    records = (
        _record("first", 0, 10),
        _record("middle", 10, 20),
        _record("last", 20, 30),
    )

    context = find_pdf_neighbor_context(
        ("first", "last"), records, top_k=2
    )

    assert context == ("middle",)


def test_keeps_a_neighbor_that_ranks_below_the_top_hits() -> None:
    records = (
        _record("hit", 0, 10),
        _record("later-ranked-neighbor", 10, 20),
        _record("unrelated", 30, 40),
    )

    context = find_pdf_neighbor_context(
        ("hit", "unrelated", "later-ranked-neighbor"),
        records,
        top_k=1,
    )

    assert context == ("later-ranked-neighbor",)


def test_equal_paths_in_different_intervals_are_not_context() -> None:
    records = (
        _record("before", 0, 10, interval="a"),
        _record("hit", 10, 20, interval="b"),
        _record("after", 20, 30, interval="c"),
    )

    assert find_pdf_neighbor_context(("hit",), records, top_k=1) == ()


def test_records_without_interval_identity_receive_no_context() -> None:
    records = (
        _record("before", 0, 10, interval=None),
        _record("hit", 10, 20, interval=None),
    )

    assert find_pdf_neighbor_context(("hit",), records, top_k=1) == ()


def test_rejects_invalid_context_inputs() -> None:
    records = (_record("known", 0, 10),)

    with pytest.raises(ValueError, match="top_k must be positive"):
        find_pdf_neighbor_context(("known",), records, top_k=0)
    with pytest.raises(ValueError, match="unknown record"):
        find_pdf_neighbor_context(("missing",), records)


def _document(*page_texts: str) -> PdfReadingDocument:
    reading_pages = tuple(
        PdfReadingPage(BOOK, index, text, text, ())
        for index, text in enumerate(page_texts, start=1)
    )
    return PdfReadingDocument(
        book_path=BOOK,
        physical_pages=(),
        margin_patterns=(),
        reading_pages=reading_pages,
        reading_flows=build_pdf_reading_flows(reading_pages),
    )


def _matched(
    title: str,
    path: tuple[str, ...],
    page: int,
    length: int,
) -> PdfHeadingMatch:
    return PdfHeadingMatch(
        PdfOutlineEntry(title, path, page),
        PdfHeadingStatus.MATCHED,
        0,
        length,
    )


def _unresolved(
    title: str,
    path: tuple[str, ...],
    page: int,
) -> PdfHeadingMatch:
    return PdfHeadingMatch(
        PdfOutlineEntry(title, path, page),
        PdfHeadingStatus.NOT_FOUND,
    )


def _records(
    document: PdfReadingDocument,
    *matches: PdfHeadingMatch,
    max_chunk_size: int = 400,
) -> tuple[PdfIndexRecord, ...]:
    intervals = build_pdf_section_intervals(
        map_pdf_section_boundaries(matches, document.reading_flows),
        document.reading_flows,
    )
    chunks = chunk_pdf_reading_document(
        document,
        section_intervals=intervals,
        max_chunk_size=max_chunk_size,
        min_chunk_size=10,
    )
    return build_pdf_index_records(chunks)


def _context(records: tuple[PdfIndexRecord, ...], index: int) -> set[int]:
    context = find_pdf_neighbor_context(
        (records[index].record_id,), records, top_k=1,
    )
    positions = {record.record_id: i for i, record in enumerate(records)}
    return {positions[record_id] for record_id in context}


def _guide_document() -> PdfReadingDocument:
    return _document(
        "Intro\nThe guide starts with an overview.",
        "Chapter A opens with its own material.",
        "Chapter B opens with different material.",
    )


def test_unresolved_chapters_with_shared_ancestor_stay_separate() -> None:
    records = _records(
        _guide_document(),
        _matched("Intro", ("Guide", "Intro"), 1, 5),
        _unresolved("Chapter A", ("Guide", "Chapter A"), 2),
        _unresolved("Chapter B", ("Guide", "Chapter B"), 3),
    )

    assert [record.section_path for record in records] == [
        ("Guide", "Intro"), ("Guide",), ("Guide",),
    ]
    assert records[1].section_resolution is PdfSectionResolution.UNRESOLVED
    assert records[1].flow_end == records[2].flow_start
    assert _context(records, 1) == set()
    assert _context(records, 2) == set()


def test_repeated_unresolved_hints_stay_separate() -> None:
    records = _records(
        _guide_document(),
        _matched("Intro", ("Guide", "Intro"), 1, 5),
        _unresolved("Exercises", ("Guide", "Exercises"), 2),
        _unresolved("Exercises", ("Guide", "Exercises"), 3),
    )

    first, second = records[1], records[2]
    assert (
        first.section_path, first.section_resolution, first.section_hint,
        first.unresolved_reason,
    ) == (
        second.section_path, second.section_resolution, second.section_hint,
        second.unresolved_reason,
    )
    assert first.section_interval_id != second.section_interval_id
    assert _context(records, 1) == set()
    assert _context(records, 2) == set()


def test_front_matter_is_not_context_for_an_unresolved_interval() -> None:
    records = _records(
        _document(
            "Front matter introduces the guide.",
            "Opening text of a chapter that was not verified.",
        ),
        _unresolved("Chapter A", ("Chapter A",), 2),
    )

    assert [record.section_resolution for record in records] == [
        PdfSectionResolution.NONE, PdfSectionResolution.UNRESOLVED,
    ]
    assert records[0].section_path == records[1].section_path == ()
    assert _context(records, 0) == set()
    assert _context(records, 1) == set()


def test_verified_sections_with_identical_paths_stay_separate() -> None:
    records = _records(
        _document(
            "Summary\nThe first summary closes one part.",
            "Summary\nThe second summary closes another part.",
        ),
        _matched("Summary", ("Guide", "Summary"), 1, 7),
        _matched("Summary", ("Guide", "Summary"), 2, 7),
    )

    assert records[0].section_path == records[1].section_path
    assert records[0].section_interval_id != records[1].section_interval_id
    assert _context(records, 0) == set()
    assert _context(records, 1) == set()


def _single_section_records() -> tuple[PdfIndexRecord, ...]:
    sentences = " ".join(f"Sentence number {index}." for index in range(12))
    return _records(
        _document(f"Overview\n{sentences}"),
        _matched("Overview", ("Overview",), 1, 8),
        max_chunk_size=80,
    )


def test_records_within_one_verified_interval_expand_both_ways() -> None:
    records = _single_section_records()

    assert len(records) >= 3
    assert len({record.section_interval_id for record in records}) == 1
    assert _context(records, 1) == {0, 2}


def test_first_and_last_records_expand_only_inward() -> None:
    records = _single_section_records()

    assert _context(records, 0) == {1}
    assert _context(records, len(records) - 1) == {len(records) - 2}


def test_interval_identities_are_stable_across_runs() -> None:
    def run() -> tuple[PdfIndexRecord, ...]:
        return _records(
            _guide_document(),
            _matched("Intro", ("Guide", "Intro"), 1, 5),
            _unresolved("Chapter A", ("Guide", "Chapter A"), 2),
            _unresolved("Chapter B", ("Guide", "Chapter B"), 3),
        )

    first, second = run(), run()

    assert first == second
    identities = [record.section_interval_id for record in first]
    assert all(identity is not None for identity in identities)
    assert len(set(identities)) == 3
