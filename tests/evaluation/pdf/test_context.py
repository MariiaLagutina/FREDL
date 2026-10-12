"""Tests for expanding PDF hits with adjacent reading context."""

import pytest

from src.evaluation.pdf.context import find_pdf_neighbor_context
from src.ingestion.pdf.chunks import PdfChunkPageSpan
from src.ingestion.pdf.index_records import PdfIndexRecord


def _record(
    record_id: str,
    start: int,
    end: int,
    *,
    section: str = "Section",
) -> PdfIndexRecord:
    return PdfIndexRecord(
        record_id=record_id,
        text=record_id,
        book_path="books/example.pdf",
        section_path=(section,),
        flow_index=0,
        flow_start=start,
        flow_end=end,
        page_spans=(PdfChunkPageSpan(1, 0, 1, start, end),),
    )


def test_returns_touching_neighbors_in_the_same_section() -> None:
    records = (
        _record("before", 0, 10),
        _record("hit", 10, 20),
        _record("after", 20, 30),
        _record("next-section", 30, 40, section="Next"),
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


def test_rejects_invalid_context_inputs() -> None:
    records = (_record("known", 0, 10),)

    with pytest.raises(ValueError, match="top_k must be positive"):
        find_pdf_neighbor_context(("known",), records, top_k=0)
    with pytest.raises(ValueError, match="unknown record"):
        find_pdf_neighbor_context(("missing",), records)
