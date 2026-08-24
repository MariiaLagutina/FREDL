"""Tests for bounded overlap inside oversized text blocks."""

import pytest

from src.ingestion import FileKind, SourceDocument, chunk_text_document
from src.ingestion.text_chunks import _overlap_start


def make_plain_document(text: str) -> SourceDocument:
    """Create one in-memory plain-text fixture."""
    return SourceDocument("notes/readme.txt", FileKind.TEXT, text)


def test_overlap_repeats_complete_lines_only_inside_oversized_block() -> None:
    """Fallback chunks overlap on a complete trailing line when possible."""
    source = (
        "111111111\n"
        "222222222\n"
        "333333333\n"
        "444444444\n"
    )
    document = make_plain_document(source)

    chunks = chunk_text_document(
        document,
        max_chunk_size=25,
        overlap_size=5,
    )

    assert [chunk.text for chunk in chunks] == [
        "111111111\n222222222\n",
        "222222222\n333333333\n",
        "333333333\n444444444\n",
    ]
    assert chunks[0].end - chunks[1].start == 10
    assert chunks[1].end - chunks[2].start == 10


def test_long_trailing_line_does_not_inflate_overlap() -> None:
    """Line alignment is skipped when it would duplicate too much text."""
    text = "a\n" + "x" * 23

    assert _overlap_start(text, 0, len(text), overlap_size=5) == 20


def test_zero_overlap_keeps_fallback_ranges_disjoint() -> None:
    """Callers can disable repeated fallback context explicitly."""
    document = make_plain_document(
        "111111111\n222222222\n333333333\n"
    )

    chunks = chunk_text_document(
        document,
        max_chunk_size=15,
        overlap_size=0,
    )

    assert all(
        first.end == second.start
        for first, second in zip(chunks, chunks[1:])
    )


@pytest.mark.parametrize("overlap", [-1, 20, 21])
def test_text_chunker_rejects_invalid_overlap(overlap: int) -> None:
    """Overlap cannot be negative or consume a complete chunk."""
    with pytest.raises(
        ValueError,
        match="Overlap size must be smaller than maximum chunk size",
    ):
        chunk_text_document(
            make_plain_document("Text."),
            max_chunk_size=20,
            overlap_size=overlap,
        )
