"""Tests for deterministic PDF reading-flow chunk boundaries."""

import pytest

from src.ingestion.pdf.chunker import chunk_pdf_reading_document
from src.ingestion.pdf.headings import PdfHeadingMatch, PdfHeadingStatus
from src.ingestion.pdf.outline import PdfOutlineEntry
from src.ingestion.pdf.reading_document import PdfReadingDocument
from src.ingestion.pdf.reading_flow import build_pdf_reading_flows
from src.ingestion.pdf.reading_text import PdfReadingPage
from src.ingestion.pdf.sections import (
    build_pdf_section_intervals,
    map_pdf_section_boundaries,
    PdfSectionInterval,
)


def _document(*page_texts: str) -> PdfReadingDocument:
    reading_pages = tuple(
        PdfReadingPage(
            book_path="books/example.pdf",
            page_number=index,
            raw_text=text,
            reading_text=text,
            removed_margin_signatures=(),
        )
        for index, text in enumerate(page_texts, start=1)
    )
    return PdfReadingDocument(
        book_path="books/example.pdf",
        physical_pages=(),
        margin_patterns=(),
        reading_pages=reading_pages,
        reading_flows=build_pdf_reading_flows(reading_pages),
    )


def _section_intervals(
    document: PdfReadingDocument,
    headings: tuple[
        tuple[str, tuple[str, ...], int, int, int],
        ...,
    ],
) -> tuple[PdfSectionInterval, ...]:
    matches = tuple(
        PdfHeadingMatch(
            entry=PdfOutlineEntry(title, section_path, page_number),
            status=PdfHeadingStatus.MATCHED,
            source_start=start,
            source_end=end,
        )
        for title, section_path, page_number, start, end in headings
    )
    boundaries = map_pdf_section_boundaries(
        matches,
        document.reading_flows,
    )
    return build_pdf_section_intervals(
        boundaries,
        document.reading_flows,
    )


def test_prefers_sentence_boundaries_within_size_limit() -> None:
    document = _document(
        "First complete sentence. Second complete sentence. "
        "Third complete sentence."
    )

    chunks = chunk_pdf_reading_document(
        document,
        max_chunk_size=55,
        min_chunk_size=20,
    )

    assert tuple(chunk.text for chunk in chunks) == (
        "First complete sentence. Second complete sentence. ",
        "Third complete sentence.",
    )
    assert all(len(chunk.text) <= 55 for chunk in chunks)


def test_uses_page_boundary_when_no_sentence_boundary_exists() -> None:
    document = _document(
        "first page without punctuation",
        "second page without punctuation",
    )

    chunks = chunk_pdf_reading_document(
        document,
        max_chunk_size=45,
        min_chunk_size=20,
    )

    assert chunks[0].text == "first page without punctuation "
    assert chunks[0].first_page_number == 1
    assert chunks[1].first_page_number == 2


def test_does_not_treat_reconstructed_word_as_page_boundary() -> None:
    document = _document(
        "A docu-",
        "ment without an early sentence ending keeps going.",
    )

    chunks = chunk_pdf_reading_document(
        document,
        max_chunk_size=25,
        min_chunk_size=8,
    )

    assert chunks[0].text == "A document without an "
    assert chunks[0].last_page_number == 2


def test_chunk_ids_remain_stable_for_repeated_runs() -> None:
    document = _document("One sentence. Two sentences. Three sentences.")

    first = chunk_pdf_reading_document(
        document,
        max_chunk_size=25,
        min_chunk_size=10,
    )
    second = chunk_pdf_reading_document(
        document,
        max_chunk_size=25,
        min_chunk_size=10,
    )

    assert tuple(chunk.chunk_id for chunk in first) == tuple(
        chunk.chunk_id for chunk in second
    )


def test_section_starts_force_chunk_boundaries_and_assign_paths() -> None:
    text = "Front matter. Heading\nShort section. Next\nOther."
    document = _document(text)
    intervals = _section_intervals(
        document,
        (
            (
                "Heading",
                ("Chapter", "Heading"),
                1,
                text.index("Heading"),
                text.index("Heading") + len("Heading"),
            ),
            (
                "Next",
                ("Chapter", "Next"),
                1,
                text.index("Next"),
                text.index("Next") + len("Next"),
            ),
        ),
    )

    chunks = chunk_pdf_reading_document(
        document,
        section_intervals=intervals,
        max_chunk_size=100,
        min_chunk_size=50,
    )

    assert tuple(chunk.text for chunk in chunks) == (
        "Front matter. ",
        "Heading\nShort section. ",
        "Next\nOther.",
    )
    assert tuple(chunk.section_path for chunk in chunks) == (
        (),
        ("Chapter", "Heading"),
        ("Chapter", "Next"),
    )


def test_section_path_continues_across_empty_page_and_flows() -> None:
    document = _document("Heading and text.", "", "Continued text.")
    intervals = _section_intervals(
        document,
        (("Heading", ("Chapter", "Heading"), 1, 0, 7),),
    )

    chunks = chunk_pdf_reading_document(
        document,
        section_intervals=intervals,
    )

    assert len(chunks) == 2
    assert all(
        chunk.section_path == ("Chapter", "Heading")
        for chunk in chunks
    )


def test_balances_short_tail_inside_section() -> None:
    text = "word " * 12
    document = _document(text)
    intervals = _section_intervals(
        document,
        (("word", ("Chapter", "Section"), 1, 0, 4),),
    )

    chunks = chunk_pdf_reading_document(
        document,
        section_intervals=intervals,
        max_chunk_size=50,
        min_chunk_size=20,
    )

    assert len(chunks) == 2
    assert all(len(chunk.text) >= 20 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == text.rstrip()
    assert all(
        chunk.section_path == ("Chapter", "Section")
        for chunk in chunks
    )


def test_keeps_short_section_separate_from_next_section() -> None:
    text = "First short. Second section is long enough to stand alone."
    document = _document(text)
    second_start = text.index("Second")
    intervals = _section_intervals(
        document,
        (
            ("First", ("Chapter", "First"), 1, 0, 5),
            (
                "Second",
                ("Chapter", "Second"),
                1,
                second_start,
                second_start + len("Second"),
            ),
        ),
    )

    chunks = chunk_pdf_reading_document(
        document,
        section_intervals=intervals,
        max_chunk_size=50,
        min_chunk_size=20,
    )

    assert chunks[0].text == "First short. "
    assert chunks[0].section_path == ("Chapter", "First")
    assert chunks[1].section_path == ("Chapter", "Second")


@pytest.mark.parametrize(
    ("maximum", "minimum"),
    [(0, 0), (100, 0), (100, 101)],
)
def test_rejects_invalid_chunk_sizes(maximum: int, minimum: int) -> None:
    document = _document("Text.")

    with pytest.raises(ValueError):
        chunk_pdf_reading_document(
            document,
            max_chunk_size=maximum,
            min_chunk_size=minimum,
        )
