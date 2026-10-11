"""Tests for deterministic PDF reading-flow chunk boundaries."""

import pytest

from src.ingestion.pdf.chunker import chunk_pdf_reading_document
from src.ingestion.pdf.reading_document import PdfReadingDocument
from src.ingestion.pdf.reading_flow import build_pdf_reading_flows
from src.ingestion.pdf.reading_text import PdfReadingPage


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
