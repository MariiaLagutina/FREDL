"""Verify heading candidates retain exact offsets and uncertainty."""

import pytest

from src.ingestion.pdf.headings import match_pdf_headings, PdfHeadingStatus
from src.ingestion.pdf.outline import PdfOutlineEntry
from src.ingestion.pdf.reading_text import PdfReadingPage


def _page(text: str) -> PdfReadingPage:
    return PdfReadingPage("books/test.pdf", 17, text, text, ())


def _entry(title: str, number: int | None = 17) -> PdfOutlineEntry:
    return PdfOutlineEntry(title, ("Chapter", title), number)


def test_multiline_heading_keeps_original_offsets() -> None:
    page = _page("Introduction\n  Choosing\nmetrics  \nBody text.")
    entry = _entry("Choosing metrics")

    match = match_pdf_headings((entry,), (page,))[0]

    assert match.status is PdfHeadingStatus.MATCHED
    assert match.entry.section_path == ("Chapter", "Choosing metrics")
    assert page.reading_text[match.source_start:match.source_end] == (
        "Choosing\nmetrics"
    )


@pytest.mark.parametrize("text,status", [
    ("Discuss Choosing metrics here.", PdfHeadingStatus.NOT_FOUND),
    ("Choosing metrics\nChoosing metrics", PdfHeadingStatus.AMBIGUOUS),
    ("Choosing other metrics", PdfHeadingStatus.NOT_FOUND),
])
def test_does_not_guess_location(text: str, status: PdfHeadingStatus) -> None:
    match = match_pdf_headings(
        (_entry("Choosing metrics"),), (_page(text),),
    )[0]

    assert match.status is status
    assert match.source_start is None
    assert match.source_end is None


@pytest.mark.parametrize("number", [None, 99])
def test_missing_destination_is_explicit(number: int | None) -> None:
    match = match_pdf_headings(
        (_entry("Title", number),), (_page("Title"),),
    )[0]
    assert match.status is PdfHeadingStatus.PAGE_UNAVAILABLE


def test_multiple_sections_on_same_page_have_distinct_ranges() -> None:
    entries = (_entry("First"), _entry("Second"))
    matches = match_pdf_headings(
        entries, (_page("First\nBody\nSecond\nBody"),),
    )
    assert tuple(m.source_start for m in matches) == (0, 11)


def test_rejects_duplicate_page_numbers() -> None:
    with pytest.raises(ValueError, match="unique page numbers"):
        match_pdf_headings((), (_page("A"), _page("B")))
