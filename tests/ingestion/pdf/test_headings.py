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


@pytest.mark.parametrize("title", ["Cover", " Copyright ", "COPYRIGHT"])
def test_marks_navigation_only_pages_as_non_content(title: str) -> None:
    match = match_pdf_headings(
        (_entry(title, None),),
        (_page("The title does not need to appear."),),
    )[0]

    assert match.status is PdfHeadingStatus.NON_CONTENT
    assert match.source_start is None
    assert match.source_end is None


@pytest.mark.parametrize("fragment_text,expected", [
    ("Choosing metrics", PdfHeadingStatus.MATCHED),
    ("Discuss Choosing metrics here.", PdfHeadingStatus.NOT_FOUND),
])
def test_fragment_evidence_for_concatenated_title(
    fragment_text: str, expected: PdfHeadingStatus,
) -> None:
    from src.ingestion.pdf.layout import PdfTextFragment
    from src.ingestion.pdf.pages import PdfPage

    text = "Body.Choosing metricsMore body."
    fragment = PdfTextFragment(fragment_text, 0, 0, 0, 0, 20, 200, 18)
    physical = PdfPage(
        "books/test.pdf", 17, 300, 400, 0, 0, text, (fragment,), "test",
    )
    match = match_pdf_headings(
        (_entry("Choosing metrics"),), (_page(text),),
        physical_pages=(physical,),
    )[0]
    assert match.status is expected
    if expected is PdfHeadingStatus.MATCHED:
        assert text[match.source_start:match.source_end] == "Choosing metrics"


@pytest.mark.parametrize("marker,expected", [
    ("CHAPTER 2", PdfHeadingStatus.MATCHED),
    ("CHAPTER 3", PdfHeadingStatus.NOT_FOUND),
])
def test_split_title_ligature_and_verified_chapter_marker(
    marker: str, expected: PdfHeadingStatus,
) -> None:
    from src.ingestion.pdf.layout import PdfTextFragment
    from src.ingestion.pdf.pages import PdfPage

    pieces = (marker, "Effective\n", "workﬂows")
    fragments = tuple(
        PdfTextFragment(text, 0, 0, 0, 0, 20, 300 - i * 20, 18)
        for i, text in enumerate(pieces)
    )
    text = "".join(pieces) + "Body text."
    physical = PdfPage(
        "books/test.pdf", 17, 300, 400, 0, 0, text, fragments, "test",
    )
    match = match_pdf_headings(
        (_entry("Chapter 2. Effective workflows"),), (_page(text),),
        physical_pages=(physical,),
    )[0]
    assert match.status is expected
    if expected is PdfHeadingStatus.MATCHED:
        assert text[match.source_start:match.source_end] == (
            "Effective\nworkﬂows"
        )


@pytest.mark.parametrize("second_y,expected", [
    (180, PdfHeadingStatus.MATCHED),
    (200, PdfHeadingStatus.NOT_FOUND),
])
def test_omitted_line_separator_requires_geometric_evidence(
    second_y: float, expected: PdfHeadingStatus,
) -> None:
    from src.ingestion.pdf.layout import PdfTextFragment
    from src.ingestion.pdf.pages import PdfPage

    fragments = (
        PdfTextFragment("Tools for", 0, 0, 0, 0, 20, 200, 18),
        PdfTextFragment("Reliable", 0, 0, 0, 0, 20, second_y, 18),
        PdfTextFragment(" Systems", 0, 0, 0, 0, 100, second_y, 18),
    )
    text = "Before.Tools forReliable SystemsAfter."
    physical = PdfPage(
        "books/test.pdf", 17, 300, 400, 0, 0, text, fragments, "test",
    )
    match = match_pdf_headings(
        (_entry("Tools for Reliable Systems"),), (_page(text),),
        physical_pages=(physical,),
    )[0]
    assert match.status is expected
    if expected is PdfHeadingStatus.MATCHED:
        assert text[match.source_start:match.source_end] == (
            "Tools forReliable Systems"
        )


@pytest.mark.parametrize(
    ("heading_size", "body_size", "expected"),
    [
        (16.0, 10.0, PdfHeadingStatus.MATCHED),
        (10.0, 10.0, PdfHeadingStatus.AMBIGUOUS),
    ],
)
def test_repeated_title_requires_unique_largest_font(
    heading_size: float,
    body_size: float,
    expected: PdfHeadingStatus,
) -> None:
    from src.ingestion.pdf.layout import PdfTextFragment
    from src.ingestion.pdf.pages import PdfPage

    text = "Before.MetapromptingMetaprompting explains the technique."
    fragments = (
        PdfTextFragment(
            "Metaprompting", 0, 0, 0, 0, 20, 200, heading_size,
        ),
        PdfTextFragment(
            "Metaprompting", 0, 0, 0, 0, 20, 180, body_size,
        ),
    )
    physical = PdfPage(
        "books/test.pdf", 17, 300, 400, 0, 0, text, fragments, "test",
    )

    match = match_pdf_headings(
        (_entry("Metaprompting"),),
        (_page(text),),
        physical_pages=(physical,),
    )[0]

    assert match.status is expected
    if expected is PdfHeadingStatus.MATCHED:
        assert text[match.source_start:match.source_end] == "Metaprompting"
        assert match.source_start == len("Before.")
