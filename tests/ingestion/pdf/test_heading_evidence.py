"""Verify heading conventions accepted only with layout evidence."""

from src.ingestion.pdf.headings import (
    match_pdf_headings,
    PdfHeadingPageSource,
    PdfHeadingStatus,
)
from src.ingestion.pdf.layout import PdfTextFragment
from src.ingestion.pdf.outline import PdfOutlineEntry
from src.ingestion.pdf.pages import PdfPage
from src.ingestion.pdf.reading_text import PdfReadingPage


BOOK = "books/test.pdf"
BODY_SIZE = 10.0


def _fragment(text: str, page_y: float, size: float) -> PdfTextFragment:
    return PdfTextFragment(text, 0, 0, 0, 0, 40, page_y, size)


def _pages(
    page_number: int,
    *parts: tuple[str, float, float] | str,
) -> tuple[PdfReadingPage, PdfPage]:
    """Build a page from (text, y, size) fragments and raw separators."""
    text = "".join(
        part if isinstance(part, str) else part[0] for part in parts
    )
    fragments = tuple(
        _fragment(*part) for part in parts if not isinstance(part, str)
    )
    reading = PdfReadingPage(BOOK, page_number, text, text, ())
    physical = PdfPage(
        BOOK, page_number, 600, 800, 0, 0, text, fragments, "test",
    )
    return reading, physical


def _body_page(page_number: int) -> tuple[PdfReadingPage, PdfPage]:
    """Provide book-wide body-size evidence."""
    lines = tuple(
        part
        for index in range(9)
        for part in ((f"Body line {index}.", 700.0 - index * 12, BODY_SIZE),
                     "\n")
    )
    return _pages(page_number, *lines)


def _match(
    title: str,
    destination: int,
    *pages: tuple[PdfReadingPage, PdfPage],
    path: tuple[str, ...] | None = None,
    top: float | None = None,
) -> tuple[PdfHeadingStatus, str | None, int | None, bool]:
    entry = PdfOutlineEntry(title, path or (title,), destination, top)
    all_pages = (*pages, _body_page(99))
    match = match_pdf_headings(
        (entry,),
        tuple(reading for reading, _ in all_pages),
        physical_pages=tuple(physical for _, physical in all_pages),
    )[0]
    text = None
    if match.source_start is not None and match.source_end is not None:
        page = next(
            reading for reading, _ in all_pages
            if reading.page_number == match.heading_page_number
        )
        text = page.reading_text[match.source_start:match.source_end]
    return (
        match.status,
        text,
        match.heading_page_number,
        match.continues_on_next_page,
    )


def test_numbered_outline_title_matches_displayed_chapter_label() -> None:
    page = _pages(
        7,
        ("Chapter 1. ", 690, 40), ("Planning Data", 690, 40), "\n",
        ("Pipelines for Teams", 655, 40), "\n",
        ("Body text starts here.", 600, BODY_SIZE),
    )

    status, text, page_number, _ = _match(
        "1. Planning Data Pipelines for Teams", 7, page,
    )

    assert status is PdfHeadingStatus.MATCHED
    assert text == "Chapter 1. Planning Data\nPipelines for Teams"
    assert page_number == 7


def test_chapter_label_in_body_font_is_not_a_heading() -> None:
    page = _pages(
        7,
        ("Chapter 1. Planning Data Pipelines for Teams", 690, BODY_SIZE),
        "\n",
        ("Body text.", 670, BODY_SIZE),
    )

    status, *_ = _match("1. Planning Data Pipelines for Teams", 7, page)

    assert status is PdfHeadingStatus.NOT_FOUND


def test_chapter_number_must_agree() -> None:
    page = _pages(
        7,
        ("Chapter 2. ", 690, 40), ("Planning Data Pipelines", 690, 40),
        "\n", ("Body.", 600, BODY_SIZE),
    )

    status, *_ = _match("1. Planning Data Pipelines", 7, page)

    assert status is PdfHeadingStatus.NOT_FOUND


def test_prose_mention_of_a_chapter_is_not_a_heading() -> None:
    page = _pages(
        7,
        ("Chapter 1. ", 690, 40), ("Other Title", 690, 40), "\n",
        ("As Chapter 1. Sketching explains, we start small.", 600,
         BODY_SIZE),
    )

    status, *_ = _match("1. Sketching", 7, page)

    assert status is PdfHeadingStatus.NOT_FOUND


def test_heading_may_wrap_after_a_hyphen() -> None:
    page = _pages(
        120,
        ("Chapter 10. ", 690, 40), ("Shipping a Cloud-", 690, 40), "\n",
        ("Native Service", 655, 40), "\n",
        ("Body.", 600, BODY_SIZE),
    )

    status, text, *_ = _match(
        "10. Shipping a Cloud-Native Service", 120, page,
    )

    assert status is PdfHeadingStatus.MATCHED
    assert text == "Chapter 10. Shipping a Cloud-\nNative Service"


def _destination_page(lowest_y: float) -> tuple[PdfReadingPage, PdfPage]:
    return _pages(
        30,
        ("Earlier paragraph ends here.", 700, BODY_SIZE), "\n",
        ("Questions are welcome.", lowest_y, BODY_SIZE),
    )


def _next_page_with_heading(
    *,
    heading_first: bool = True,
    heading_size: float = 30,
) -> tuple[PdfReadingPage, PdfPage]:
    heading = ("Scope of This Guide", 690, heading_size)
    body = ("The guide covers the basics.", 637, BODY_SIZE)
    if heading_first:
        return _pages(31, heading, "\n", body)
    return _pages(31, body, "\n", heading)


def test_displaced_heading_is_accepted_with_layout_evidence() -> None:
    status, text, page_number, _ = _match(
        "Scope of This Guide", 30,
        _destination_page(420), _next_page_with_heading(),
        top=400.0,
    )

    assert status is PdfHeadingStatus.MATCHED
    assert text == "Scope of This Guide"
    assert page_number == 31


def test_displacement_requires_destination_below_all_text() -> None:
    status, *_ = _match(
        "Scope of This Guide", 30,
        _destination_page(380), _next_page_with_heading(),
        top=400.0,
    )

    assert status is PdfHeadingStatus.NOT_FOUND


def test_displacement_requires_declared_destination_top() -> None:
    status, *_ = _match(
        "Scope of This Guide", 30,
        _destination_page(420), _next_page_with_heading(),
    )

    assert status is PdfHeadingStatus.NOT_FOUND


def test_displaced_heading_must_open_the_next_page() -> None:
    status, *_ = _match(
        "Scope of This Guide", 30,
        _destination_page(420), _next_page_with_heading(heading_first=False),
        top=400.0,
    )

    assert status is PdfHeadingStatus.NOT_FOUND


def test_displaced_heading_requires_display_font() -> None:
    status, *_ = _match(
        "Scope of This Guide", 30,
        _destination_page(420),
        _next_page_with_heading(heading_size=BODY_SIZE),
        top=400.0,
    )

    assert status is PdfHeadingStatus.NOT_FOUND


def _split_pages(
    head_size: float = 30.0,
    tail_size: float = 30.0,
    *,
    tail_first: bool = True,
) -> tuple[tuple[PdfReadingPage, PdfPage], tuple[PdfReadingPage, PdfPage]]:
    first = _pages(
        50,
        ("Body before the heading.", 300, BODY_SIZE), "\n",
        ("Comparing Retrieval Strategies at", 120, head_size),
    )
    tail = ("Scale", 690, tail_size)
    body = ("Scale changes the trade-offs.", 650, BODY_SIZE)
    second = _pages(
        51, *((tail, "\n", body) if tail_first else (body, "\n", tail)),
    )
    return first, second


def test_heading_split_across_pages_is_matched_on_first_page() -> None:
    status, text, page_number, continues = _match(
        "Comparing Retrieval Strategies at Scale", 50,
        *_split_pages(), top=140.0,
    )

    assert status is PdfHeadingStatus.MATCHED
    assert text == "Comparing Retrieval Strategies at"
    assert page_number == 50
    assert continues is True


def test_split_heading_requires_equal_display_fonts() -> None:
    status, *_ = _match(
        "Comparing Retrieval Strategies at Scale", 50,
        *_split_pages(tail_size=20), top=140.0,
    )

    assert status is PdfHeadingStatus.NOT_FOUND


def test_split_heading_requires_tail_at_next_page_start() -> None:
    status, *_ = _match(
        "Comparing Retrieval Strategies at Scale", 50,
        *_split_pages(tail_first=False), top=140.0,
    )

    assert status is PdfHeadingStatus.NOT_FOUND


def test_part_marker_directly_above_title_joins_the_heading() -> None:
    page = _pages(
        40, ("PART IV", 560, 18.0), "\n", ("Operations", 530, 28.0),
    )

    status, text, *_ = _match("Part IV. Operations", 40, page)

    assert status is PdfHeadingStatus.MATCHED
    assert text == "PART IV\nOperations"


def test_marker_separated_by_body_text_stays_outside_heading() -> None:
    page = _pages(
        45,
        ("CHAPTER 5", 560, 16.0), "\n",
        ("An epigraph line.", 545, BODY_SIZE), "\n",
        ("Deployment", 530, 24.0), "\n",
        ("Body.", 500, BODY_SIZE),
    )

    status, text, *_ = _match("Chapter 5. Deployment", 45, page)

    assert status is PdfHeadingStatus.MATCHED
    assert text == "Deployment"


def test_marker_in_body_font_stays_outside_heading() -> None:
    page = _pages(
        45,
        ("CHAPTER 5", 560, BODY_SIZE), "\n",
        ("Deployment", 530, 24.0), "\n",
        ("Body.", 500, BODY_SIZE),
    )

    status, text, *_ = _match("Chapter 5. Deployment", 45, page)

    assert status is PdfHeadingStatus.MATCHED
    assert text == "Deployment"


def test_displaced_heading_records_its_page_source() -> None:
    pages = (_destination_page(420), _next_page_with_heading(), _body_page(99))
    entry = PdfOutlineEntry(
        "Scope of This Guide", ("Scope of This Guide",), 30, 400.0,
    )

    match = match_pdf_headings(
        (entry,),
        tuple(reading for reading, _ in pages),
        physical_pages=tuple(physical for _, physical in pages),
    )[0]

    assert match.page_source is PdfHeadingPageSource.FOLLOWING_PAGE
