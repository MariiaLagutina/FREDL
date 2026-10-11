"""Locate outline title candidates without guessing semantic boundaries."""

from dataclasses import dataclass
from enum import Enum
import re

from src.ingestion.pdf.outline import PdfOutlineEntry
from src.ingestion.pdf.reading_text import PdfReadingPage


class PdfHeadingStatus(str, Enum):
    """Describe the evidence available for one outline destination."""

    MATCHED = "matched"
    NOT_FOUND = "not_found"
    AMBIGUOUS = "ambiguous"
    PAGE_UNAVAILABLE = "page_unavailable"


@dataclass(frozen=True, slots=True)
class PdfHeadingMatch:
    """Keep an outline entry and an optional exact reading-page range.

    A unique full-line title match is a candidate location, not proof of
    typography or a complete chapter boundary. Offsets are half-open and
    refer to reading_text, not raw extracted text.
    """

    entry: PdfOutlineEntry
    status: PdfHeadingStatus
    source_start: int | None = None
    source_end: int | None = None


def match_pdf_headings(
    entries: tuple[PdfOutlineEntry, ...],
    pages: tuple[PdfReadingPage, ...],
) -> tuple[PdfHeadingMatch, ...]:
    """Match titles on their destination page, preserving uncertainty.

    Only whitespace differences are tolerated. A title must occupy complete
    lines so a mention embedded in a prose sentence is not a heading match.
    Multiple matches are deliberately left unresolved.
    """
    if len({page.book_path for page in pages}) > 1:
        raise ValueError("PDF heading matching requires pages from one book")
    by_number = {page.page_number: page for page in pages}
    if len(by_number) != len(pages):
        raise ValueError("PDF heading matching requires unique page numbers")

    results = []
    for entry in entries:
        page = by_number.get(entry.page_number) if (
            entry.page_number is not None
        ) else None
        if page is None:
            results.append(PdfHeadingMatch(
                entry, PdfHeadingStatus.PAGE_UNAVAILABLE,
            ))
            continue
        words = entry.title.split()
        if not words:
            results.append(PdfHeadingMatch(entry, PdfHeadingStatus.NOT_FOUND))
            continue
        title_pattern = r"\s+".join(re.escape(word) for word in words)
        pattern = re.compile(
            r"^[^\S\r\n]*(?P<title>" + title_pattern
            + r")[^\S\r\n]*\r?$",
            re.MULTILINE,
        )
        matches = list(pattern.finditer(page.reading_text))
        if len(matches) == 1:
            start, end = matches[0].span("title")
            results.append(PdfHeadingMatch(
                entry, PdfHeadingStatus.MATCHED, start, end,
            ))
        else:
            status = (
                PdfHeadingStatus.AMBIGUOUS if matches
                else PdfHeadingStatus.NOT_FOUND
            )
            results.append(PdfHeadingMatch(entry, status))
    return tuple(results)
