"""Build auditable page text without repeated PDF margins."""

from dataclasses import dataclass

from src.ingestion.pdf.margins import (
    find_margin_fragment_indexes,
    find_repeated_margin_patterns,
    PdfMarginPattern,
)
from src.ingestion.pdf.pages import PdfPage


@dataclass(frozen=True, slots=True)
class PdfReadingPage:
    """Keep source and reading text for one physical PDF page."""

    book_path: str
    page_number: int
    raw_text: str
    reading_text: str
    removed_margin_signatures: tuple[str, ...]


def build_pdf_reading_pages(
    pages: tuple[PdfPage, ...],
    patterns: tuple[PdfMarginPattern, ...] | None = None,
) -> tuple[PdfReadingPage, ...]:
    """Remove confirmed repeated margins while preserving raw text."""
    confirmed_patterns = (
        find_repeated_margin_patterns(pages)
        if patterns is None
        else patterns
    )
    return tuple(
        _build_reading_page(page, confirmed_patterns) for page in pages
    )


def _build_reading_page(
    page: PdfPage,
    patterns: tuple[PdfMarginPattern, ...],
) -> PdfReadingPage:
    removed_indexes = set(find_margin_fragment_indexes(page, patterns))
    removal_ranges = (
        _margin_removal_ranges(page, removed_indexes)
        if removed_indexes
        else ()
    )
    if removal_ranges is None:
        # Margin text that cannot be located in the raw text stays visible.
        reading_text = page.text
        removed_signatures: tuple[str, ...] = ()
    else:
        reading_text = _delete_ranges(page.text, removal_ranges)
        removed_signatures = tuple(
            sorted(
                pattern.signature
                for pattern in patterns
                if page.page_number in pattern.page_numbers
            )
        )
    return PdfReadingPage(
        book_path=page.book_path,
        page_number=page.page_number,
        raw_text=page.text,
        reading_text=reading_text,
        removed_margin_signatures=removed_signatures,
    )


def _fragment_spans(page: PdfPage) -> tuple[tuple[int, int], ...] | None:
    """Locate fragments in raw text, allowing only whitespace between them.

    pypdf reports line breaks and indentation as separate whitespace text
    that is not kept as a fragment, so separators live only in raw text.
    """
    spans = []
    position = 0
    for fragment in page.text_fragments:
        start = page.text.find(fragment.text, position)
        if start < 0 or page.text[position:start].strip():
            return None
        position = start + len(fragment.text)
        spans.append((start, position))
    return tuple(spans)


def _margin_removal_ranges(
    page: PdfPage,
    removed_indexes: set[int],
) -> tuple[tuple[int, int], ...] | None:
    """Return raw-text ranges covering margin runs and one adjacent gap.

    The kept gap is the one containing a line break, so surrounding lines
    stay separated exactly as pypdf extracted them.
    """
    spans = _fragment_spans(page)
    if spans is None:
        return None

    ranges = []
    for first, last in _index_runs(removed_indexes):
        start, end = spans[first][0], spans[last][1]
        left_start = spans[first - 1][1] if first > 0 else 0
        right_end = (
            spans[last + 1][0]
            if last + 1 < len(spans)
            else len(page.text) - len(page.text[end:].lstrip())
        )
        left_gap = page.text[left_start:start]
        right_gap = page.text[end:right_end]
        if first > 0 and "\n" in right_gap and "\n" not in left_gap:
            ranges.append((left_start, end))
        else:
            ranges.append((start, right_end))
    return tuple(ranges)


def _index_runs(indexes: set[int]) -> tuple[tuple[int, int], ...]:
    """Group sorted fragment indexes into consecutive inclusive runs."""
    runs: list[tuple[int, int]] = []
    for index in sorted(indexes):
        if runs and runs[-1][1] == index - 1:
            runs[-1] = (runs[-1][0], index)
        else:
            runs.append((index, index))
    return tuple(runs)


def _delete_ranges(text: str, ranges: tuple[tuple[int, int], ...]) -> str:
    """Remove ordered, non-overlapping ranges without inserting text."""
    parts = []
    position = 0
    for start, end in ranges:
        parts.append(text[position:start])
        position = end
    parts.append(text[position:])
    return "".join(parts)
