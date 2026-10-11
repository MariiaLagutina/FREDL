"""Locate outline title candidates without guessing semantic boundaries."""

from dataclasses import dataclass
from enum import Enum
import re
import unicodedata

from src.ingestion.pdf.outline import PdfOutlineEntry
from src.ingestion.pdf.pages import PdfPage
from src.ingestion.pdf.reading_text import PdfReadingPage


class PdfHeadingStatus(str, Enum):
    """Describe the evidence available for one outline destination."""

    MATCHED = "matched"
    NON_CONTENT = "non_content"
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
    *,
    physical_pages: tuple[PdfPage, ...] = (),
) -> tuple[PdfHeadingMatch, ...]:
    """Match titles on their destination page, preserving uncertainty.

    Whitespace differences and standard typographic ligatures are tolerated.
    A title must occupy complete
    lines so a mention embedded in a prose sentence is not a heading match.
    Optional physical fragments support standalone titles concatenated with
    surrounding text. Multiple matches are deliberately left unresolved.
    """
    if len({page.book_path for page in pages}) > 1:
        raise ValueError("PDF heading matching requires pages from one book")
    by_number = {page.page_number: page for page in pages}
    if len(by_number) != len(pages):
        raise ValueError("PDF heading matching requires unique page numbers")

    physical_by_number = {page.page_number: page for page in physical_pages}
    if len(physical_by_number) != len(physical_pages):
        raise ValueError("Physical page numbers must be unique")
    for source_page in physical_pages:
        reading = by_number.get(source_page.page_number)
        if reading is None or reading.book_path != source_page.book_path:
            raise ValueError("Physical pages must match reading-page sources")

    results = []
    for entry in entries:
        if _is_non_content_entry(entry):
            results.append(PdfHeadingMatch(
                entry, PdfHeadingStatus.NON_CONTENT,
            ))
            continue
        page = by_number.get(entry.page_number) if (
            entry.page_number is not None
        ) else None
        if page is None:
            results.append(PdfHeadingMatch(
                entry, PdfHeadingStatus.PAGE_UNAVAILABLE,
            ))
            continue
        physical = physical_by_number.get(page.page_number)
        title = _page_title(entry.title, physical)
        if not title.strip():
            results.append(PdfHeadingMatch(entry, PdfHeadingStatus.NOT_FOUND))
            continue
        title_pattern = _title_pattern(title)
        pattern = re.compile(
            r"^[^\S\r\n]*(?P<title>" + title_pattern
            + r")[^\S\r\n]*\r?$",
            re.MULTILINE,
        )
        matches = list(pattern.finditer(page.reading_text))
        if not matches and physical is not None:
            evidence_patterns = _title_fragment_patterns(
                title_pattern, physical,
            )
            located = {}
            for evidence in evidence_patterns:
                for match in re.finditer(
                    r"(?P<title>" + evidence + r")", page.reading_text,
                ):
                    located[match.span("title")] = match
            matches = list(located.values())
        if len(matches) > 1 and physical is not None:
            matches = _resolve_repeated_title_by_font(
                matches,
                title_pattern,
                physical,
            )
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


_LIGATURES = {"ff": "ﬀ", "fi": "ﬁ", "fl": "ﬂ", "ffi": "ﬃ", "ffl": "ﬄ"}
_NON_CONTENT_TITLES = frozenset({"cover", "copyright"})


def _is_non_content_entry(entry: PdfOutlineEntry) -> bool:
    """Identify explicit navigation labels excluded from knowledge content."""
    normalized_title = " ".join(entry.title.split()).casefold()
    return normalized_title in _NON_CONTENT_TITLES


def _title_pattern(title: str) -> str:
    """Allow ligature glyphs while matching the original, unmodified text."""
    title = unicodedata.normalize("NFKC", title)
    parts = []
    for token in re.findall(r"\s+|ffi|ffl|ff|fi|fl|.", title):
        if token.isspace():
            parts.append(r"\s+")
        elif token in _LIGATURES:
            parts.append("(?:" + token + "|" + _LIGATURES[token] + ")")
        else:
            parts.append(re.escape(token))
    return "".join(parts)


def _page_title(title: str, page: PdfPage | None) -> str:
    """Drop an outline prefix only when its separate page marker exists."""
    prefix = re.match(r"^(Chapter|Part) ([0-9]+|[IVXLCDM]+)\.\s+", title)
    if prefix is None or page is None:
        return title
    marker = f"{prefix[1]} {prefix[2]}".casefold()
    if any(" ".join(f.text.split()).casefold() == marker
           for f in page.text_fragments):
        return title[prefix.end():]
    return title


def _title_fragment_patterns(pattern: str, page: PdfPage) -> tuple[str, ...]:
    """Require a complete, nearby run of equally sized PDF fragments."""
    fragments = page.text_fragments
    patterns: set[str] = set()
    for start, first in enumerate(fragments):
        text = ""
        source_parts: list[str] = []
        previous = first
        for fragment in fragments[start:start + 8]:
            if (abs(fragment.font_size - first.font_size) > 0.1
                    or abs(fragment.page_y - previous.page_y)
                    > max(first.font_size * 2, 1)):
                break
            # A descending baseline is evidence of a visual line break.
            if text and previous.page_y - fragment.page_y > 1:
                text += "\n"
            text += fragment.text
            source_parts.append(_title_pattern(fragment.text.strip()))
            if re.fullmatch(pattern, text.strip()):
                # Match either preserved or omitted extraction whitespace.
                # Only boundaries verified against the title may omit it.
                patterns.add(r"\s*".join(source_parts))
            previous = fragment
    return tuple(sorted(patterns))


def _resolve_repeated_title_by_font(
    matches: list[re.Match[str]],
    pattern: str,
    page: PdfPage,
) -> list[re.Match[str]]:
    """Resolve repeated text only when fragment order and font agree."""
    matching_fragments = [
        fragment
        for fragment in page.text_fragments
        if re.fullmatch(pattern, fragment.text.strip())
    ]
    if len(matching_fragments) != len(matches):
        return matches

    largest_size = max(fragment.font_size for fragment in matching_fragments)
    largest_indexes = [
        index
        for index, fragment in enumerate(matching_fragments)
        if fragment.font_size == largest_size
    ]
    if len(largest_indexes) != 1:
        return matches
    return [matches[largest_indexes[0]]]
