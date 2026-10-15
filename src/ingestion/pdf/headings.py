"""Locate outline title candidates without guessing semantic boundaries."""

from dataclasses import dataclass, replace
from enum import Enum
import re
import statistics
import unicodedata

from src.ingestion.pdf.layout import PdfTextFragment
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


class PdfHeadingPageSource(str, Enum):
    """Explain which evidence selected the page of a matched heading.

    DESTINATION uses the outline destination page. FOLLOWING_PAGE uses the
    next page after the destination was shown to lie below all page text.
    TEXT_SEARCH is used only when the outline has no destination: the title
    was found as a display-font heading whose position agrees with the
    outline order of every other located heading.
    """

    DESTINATION = "destination"
    FOLLOWING_PAGE = "following_page"
    TEXT_SEARCH = "text_search"


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
    # Page holding the verified heading start when it is not the outline
    # destination, which happens only after layout evidence shows the
    # converter moved the heading to the following page.
    page_number: int | None = None
    # The offsets cover the first part of a heading that is completed by
    # the first lines of the next physical page.
    continues_on_next_page: bool = False
    page_source: PdfHeadingPageSource = PdfHeadingPageSource.DESTINATION

    @property
    def heading_page_number(self) -> int | None:
        """Return the physical page holding the matched heading start."""
        if self.page_number is not None:
            return self.page_number
        return self.entry.page_number


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

    body_size = _body_font_size(physical_pages)
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
        result = _match_on_page(entry, page, physical, body_size)
        next_page = by_number.get(page.page_number + 1)
        next_physical = physical_by_number.get(page.page_number + 1)
        if (
            result.status is PdfHeadingStatus.NOT_FOUND
            and physical is not None
            and next_page is not None
            and next_physical is not None
        ):
            result = (
                _match_displaced_heading(
                    entry, physical, next_page, next_physical, body_size,
                )
                or _match_page_split_heading(
                    entry, page, physical, next_page, next_physical,
                    body_size,
                )
                or result
            )
        results.append(result)
    if physical_pages:
        _search_headings_without_destination(
            results, pages, physical_by_number, body_size,
        )
    return tuple(results)


def _search_headings_without_destination(
    results: list[PdfHeadingMatch],
    pages: tuple[PdfReadingPage, ...],
    physical_by_number: dict[int, PdfPage],
    body_size: float | None,
) -> None:
    """Locate outline entries that have no destination, in place.

    A candidate is a complete-line title opening with a display-font
    fragment. The outline order never creates a candidate; it only chooses
    among verified candidates. A single candidate is accepted when its
    position agrees with every other located heading. Repeated titles are
    accepted only when exactly one candidate lies between the nearest
    located headings before and after them and that choice agrees with all
    choices made in the same round. Everything else stays unresolved with
    no page.
    """
    searchable = [
        index for index, result in enumerate(results)
        if result.status is PdfHeadingStatus.PAGE_UNAVAILABLE
        and result.entry.page_number is None
    ]
    if not searchable:
        return
    compact_pages = tuple(
        (page, _compact(page.reading_text)) for page in pages
    )
    candidates = {
        index: _display_heading_candidates(
            results[index].entry, compact_pages, physical_by_number,
            body_size,
        )
        for index in searchable
    }
    located: dict[int, tuple[int, int]] = {
        index: _match_position(result)
        for index, result in enumerate(results)
        if result.status is PdfHeadingStatus.MATCHED
    }
    unique = {
        index: options[0]
        for index, options in candidates.items()
        if len(options) == 1
    }
    accepted = {
        index: match
        for index, match in unique.items()
        if _agrees_with_order(
            index,
            _match_position(match),
            located
            | {
                other: _match_position(other_match)
                for other, other_match in unique.items()
                if other != index
            },
        )
    }
    located.update(
        (index, _match_position(match)) for index, match in accepted.items()
    )
    pending = {
        index: options
        for index, options in candidates.items()
        if len(options) > 1
    }
    while True:
        # Every round proposes against the same located set, so a choice
        # never depends on processing order. Proposals that disagree with
        # each other are all rejected and stay unresolved.
        proposals = {}
        for index, options in pending.items():
            between = [
                option for option in options
                if _agrees_with_order(index, _match_position(option), located)
            ]
            if len(between) == 1:
                proposals[index] = between[0]
        consistent = {
            index: match
            for index, match in proposals.items()
            if _agrees_with_order(
                index,
                _match_position(match),
                {
                    other: _match_position(other_match)
                    for other, other_match in proposals.items()
                    if other != index
                },
            )
        }
        if not consistent:
            break
        for index, match in consistent.items():
            accepted[index] = match
            located[index] = _match_position(match)
            del pending[index]

    for index in searchable:
        if index in accepted:
            results[index] = accepted[index]
        elif candidates[index]:
            results[index] = PdfHeadingMatch(
                results[index].entry, PdfHeadingStatus.AMBIGUOUS,
            )
        else:
            results[index] = PdfHeadingMatch(
                results[index].entry, PdfHeadingStatus.NOT_FOUND,
            )


def _display_heading_candidates(
    entry: PdfOutlineEntry,
    compact_pages: tuple[tuple[PdfReadingPage, str], ...],
    physical_by_number: dict[int, PdfPage],
    body_size: float | None,
) -> tuple[PdfHeadingMatch, ...]:
    """Return every page where the title is a display-font heading."""
    # A necessary condition only: every accepted heading contains its
    # title core once whitespace is ignored.
    core = _compact(re.sub(
        r"^(?:(?:Chapter|Part) \S+|[0-9]+)\.\s+", "", entry.title,
    ))
    candidates = []
    for page, compact_text in compact_pages:
        physical = physical_by_number.get(page.page_number)
        if physical is None or core not in compact_text:
            continue
        match = _match_on_page(
            replace(entry, page_number=page.page_number),
            page,
            physical,
            body_size,
        )
        if (
            match.status is not PdfHeadingStatus.MATCHED
            or match.source_start is None
            or match.source_end is None
            or not _opens_with_display_fragment(
                physical,
                page.reading_text[match.source_start:match.source_end],
                body_size,
            )
        ):
            continue
        candidates.append(replace(
            match,
            entry=entry,
            page_source=PdfHeadingPageSource.TEXT_SEARCH,
        ))
    return tuple(candidates)


def _compact(text: str) -> str:
    """Normalize text for a whitespace-insensitive containment check."""
    return "".join(unicodedata.normalize("NFKC", text).casefold().split())


def _opens_with_display_fragment(
    page: PdfPage,
    heading: str,
    body_size: float | None,
) -> bool:
    """Require the heading text to begin with a display-font fragment."""
    if body_size is None:
        return False
    normalized_heading = " ".join(heading.split()).casefold()
    for fragment in page.text_fragments:
        normalized = " ".join(fragment.text.split()).casefold()
        if (
            len(normalized) >= 3
            and fragment.font_size > body_size
            and normalized_heading.startswith(normalized)
        ):
            return True
    return False


def _match_position(match: PdfHeadingMatch) -> tuple[int, int]:
    """Return a matched heading's physical page and page offset."""
    page_number = match.heading_page_number
    if page_number is None or match.source_start is None:
        raise ValueError("Located PDF heading must have exact coordinates")
    return page_number, match.source_start


def _agrees_with_order(
    index: int,
    position: tuple[int, int],
    located: dict[int, tuple[int, int]],
) -> bool:
    """Require a position strictly between earlier and later outline items."""
    return all(
        (other < index and other_position < position)
        or (other > index and other_position > position)
        for other, other_position in located.items()
        if other != index
    )


def _match_on_page(
    entry: PdfOutlineEntry,
    page: PdfReadingPage,
    physical: PdfPage | None,
    body_size: float | None,
) -> PdfHeadingMatch:
    """Match one outline title on one reading page."""
    title = _page_title(entry.title, physical)
    if not title.strip():
        return PdfHeadingMatch(entry, PdfHeadingStatus.NOT_FOUND)
    title_pattern = _title_pattern(title)
    matches = _full_line_matches(title_pattern, page.reading_text)
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
    if not matches and physical is not None:
        matches = _numbered_chapter_matches(
            entry.title, page, physical, body_size,
        )
    if len(matches) == 1:
        start, end = matches[0].span("title")
        if title != entry.title and physical is not None:
            start = _marker_line_start(
                entry.title, page, physical, start, body_size,
            )
        return PdfHeadingMatch(
            entry, PdfHeadingStatus.MATCHED, start, end,
            page_number=page.page_number,
        )
    status = (
        PdfHeadingStatus.AMBIGUOUS if matches
        else PdfHeadingStatus.NOT_FOUND
    )
    return PdfHeadingMatch(entry, status)


def _full_line_matches(pattern: str, text: str) -> list[re.Match[str]]:
    """Find a title that occupies complete lines of reading text."""
    return list(re.finditer(
        r"^[^\S\r\n]*(?P<title>" + pattern + r")[^\S\r\n]*\r?$",
        text,
        re.MULTILINE,
    ))


def _body_font_size(pages: tuple[PdfPage, ...]) -> float | None:
    """Return the book's median fragment size as body-text evidence.

    A book-wide value also serves part and chapter opening pages, which
    may contain only display text.
    """
    sizes = [
        fragment.font_size
        for page in pages
        for fragment in page.text_fragments
    ]
    return statistics.median(sizes) if sizes else None


def _has_display_fragment(
    page: PdfPage,
    prefix: str,
    body_size: float | None,
) -> bool:
    """Require a fragment that starts with prefix in a display font."""
    if body_size is None:
        return False
    normalized_prefix = prefix.casefold()
    return any(
        " ".join(fragment.text.split()).casefold().startswith(
            normalized_prefix
        )
        and fragment.font_size > body_size
        for fragment in page.text_fragments
    )


def _numbered_chapter_matches(
    title: str,
    page: PdfReadingPage,
    physical: PdfPage,
    body_size: float | None,
) -> list[re.Match[str]]:
    """Match outline `N. Title` against a displayed `Chapter N. Title`.

    The chapter number must agree and the `Chapter N.` label must be set
    in a font larger than the page body, so a prose sentence that repeats
    the title is never accepted.
    """
    numbered = re.match(r"^(?P<number>[0-9]+)\.\s+(?P<rest>\S.*)$", title)
    if numbered is None:
        return []
    label = f"Chapter {numbered['number']}."
    if not _has_display_fragment(physical, label, body_size):
        return []
    pattern = (
        r"Chapter\s+" + numbered["number"] + r"\.\s+"
        + _title_pattern(numbered["rest"])
    )
    return _full_line_matches(pattern, page.reading_text)


def _marker_line_start(
    title: str,
    page: PdfReadingPage,
    physical: PdfPage,
    start: int,
    body_size: float | None,
) -> int:
    """Include a `PART N` or `CHAPTER N` line directly above the title.

    The marker must be its own display-font fragment and only whitespace
    with a line break may separate it from the verified title.
    """
    prefix = re.match(r"^(Chapter|Part) ([0-9]+|[IVXLCDM]+)\.\s+", title)
    if prefix is None:
        return start
    marker = f"{prefix[1]} {prefix[2]}"
    if body_size is None or not any(
        " ".join(fragment.text.split()).casefold() == marker.casefold()
        and fragment.font_size > body_size
        for fragment in physical.text_fragments
    ):
        return start
    preceding = re.search(
        r"(?:^|\n)[^\S\n]*(?P<marker>" + prefix[1] + r"[^\S\n]+"
        + prefix[2] + r")[^\S\n]*\n\s*\Z",
        page.reading_text[:start],
        re.IGNORECASE,
    )
    return preceding.start("marker") if preceding else start


def _match_displaced_heading(
    entry: PdfOutlineEntry,
    destination: PdfPage,
    next_page: PdfReadingPage,
    next_physical: PdfPage,
    body_size: float | None,
) -> PdfHeadingMatch | None:
    """Accept a heading moved to the next page only with layout evidence.

    The destination must point below every text fragment on its page, and
    the title must open the next page in a display font.
    """
    top = entry.destination_top
    if top is None or not destination.text_fragments or any(
        fragment.page_y <= top for fragment in destination.text_fragments
    ):
        return None
    match = _match_on_page(entry, next_page, next_physical, body_size)
    if (
        match.status is not PdfHeadingStatus.MATCHED
        or match.source_start is None
        or next_page.reading_text[:match.source_start].strip()
    ):
        return None
    opening = next_physical.text_fragments[0]
    if (
        body_size is None
        or opening.font_size <= body_size
        or not next_page.reading_text[match.source_start:].startswith(
            opening.text.strip()
        )
    ):
        return None
    return replace(match, page_source=PdfHeadingPageSource.FOLLOWING_PAGE)


def _match_page_split_heading(
    entry: PdfOutlineEntry,
    page: PdfReadingPage,
    physical: PdfPage,
    next_page: PdfReadingPage,
    next_physical: PdfPage,
    body_size: float | None,
) -> PdfHeadingMatch | None:
    """Accept a title whose lines continue at the top of the next page.

    The first part must be the last text on the destination page and the
    rest the first text on the next page, both in the same display font.
    """
    words = _page_title(entry.title, physical).split()
    candidates = []
    for split in range(1, len(words)):
        head, tail = " ".join(words[:split]), " ".join(words[split:])
        ending = re.search(
            r"(?:^|\n)[^\S\n]*(?P<title>" + _title_pattern(head)
            + r")\s*\Z",
            page.reading_text,
        )
        opening = re.match(
            r"\s*" + _title_pattern(tail) + r"[^\S\n]*(?:\n|\Z)",
            next_page.reading_text,
        )
        if ending is None or opening is None:
            continue
        head_size = _edge_font_size(
            physical.text_fragments, head, from_end=True,
        )
        tail_size = _edge_font_size(
            next_physical.text_fragments, tail, from_end=False,
        )
        if (
            head_size is None
            or head_size != tail_size
            or body_size is None
            or head_size <= body_size
        ):
            continue
        candidates.append(ending)
    if len(candidates) != 1:
        return None
    start, end = candidates[0].span("title")
    return PdfHeadingMatch(
        entry, PdfHeadingStatus.MATCHED, start, end,
        page_number=page.page_number,
        continues_on_next_page=True,
    )


def _edge_font_size(
    fragments: tuple[PdfTextFragment, ...],
    text: str,
    *,
    from_end: bool,
) -> float | None:
    """Return the single font size of the edge fragments spelling text.

    Fragments are taken from the first or last fragment of the page until
    their text, ignoring whitespace, equals the expected title part.
    """
    target = "".join(text.split())
    edge = reversed(fragments) if from_end else iter(fragments)
    collected: list[PdfTextFragment] = []
    for fragment in edge:
        collected.append(fragment)
        ordered = reversed(collected) if from_end else iter(collected)
        joined = "".join("".join(item.text.split()) for item in ordered)
        if joined == target:
            sizes = {item.font_size for item in collected}
            return sizes.pop() if len(sizes) == 1 else None
        if len(joined) >= len(target):
            return None
    return None


# A displayed heading may wrap after a hyphen that the outline keeps.
_HYPHEN_BREAK = r"-(?:[^\S\r\n]*\r?\n[^\S\r\n]*)?"
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
            parts.append(
                _HYPHEN_BREAK if token == "-" else re.escape(token)
            )
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
