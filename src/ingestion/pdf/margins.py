"""Find repeated text patterns near the physical edges of PDF pages."""

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import hashlib
import re

from src.ingestion.pdf.layout import PdfTextFragment
from src.ingestion.pdf.pages import PdfPage


NUMBER_PATTERN = re.compile(r"\d+")
WHITESPACE_PATTERN = re.compile(r"\s+")


class PdfMarginLocation(str, Enum):
    """Identify the physical page edge containing a repeated pattern."""

    TOP = "top"
    BOTTOM = "bottom"


@dataclass(frozen=True, slots=True)
class PdfMarginPattern:
    """Describe one repeated edge pattern without retaining its text."""

    location: PdfMarginLocation
    signature: str
    page_numbers: tuple[int, ...]
    y_ratio: float
    font_size: float


@dataclass(frozen=True, slots=True)
class _MarginCandidate:
    location: PdfMarginLocation
    signature: str
    page_number: int
    y_ratio: float
    font_size: float
    fragment_indexes: tuple[int, ...]


def find_repeated_margin_patterns(
    pages: tuple[PdfPage, ...],
    *,
    edge_ratio: float = 0.08,
    min_occurrences: int = 3,
) -> tuple[PdfMarginPattern, ...]:
    """Find recurring text lines near the top and bottom page edges."""
    if not 0 < edge_ratio < 0.5:
        message = "PDF margin edge ratio must be between 0 and 0.5"
        raise ValueError(message)
    if min_occurrences < 2:
        message = "PDF margin patterns must occur on at least two pages"
        raise ValueError(message)

    grouped: dict[
        tuple[PdfMarginLocation, str, float, float],
        list[_MarginCandidate],
    ] = defaultdict(list)
    for page in pages:
        for candidate in _page_margin_candidates(page, edge_ratio):
            key = (
                candidate.location,
                candidate.signature,
                candidate.y_ratio,
                candidate.font_size,
            )
            grouped[key].append(candidate)

    patterns = []
    for candidates in grouped.values():
        page_numbers = tuple(
            sorted({candidate.page_number for candidate in candidates})
        )
        if len(page_numbers) < min_occurrences:
            continue
        sample = candidates[0]
        patterns.append(
            PdfMarginPattern(
                location=sample.location,
                signature=sample.signature,
                page_numbers=page_numbers,
                y_ratio=sample.y_ratio,
                font_size=sample.font_size,
            )
        )

    return tuple(
        sorted(
            patterns,
            key=lambda pattern: (
                pattern.location.value,
                pattern.y_ratio,
                pattern.signature,
            ),
        )
    )


def find_margin_fragment_indexes(
    page: PdfPage,
    patterns: tuple[PdfMarginPattern, ...],
    *,
    edge_ratio: float = 0.08,
) -> tuple[int, ...]:
    """Locate fragments matched by confirmed patterns on one page."""
    matching_indexes: set[int] = set()
    pattern_keys = {
        (
            pattern.location,
            pattern.signature,
            pattern.y_ratio,
            pattern.font_size,
        )
        for pattern in patterns
        if page.page_number in pattern.page_numbers
    }
    for candidate in _page_margin_candidates(page, edge_ratio):
        candidate_key = (
            candidate.location,
            candidate.signature,
            candidate.y_ratio,
            candidate.font_size,
        )
        if candidate_key in pattern_keys:
            matching_indexes.update(candidate.fragment_indexes)
    return tuple(sorted(matching_indexes))


def _page_margin_candidates(
    page: PdfPage,
    edge_ratio: float,
) -> tuple[_MarginCandidate, ...]:
    line_fragments: dict[
        tuple[PdfMarginLocation, float, float],
        list[tuple[int, PdfTextFragment]],
    ] = defaultdict(list)
    for fragment_index, fragment in enumerate(page.text_fragments):
        location = _margin_location(
            fragment.page_y,
            page.height_points,
            edge_ratio,
        )
        if location is None:
            continue
        y_ratio = round(fragment.page_y / page.height_points, 3)
        font_size = round(fragment.font_size, 1)
        line_fragments[(location, y_ratio, font_size)].append(
            (fragment_index, fragment)
        )

    candidates = []
    for (location, y_ratio, font_size), indexed_fragments in (
        line_fragments.items()
    ):
        ordered_fragments = sorted(
            indexed_fragments,
            key=lambda item: item[1].page_x,
        )
        line = " ".join(
            fragment.text.strip()
            for _, fragment in ordered_fragments
        )
        normalized_line = _normalize_margin_text(line)
        if not normalized_line:
            continue
        candidates.append(
            _MarginCandidate(
                location=location,
                signature=_text_signature(normalized_line),
                page_number=page.page_number,
                y_ratio=y_ratio,
                font_size=font_size,
                fragment_indexes=tuple(
                    fragment_index
                    for fragment_index, _ in ordered_fragments
                ),
            )
        )
    return tuple(candidates)


def _margin_location(
    page_y: float,
    page_height: float,
    edge_ratio: float,
) -> PdfMarginLocation | None:
    if page_y < page_height * edge_ratio:
        return PdfMarginLocation.BOTTOM
    if page_y > page_height * (1 - edge_ratio):
        return PdfMarginLocation.TOP
    return None


def _normalize_margin_text(text: str) -> str:
    normalized = WHITESPACE_PATTERN.sub(" ", text).strip().casefold()
    return NUMBER_PATTERN.sub("#", normalized)


def _text_signature(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
