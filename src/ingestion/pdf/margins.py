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


def _page_margin_candidates(
    page: PdfPage,
    edge_ratio: float,
) -> tuple[_MarginCandidate, ...]:
    line_fragments: dict[
        tuple[PdfMarginLocation, float, float],
        list[PdfTextFragment],
    ] = defaultdict(list)
    for fragment in page.text_fragments:
        location = _margin_location(
            fragment.page_y,
            page.height_points,
            edge_ratio,
        )
        if location is None:
            continue
        y_ratio = round(fragment.page_y / page.height_points, 3)
        font_size = round(fragment.font_size, 1)
        line_fragments[(location, y_ratio, font_size)].append(fragment)

    candidates = []
    for (location, y_ratio, font_size), fragments in line_fragments.items():
        line = " ".join(
            fragment.text.strip()
            for fragment in sorted(fragments, key=lambda item: item.page_x)
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
