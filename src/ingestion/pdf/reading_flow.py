"""Reconstruct continuous reading text across physical PDF pages."""

from dataclasses import dataclass
from enum import Enum

from src.ingestion.pdf.reading_text import PdfReadingPage


class PdfPageJoinKind(str, Enum):
    """Explain how text was joined across a physical page boundary."""

    SPACE = "space"
    WORD_CONTINUATION = "word_continuation"


@dataclass(frozen=True, slots=True)
class PdfReadingPageSpan:
    """Map a source-page text slice into reconstructed reading text."""

    page_number: int
    flow_start: int
    flow_end: int
    source_start: int
    source_end: int


@dataclass(frozen=True, slots=True)
class PdfPageJoin:
    """Record one interpreted boundary between adjacent pages."""

    left_page_number: int
    right_page_number: int
    kind: PdfPageJoinKind
    flow_offset: int


@dataclass(frozen=True, slots=True)
class PdfReadingFlow:
    """Store reconstructed reading text with page-level provenance."""

    book_path: str
    text: str
    page_spans: tuple[PdfReadingPageSpan, ...]
    page_joins: tuple[PdfPageJoin, ...]


@dataclass(frozen=True, slots=True)
class _PageSegment:
    page_number: int
    text: str
    source_start: int
    source_end: int


def build_pdf_reading_flows(
    pages: tuple[PdfReadingPage, ...],
) -> tuple[PdfReadingFlow, ...]:
    """Join adjacent reading pages and preserve interpreted boundaries."""
    if not pages:
        return ()

    _validate_page_sequence(pages)
    flows: list[PdfReadingFlow] = []
    run: list[_PageSegment] = []

    for page in pages:
        segment = _page_segment(page)
        if segment is None:
            _finish_run(flows, run, page.book_path)
            continue
        run.append(segment)

    _finish_run(flows, run, pages[-1].book_path)
    return tuple(flows)


def _page_segment(page: PdfReadingPage) -> _PageSegment | None:
    source_start = len(page.reading_text) - len(page.reading_text.lstrip())
    source_end = len(page.reading_text.rstrip())
    text = page.reading_text[source_start:source_end]
    if not text:
        return None
    return _PageSegment(
        page_number=page.page_number,
        text=text,
        source_start=source_start,
        source_end=source_end,
    )


def _finish_run(
    flows: list[PdfReadingFlow],
    run: list[_PageSegment],
    book_path: str,
) -> None:
    if not run:
        return

    parts: list[str] = []
    spans: list[PdfReadingPageSpan] = []
    joins: list[PdfPageJoin] = []
    flow_length = 0

    for index, segment in enumerate(run):
        contribution = segment.text
        source_end = segment.source_end
        if index + 1 < len(run):
            next_segment = run[index + 1]
            join_kind = _join_kind(segment.text, next_segment.text)
            if join_kind is PdfPageJoinKind.WORD_CONTINUATION:
                contribution = contribution[:-1]
                source_end -= 1
        else:
            next_segment = None
            join_kind = None

        flow_start = flow_length
        parts.append(contribution)
        flow_length += len(contribution)
        spans.append(
            PdfReadingPageSpan(
                page_number=segment.page_number,
                flow_start=flow_start,
                flow_end=flow_length,
                source_start=segment.source_start,
                source_end=source_end,
            )
        )

        if next_segment is not None and join_kind is not None:
            joins.append(
                PdfPageJoin(
                    left_page_number=segment.page_number,
                    right_page_number=next_segment.page_number,
                    kind=join_kind,
                    flow_offset=flow_length,
                )
            )
            if join_kind is PdfPageJoinKind.SPACE:
                parts.append(" ")
                flow_length += 1

    flows.append(
        PdfReadingFlow(
            book_path=book_path,
            text="".join(parts),
            page_spans=tuple(spans),
            page_joins=tuple(joins),
        )
    )
    run.clear()


def _join_kind(left_text: str, right_text: str) -> PdfPageJoinKind:
    if left_text.endswith("-") and right_text[0].islower():
        return PdfPageJoinKind.WORD_CONTINUATION
    return PdfPageJoinKind.SPACE


def _validate_page_sequence(pages: tuple[PdfReadingPage, ...]) -> None:
    book_path = pages[0].book_path
    previous_page_number = 0
    for page in pages:
        if page.book_path != book_path:
            message = "PDF reading flow requires pages from one book"
            raise ValueError(message)
        if page.page_number <= previous_page_number:
            message = "PDF reading pages must be in increasing physical order"
            raise ValueError(message)
        previous_page_number = page.page_number
