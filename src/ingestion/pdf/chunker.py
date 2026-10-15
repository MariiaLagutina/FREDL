"""Split reconstructed PDF reading flows at observable text boundaries."""

from dataclasses import dataclass
import re

from src.ingestion.pdf.chunks import make_pdf_chunk, PdfChunk
from src.ingestion.pdf.reading_document import PdfReadingDocument
from src.ingestion.pdf.reading_flow import (
    PdfPageJoinKind,
    PdfReadingFlow,
)
from src.ingestion.pdf.sections import (
    pdf_section_interval_id,
    PdfSectionInterval,
    PdfSectionResolution,
)


DEFAULT_MAX_CHUNK_SIZE = 1600
DEFAULT_MIN_CHUNK_SIZE = 600
SENTENCE_BOUNDARY = re.compile(r"[.!?][\"')\]]*\s+")
LINE_BOUNDARY = re.compile(r"\r\n|\r|\n")
WORD_BOUNDARY = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class _FlowSectionRange:
    start: int
    end: int
    interval: PdfSectionInterval | None


def chunk_pdf_reading_document(
    document: PdfReadingDocument,
    *,
    section_intervals: tuple[PdfSectionInterval, ...] = (),
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
) -> tuple[PdfChunk, ...]:
    """Create bounded PDF chunks while preferring readable boundaries."""
    _validate_chunk_sizes(max_chunk_size, min_chunk_size)
    _validate_section_intervals(document, section_intervals)
    chunks = []
    for flow_index, flow in enumerate(document.reading_flows):
        for section_range in _flow_section_ranges(
            flow_index,
            flow,
            section_intervals,
        ):
            chunks.extend(
                _chunk_flow_range(
                    flow,
                    flow_index,
                    section_range,
                    max_chunk_size,
                    min_chunk_size,
                )
            )
    return tuple(chunks)


def _chunk_flow_range(
    flow: PdfReadingFlow,
    flow_index: int,
    section_range: _FlowSectionRange,
    max_chunk_size: int,
    min_chunk_size: int,
) -> list[PdfChunk]:
    chunks = []
    start = section_range.start
    while start < section_range.end:
        limit = _chunk_limit(
            start,
            section_range.end,
            max_chunk_size,
            min_chunk_size,
        )
        if limit == section_range.end:
            end = limit
        else:
            end = _select_boundary(
                flow,
                start,
                limit,
                min(start + min_chunk_size, limit),
            )
        interval = section_range.interval
        chunks.append(
            make_pdf_chunk(
                flow,
                flow_index,
                start,
                end,
                section_path=interval.section_path if interval else (),
                section_resolution=(
                    interval.resolution if interval
                    else PdfSectionResolution.NONE
                ),
                section_hint=interval.outline_hint if interval else (),
                unresolved_reason=(
                    interval.unresolved_reason if interval else None
                ),
                section_interval_id=(
                    interval.interval_id if interval
                    else pdf_section_interval_id(flow.book_path, 0, 0)
                ),
            )
        )
        start = end
    return chunks


def _chunk_limit(
    start: int,
    section_end: int,
    max_chunk_size: int,
    min_chunk_size: int,
) -> int:
    limit = min(start + max_chunk_size, section_end)
    short_tail_size = section_end - limit
    remaining_size = section_end - start
    if (
        0 < short_tail_size < min_chunk_size
        and remaining_size >= 2 * min_chunk_size
    ):
        return section_end - min_chunk_size
    return limit


def _flow_section_ranges(
    flow_index: int,
    flow: PdfReadingFlow,
    intervals: tuple[PdfSectionInterval, ...],
) -> tuple[_FlowSectionRange, ...]:
    cut_points = {0, len(flow.text)}
    cut_points.update(
        interval.start_flow_offset
        for interval in intervals
        if interval.start_flow_index == flow_index
    )
    ordered_points = sorted(cut_points)
    return tuple(
        _FlowSectionRange(
            start=start,
            end=end,
            interval=_interval_at(intervals, flow_index, start),
        )
        for start, end in zip(
            ordered_points,
            ordered_points[1:],
        )
        if start < end
    )


def _interval_at(
    intervals: tuple[PdfSectionInterval, ...],
    flow_index: int,
    flow_offset: int,
) -> PdfSectionInterval | None:
    coordinate = (flow_index, flow_offset)
    for interval in intervals:
        start = (interval.start_flow_index, interval.start_flow_offset)
        end = (interval.end_flow_index, interval.end_flow_offset)
        if start <= coordinate < end:
            return interval
    return None


def _validate_section_intervals(
    document: PdfReadingDocument,
    intervals: tuple[PdfSectionInterval, ...],
) -> None:
    previous_end: tuple[int, int] | None = None
    for interval in intervals:
        if interval.book_path != document.book_path:
            message = "PDF section interval must match its reading document"
            raise ValueError(message)
        start = (interval.start_flow_index, interval.start_flow_offset)
        end = (interval.end_flow_index, interval.end_flow_offset)
        if not start < end:
            message = "PDF section interval must have a positive range"
            raise ValueError(message)
        if not (
            0 <= interval.start_flow_index < len(document.reading_flows)
            and 0 <= interval.end_flow_index < len(document.reading_flows)
        ):
            message = "PDF section interval flow index is outside the document"
            raise ValueError(message)
        start_flow = document.reading_flows[interval.start_flow_index]
        end_flow = document.reading_flows[interval.end_flow_index]
        if not (
            0 <= interval.start_flow_offset < len(start_flow.text)
            and 0 <= interval.end_flow_offset <= len(end_flow.text)
        ):
            message = "PDF section interval offset is outside its reading flow"
            raise ValueError(message)
        if previous_end is not None and start < previous_end:
            message = "PDF section intervals must not overlap"
            raise ValueError(message)
        previous_end = end


def _select_boundary(
    flow: PdfReadingFlow,
    start: int,
    limit: int,
    preferred_start: int,
) -> int:
    sentence_end = _last_match_end(
        SENTENCE_BOUNDARY,
        flow.text,
        preferred_start,
        limit,
    )
    if sentence_end is not None:
        return sentence_end

    page_end = _last_page_boundary(flow, preferred_start, limit)
    if page_end is not None:
        return page_end

    for pattern in (LINE_BOUNDARY, WORD_BOUNDARY):
        boundary = _last_match_end(
            pattern,
            flow.text,
            preferred_start,
            limit,
        )
        if boundary is not None:
            return boundary

    earlier_word_end = _last_match_end(
        WORD_BOUNDARY,
        flow.text,
        start + 1,
        limit,
    )
    return earlier_word_end if earlier_word_end is not None else limit


def _last_match_end(
    pattern: re.Pattern[str],
    text: str,
    start: int,
    end: int,
) -> int | None:
    boundary = None
    for match in pattern.finditer(text, start, end):
        boundary = match.end()
    return boundary


def _last_page_boundary(
    flow: PdfReadingFlow,
    start: int,
    end: int,
) -> int | None:
    boundaries = [
        join.flow_offset + 1
        for join in flow.page_joins
        if join.kind is PdfPageJoinKind.SPACE
        and start <= join.flow_offset < end
    ]
    return boundaries[-1] if boundaries else None


def _validate_chunk_sizes(max_chunk_size: int, min_chunk_size: int) -> None:
    if max_chunk_size <= 0:
        message = "Maximum PDF chunk size must be positive"
        raise ValueError(message)
    if min_chunk_size <= 0 or min_chunk_size > max_chunk_size:
        message = (
            "Minimum PDF chunk size must be positive and not exceed maximum"
        )
        raise ValueError(message)
