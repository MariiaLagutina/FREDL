"""Split reconstructed PDF reading flows at observable text boundaries."""

import re

from src.ingestion.pdf.chunks import make_pdf_chunk, PdfChunk
from src.ingestion.pdf.reading_document import PdfReadingDocument
from src.ingestion.pdf.reading_flow import (
    PdfPageJoinKind,
    PdfReadingFlow,
)


DEFAULT_MAX_CHUNK_SIZE = 1600
DEFAULT_MIN_CHUNK_SIZE = 600
SENTENCE_BOUNDARY = re.compile(r"[.!?][\"')\]]*\s+")
LINE_BOUNDARY = re.compile(r"\r\n|\r|\n")
WORD_BOUNDARY = re.compile(r"\s+")


def chunk_pdf_reading_document(
    document: PdfReadingDocument,
    *,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
) -> tuple[PdfChunk, ...]:
    """Create bounded PDF chunks while preferring readable boundaries."""
    _validate_chunk_sizes(max_chunk_size, min_chunk_size)
    chunks = []
    for flow_index, flow in enumerate(document.reading_flows):
        chunks.extend(
            _chunk_flow(
                flow,
                flow_index,
                max_chunk_size,
                min_chunk_size,
            )
        )
    return tuple(chunks)


def _chunk_flow(
    flow: PdfReadingFlow,
    flow_index: int,
    max_chunk_size: int,
    min_chunk_size: int,
) -> list[PdfChunk]:
    chunks = []
    start = 0
    while start < len(flow.text):
        limit = min(start + max_chunk_size, len(flow.text))
        if limit == len(flow.text):
            end = limit
        else:
            end = _select_boundary(
                flow,
                start,
                limit,
                min(start + min_chunk_size, limit),
            )
        chunks.append(make_pdf_chunk(flow, flow_index, start, end))
        start = end
    return chunks


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
