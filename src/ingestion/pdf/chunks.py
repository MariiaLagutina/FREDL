"""Create PDF chunks with exact reading-flow and page provenance."""

from dataclasses import dataclass
import hashlib

from src.ingestion.pdf.reading_flow import (
    PdfPageJoinKind,
    PdfReadingFlow,
)


@dataclass(frozen=True, slots=True)
class PdfChunkPageSpan:
    """Map part of a chunk back to one physical page's reading text."""

    page_number: int
    chunk_start: int
    chunk_end: int
    page_source_start: int
    page_source_end: int


@dataclass(frozen=True, slots=True)
class PdfChunkPageJoin:
    """Record an interpreted page boundary contained in a chunk."""

    left_page_number: int
    right_page_number: int
    kind: PdfPageJoinKind
    chunk_offset: int


@dataclass(frozen=True, slots=True)
class PdfChunk:
    """Store reconstructed PDF text with auditable source locations."""

    chunk_id: str
    book_path: str
    flow_index: int
    flow_start: int
    flow_end: int
    text: str
    page_spans: tuple[PdfChunkPageSpan, ...]
    page_joins: tuple[PdfChunkPageJoin, ...]
    section_path: tuple[str, ...] = ()

    @property
    def first_page_number(self) -> int:
        """Return the first physical page contributing source text."""
        return self.page_spans[0].page_number

    @property
    def last_page_number(self) -> int:
        """Return the last physical page contributing source text."""
        return self.page_spans[-1].page_number


def make_pdf_chunk(
    flow: PdfReadingFlow,
    flow_index: int,
    start: int,
    end: int,
    *,
    section_path: tuple[str, ...] = (),
) -> PdfChunk:
    """Create one chunk from a half-open reading-flow range."""
    if flow_index < 0:
        message = "PDF flow index must not be negative"
        raise ValueError(message)
    if start < 0 or end <= start:
        message = "PDF chunk range must be positive"
        raise ValueError(message)
    if end > len(flow.text):
        message = "PDF chunk end must not exceed reading flow length"
        raise ValueError(message)

    text = flow.text[start:end]
    if not text.strip():
        message = "PDF chunk must contain non-whitespace text"
        raise ValueError(message)
    if not isinstance(section_path, tuple) or any(
        not isinstance(title, str) or not title.strip()
        for title in section_path
    ):
        message = "PDF chunk section path must contain non-empty titles"
        raise ValueError(message)

    page_spans = _chunk_page_spans(flow, start, end)
    if not page_spans:
        message = "PDF chunk must include physical-page text"
        raise ValueError(message)

    return PdfChunk(
        chunk_id=_chunk_id(flow.book_path, flow_index, start, end),
        book_path=flow.book_path,
        flow_index=flow_index,
        flow_start=start,
        flow_end=end,
        text=text,
        page_spans=page_spans,
        page_joins=_chunk_page_joins(flow, start, end),
        section_path=section_path,
    )


def _chunk_page_spans(
    flow: PdfReadingFlow,
    start: int,
    end: int,
) -> tuple[PdfChunkPageSpan, ...]:
    spans = []
    for page_span in flow.page_spans:
        overlap_start = max(start, page_span.flow_start)
        overlap_end = min(end, page_span.flow_end)
        if overlap_start >= overlap_end:
            continue
        spans.append(
            PdfChunkPageSpan(
                page_number=page_span.page_number,
                chunk_start=overlap_start - start,
                chunk_end=overlap_end - start,
                page_source_start=(
                    page_span.source_start
                    + overlap_start
                    - page_span.flow_start
                ),
                page_source_end=(
                    page_span.source_start
                    + overlap_end
                    - page_span.flow_start
                ),
            )
        )
    return tuple(spans)


def _chunk_page_joins(
    flow: PdfReadingFlow,
    start: int,
    end: int,
) -> tuple[PdfChunkPageJoin, ...]:
    return tuple(
        PdfChunkPageJoin(
            left_page_number=join.left_page_number,
            right_page_number=join.right_page_number,
            kind=join.kind,
            chunk_offset=join.flow_offset - start,
        )
        for join in flow.page_joins
        if start <= join.flow_offset < end
    )


def _chunk_id(
    book_path: str,
    flow_index: int,
    start: int,
    end: int,
) -> str:
    identity = f"{book_path}\0{flow_index}\0{start}\0{end}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"pdf:{digest}"
