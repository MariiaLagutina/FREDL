"""Prepare verified PDF chunks for local retrieval indexes."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from src.ingestion.pdf.chunks import PdfChunk, PdfChunkPageSpan
from src.ingestion.pdf.headings import PdfHeadingStatus
from src.ingestion.pdf.sections import PdfSectionResolution


@dataclass(frozen=True, slots=True)
class PdfIndexRecord:
    """Keep searchable PDF text separate from exact source metadata."""

    record_id: str
    text: str
    book_path: str
    section_path: tuple[str, ...]
    flow_index: int
    flow_start: int
    flow_end: int
    page_spans: tuple[PdfChunkPageSpan, ...]
    source_type: Literal["pdf"] = "pdf"
    section_resolution: PdfSectionResolution = PdfSectionResolution.NONE
    section_hint: tuple[str, ...] = ()
    unresolved_reason: PdfHeadingStatus | None = None
    section_interval_id: str | None = None

    @property
    def first_page_number(self) -> int:
        """Return the first physical page supporting this record."""
        return self.page_spans[0].page_number

    @property
    def last_page_number(self) -> int:
        """Return the last physical page supporting this record."""
        return self.page_spans[-1].page_number


def pdf_record_order_key(
    record: PdfIndexRecord,
) -> tuple[str, int, int, int, str]:
    """Return the canonical reading-order key used for every ranking."""
    return (
        record.book_path,
        record.flow_index,
        record.flow_start,
        record.flow_end,
        record.record_id,
    )


def canonical_pdf_record_order(
    records: Sequence[PdfIndexRecord],
) -> tuple[PdfIndexRecord, ...]:
    """Order records independently of how they were supplied."""
    return tuple(sorted(records, key=pdf_record_order_key))


def build_pdf_index_records(
    chunks: Sequence[PdfChunk],
) -> tuple[PdfIndexRecord, ...]:
    """Convert verified chunks without changing text or provenance."""
    _validate_unique_chunk_ids(chunks)
    return tuple(
        PdfIndexRecord(
            record_id=chunk.chunk_id,
            text=chunk.text,
            book_path=chunk.book_path,
            section_path=chunk.section_path,
            flow_index=chunk.flow_index,
            flow_start=chunk.flow_start,
            flow_end=chunk.flow_end,
            page_spans=chunk.page_spans,
            section_resolution=chunk.section_resolution,
            section_hint=chunk.section_hint,
            unresolved_reason=chunk.unresolved_reason,
            section_interval_id=chunk.section_interval_id,
        )
        for chunk in chunks
    )


def _validate_unique_chunk_ids(chunks: Sequence[PdfChunk]) -> None:
    record_ids = [chunk.chunk_id for chunk in chunks]
    if len(record_ids) != len(set(record_ids)):
        message = "PDF index records require unique chunk identifiers"
        raise ValueError(message)
