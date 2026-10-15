"""Write and verify private JSON snapshots of PDF index record sets."""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.ingestion.pdf.chunks import PdfChunkPageSpan
from src.ingestion.pdf.fingerprints import (
    fingerprint_pdf_records,
    legacy_fingerprint_pdf_records_v1,
)
from src.ingestion.pdf.headings import PdfHeadingStatus
from src.ingestion.pdf.index_records import (
    canonical_pdf_record_order,
    PdfIndexRecord,
)
from src.ingestion.pdf.sections import PdfSectionResolution


class _StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _SnapshotRecord(_StrictModel):
    record_id: str = Field(min_length=1)
    text: str
    book_path: str = Field(min_length=1)
    section_path: tuple[str, ...]
    flow_index: int = Field(ge=0)
    flow_start: int = Field(ge=0)
    flow_end: int = Field(ge=0)
    page_spans: tuple[tuple[int, int, int, int, int], ...]
    section_interval_id: str | None = None
    section_resolution: PdfSectionResolution = PdfSectionResolution.NONE
    section_hint: tuple[str, ...] = ()
    unresolved_reason: PdfHeadingStatus | None = None


class PdfRecordSnapshot(_StrictModel):
    """Preserve one exact record set with both fingerprint versions."""

    schema_name: Literal["fredl.pdf.record-snapshot"] = Field(
        default="fredl.pdf.record-snapshot", alias="schema",
    )
    version: Literal[1] = 1
    book_path: str = Field(min_length=1)
    code_revision: str = Field(min_length=1)
    record_set_fingerprint_v1: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_set_fingerprint_v2: str = Field(pattern=r"^[0-9a-f]{64}$")
    records: tuple[_SnapshotRecord, ...]

    model_config = ConfigDict(
        frozen=True, extra="forbid", populate_by_name=True,
    )


def snapshot_pdf_records(
    records: Sequence[PdfIndexRecord],
    *,
    code_revision: str,
) -> PdfRecordSnapshot:
    """Capture records in canonical order with their fingerprints."""
    book_paths = {record.book_path for record in records}
    if len(book_paths) != 1:
        raise ValueError("PDF record snapshot requires records of one book")
    return PdfRecordSnapshot(
        book_path=next(iter(book_paths)),
        code_revision=code_revision,
        record_set_fingerprint_v1=legacy_fingerprint_pdf_records_v1(records),
        record_set_fingerprint_v2=fingerprint_pdf_records(records),
        records=tuple(
            _SnapshotRecord(
                record_id=record.record_id,
                text=record.text,
                book_path=record.book_path,
                section_path=record.section_path,
                flow_index=record.flow_index,
                flow_start=record.flow_start,
                flow_end=record.flow_end,
                page_spans=tuple(
                    (
                        span.page_number,
                        span.chunk_start,
                        span.chunk_end,
                        span.page_source_start,
                        span.page_source_end,
                    )
                    for span in record.page_spans
                ),
                section_interval_id=record.section_interval_id,
                section_resolution=record.section_resolution,
                section_hint=record.section_hint,
                unresolved_reason=record.unresolved_reason,
            )
            for record in canonical_pdf_record_order(records)
        ),
    )


def write_pdf_record_snapshot(path: Path, snapshot: PdfRecordSnapshot) -> None:
    """Write a snapshot without overwriting an existing file."""
    content = snapshot.model_dump_json(by_alias=True, indent=1)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content + "\n")
    except FileExistsError as error:
        raise ValueError("PDF record snapshot already exists") from error
    except OSError as error:
        raise ValueError("PDF record snapshot could not be written") from error


def load_pdf_record_snapshot(
    path: Path,
) -> tuple[PdfRecordSnapshot, tuple[PdfIndexRecord, ...]]:
    """Load a snapshot only when both stored fingerprints still match."""
    try:
        snapshot = PdfRecordSnapshot.model_validate_json(
            path.read_text(encoding="utf-8"),
        )
    except OSError as error:
        raise ValueError("PDF record snapshot could not be read") from error
    except ValidationError as error:
        raise ValueError("PDF record snapshot JSON is invalid") from error

    records = tuple(
        PdfIndexRecord(
            record_id=item.record_id,
            text=item.text,
            book_path=item.book_path,
            section_path=item.section_path,
            flow_index=item.flow_index,
            flow_start=item.flow_start,
            flow_end=item.flow_end,
            page_spans=tuple(
                PdfChunkPageSpan(*span) for span in item.page_spans
            ),
            section_resolution=item.section_resolution,
            section_hint=item.section_hint,
            unresolved_reason=item.unresolved_reason,
            section_interval_id=item.section_interval_id,
        )
        for item in snapshot.records
    )
    if legacy_fingerprint_pdf_records_v1(records) != (
        snapshot.record_set_fingerprint_v1
    ) or fingerprint_pdf_records(records) != (
        snapshot.record_set_fingerprint_v2
    ):
        raise ValueError("PDF record snapshot fingerprint differs")
    return snapshot, records
