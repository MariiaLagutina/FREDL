"""Persist human PDF record relevance labels for one exact chunk set."""

from collections.abc import Sequence
from enum import Enum
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic import field_validator

from src.evaluation.pdf.models import PdfGoldenDataset
from src.ingestion.pdf.index_records import PdfIndexRecord


class PdfRecordRelevance(str, Enum):
    """Describe one index record's contribution to a golden answer."""

    REQUIRED = "required"
    SUPPORTING = "supporting"
    IRRELEVANT = "irrelevant"


class _StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class PdfRecordJudgment(_StrictModel):
    """Store one human decision for one page-overlap candidate record."""

    case_id: str = Field(min_length=1)
    evidence_index: int = Field(ge=0)
    record_id: str = Field(min_length=1)
    relevance: PdfRecordRelevance


class PdfJudgmentSet(_StrictModel):
    """Bind record-level labels to one dataset and exact record set."""

    schema_version: int
    dataset_id: str = Field(min_length=1)
    book_path: str = Field(min_length=1)
    record_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    judgments: tuple[PdfRecordJudgment, ...] = Field(min_length=1)

    @field_validator("judgments")
    @classmethod
    def require_unique_judgments(
        cls,
        judgments: tuple[PdfRecordJudgment, ...],
    ) -> tuple[PdfRecordJudgment, ...]:
        """Reject multiple decisions for the same candidate record."""
        keys = [
            (item.case_id, item.evidence_index, item.record_id)
            for item in judgments
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("PDF record judgments must be unique")
        return judgments


def fingerprint_pdf_records(records: Sequence[PdfIndexRecord]) -> str:
    """Hash all source and chunk fields that can affect evidence labels."""
    _require_unique_record_ids(records)
    digest = sha256()
    for record in sorted(records, key=lambda item: item.record_id):
        values = (
            record.record_id,
            record.book_path,
            *record.section_path,
            str(record.flow_index),
            str(record.flow_start),
            str(record.flow_end),
            record.text,
        )
        for value in values:
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        for span in record.page_spans:
            for numeric_value in (
                span.page_number,
                span.chunk_start,
                span.chunk_end,
                span.page_source_start,
                span.page_source_end,
            ):
                digest.update(str(numeric_value).encode("ascii"))
                digest.update(b"\0")
    return digest.hexdigest()


def load_pdf_judgment_set(
    path: Path,
    dataset: PdfGoldenDataset,
    records: Sequence[PdfIndexRecord],
) -> PdfJudgmentSet:
    """Load labels only when dataset, book, records, and pages still match."""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError("PDF judgment set could not be read") from error
    try:
        judgment_set = PdfJudgmentSet.model_validate_json(content)
    except ValidationError as error:
        raise ValueError("PDF judgment set JSON is invalid") from error

    _validate_judgment_set(judgment_set, dataset, records)
    return judgment_set


def _validate_judgment_set(
    judgment_set: PdfJudgmentSet,
    dataset: PdfGoldenDataset,
    records: Sequence[PdfIndexRecord],
) -> None:
    if judgment_set.dataset_id != dataset.dataset_id:
        raise ValueError("PDF judgment set belongs to another dataset")
    if judgment_set.book_path != dataset.book_path:
        raise ValueError("PDF judgment set belongs to another book")
    if judgment_set.record_set_fingerprint != fingerprint_pdf_records(records):
        raise ValueError("PDF judgment set record fingerprint differs")

    cases = {case.id: case for case in dataset.cases}
    records_by_id = {record.record_id: record for record in records}
    for judgment in judgment_set.judgments:
        case = cases.get(judgment.case_id)
        if case is None:
            raise ValueError("PDF judgment references an unknown case")
        if judgment.evidence_index >= len(case.evidence):
            raise ValueError("PDF judgment references unknown evidence")
        record = records_by_id.get(judgment.record_id)
        if record is None:
            raise ValueError("PDF judgment references an unknown record")
        evidence_pages = set(
            case.evidence[judgment.evidence_index].physical_pages
        )
        record_pages = {span.page_number for span in record.page_spans}
        if not evidence_pages.intersection(record_pages):
            raise ValueError(
                "PDF judgment record does not overlap evidence pages"
            )


def _require_unique_record_ids(records: Sequence[PdfIndexRecord]) -> None:
    record_ids = [record.record_id for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("PDF record fingerprint requires unique record IDs")
