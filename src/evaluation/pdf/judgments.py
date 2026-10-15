"""Persist human PDF record relevance labels for one exact chunk set."""

from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic import field_validator, model_validator

from src.evaluation.pdf.models import PdfGoldenDataset
from src.ingestion.pdf.fingerprints import (
    fingerprint_pdf_records,
    legacy_fingerprint_pdf_records_v1,
)
from src.ingestion.pdf.index_records import PdfIndexRecord


class PdfRecordRelevance(str, Enum):
    """Describe one index record's contribution to a golden answer."""

    REQUIRED = "required"
    SUPPORTING = "supporting"
    IRRELEVANT = "irrelevant"


class _StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")


class PdfJudgmentMapping(str, Enum):
    """Describe how a migrated label reached its current record."""

    BYTE_IDENTICAL = "byte_identical"
    WHITESPACE_ONLY = "whitespace_only"
    MANUAL_REVIEW = "manual_review"


class PdfJudgmentProvenance(_StrictModel):
    """Trace one migrated label to the record a human originally judged."""

    source_record_id: str = Field(min_length=1)
    mapping: PdfJudgmentMapping


class PdfJudgmentSource(_StrictModel):
    """Identify the preserved judgment file a set was migrated from."""

    dataset_id: str = Field(min_length=1)
    judgments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    record_set_fingerprint_version: Literal[1, 2]


class PdfRecordJudgment(_StrictModel):
    """Store one human decision for one page-overlap candidate record."""

    case_id: str = Field(min_length=1)
    evidence_index: int = Field(ge=0)
    record_id: str = Field(min_length=1)
    relevance: PdfRecordRelevance
    provenance: PdfJudgmentProvenance | None = None


class PdfJudgmentSet(_StrictModel):
    """Bind record-level labels to one dataset and exact record set.

    Version 1 sets use the legacy record fingerprint. Version 2 sets use
    the canonical fingerprint and may record their migration source.
    """

    schema_version: Literal[1, 2]
    dataset_id: str = Field(min_length=1)
    book_path: str = Field(min_length=1)
    record_set_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    judgments: tuple[PdfRecordJudgment, ...] = Field(min_length=1)
    migrated_from: PdfJudgmentSource | None = None

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

    @model_validator(mode="after")
    def require_versioned_provenance(self) -> "PdfJudgmentSet":
        """Allow migration provenance only in version 2 sets."""
        has_provenance = self.migrated_from is not None or any(
            item.provenance is not None for item in self.judgments
        )
        if self.schema_version == 1 and has_provenance:
            raise ValueError("Version 1 PDF judgments cannot carry provenance")
        return self


def fingerprint_for_judgment_version(
    schema_version: int,
    records: Sequence[PdfIndexRecord],
) -> str:
    """Return the record fingerprint a judgment schema version binds to."""
    if schema_version == 1:
        return legacy_fingerprint_pdf_records_v1(records)
    if schema_version == 2:
        return fingerprint_pdf_records(records)
    raise ValueError("Unsupported PDF judgment schema version")


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

    validate_pdf_judgment_set(judgment_set, dataset, records)
    return judgment_set


def validate_pdf_judgment_set(
    judgment_set: PdfJudgmentSet,
    dataset: PdfGoldenDataset,
    records: Sequence[PdfIndexRecord],
) -> None:
    """Require labels to match the dataset, records, and evidence pages."""
    if judgment_set.dataset_id != dataset.dataset_id:
        raise ValueError("PDF judgment set belongs to another dataset")
    if judgment_set.book_path != dataset.book_path:
        raise ValueError("PDF judgment set belongs to another book")
    expected = fingerprint_for_judgment_version(
        judgment_set.schema_version, records,
    )
    if judgment_set.record_set_fingerprint != expected:
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
