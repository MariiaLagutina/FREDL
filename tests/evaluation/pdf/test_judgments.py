"""Tests for versioned human PDF record relevance judgments."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.evaluation.pdf import (
    fingerprint_pdf_records,
    load_pdf_judgment_set,
    PdfGoldenDataset,
    PdfRecordRelevance,
)
from src.evaluation.pdf.judgments import (
    PdfJudgmentMapping,
    PdfJudgmentSource,
)
from src.ingestion.pdf.fingerprints import legacy_fingerprint_pdf_records_v1
from src.ingestion.pdf.chunks import PdfChunkPageSpan
from src.ingestion.pdf.index_records import PdfIndexRecord


def _dataset() -> PdfGoldenDataset:
    data = {
        "schema_version": 1,
        "dataset_id": "golden-v1",
        "book_path": "books/example.pdf",
        "page_numbering": "physical_pdf_page",
        "language": "en",
        "cases": [{
            "id": "pdf-001",
            "question": "Where is the evidence?",
            "why_now": "I need a trusted answer.",
            "expected_answer": "It is on page seven.",
            "answerability": "complete",
            "evidence": [{
                "physical_pages": [7],
                "role": "required",
                "modality": "text",
            }],
            "tags": ["evidence"],
        }],
    }
    return PdfGoldenDataset.model_validate_json(json.dumps(data))


def _record(
    record_id: str = "pdf:record-1",
    *,
    page: int = 7,
    text: str = "Evidence.",
) -> PdfIndexRecord:
    return PdfIndexRecord(
        record_id=record_id,
        text=text,
        book_path="books/example.pdf",
        section_path=("Section",),
        flow_index=0,
        flow_start=0,
        flow_end=len(text),
        page_spans=(
            PdfChunkPageSpan(page, 0, len(text), 0, len(text)),
        ),
    )


def _write_judgments(
    tmp_path: Path,
    records: tuple[PdfIndexRecord, ...],
    **overrides: object,
) -> Path:
    data: dict[str, object] = {
        "schema_version": 2,
        "dataset_id": "golden-v1",
        "book_path": "books/example.pdf",
        "record_set_fingerprint": fingerprint_pdf_records(records),
        "judgments": [{
            "case_id": "pdf-001",
            "evidence_index": 0,
            "record_id": records[0].record_id,
            "relevance": "required",
        }],
    }
    data.update(overrides)
    path = tmp_path / "judgments.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_loads_version_1_judgments_with_the_legacy_fingerprint(
    tmp_path: Path,
) -> None:
    records = (_record(),)
    path = _write_judgments(
        tmp_path, records,
        schema_version=1,
        record_set_fingerprint=legacy_fingerprint_pdf_records_v1(records),
    )

    result = load_pdf_judgment_set(path, _dataset(), records)

    assert result.schema_version == 1
    with pytest.raises(ValueError, match="fingerprint differs"):
        load_pdf_judgment_set(
            _write_judgments(tmp_path, records, schema_version=1),
            _dataset(),
            records,
        )


def test_version_1_judgments_cannot_carry_migration_provenance(
    tmp_path: Path,
) -> None:
    records = (_record(),)
    path = _write_judgments(
        tmp_path, records,
        schema_version=1,
        record_set_fingerprint=legacy_fingerprint_pdf_records_v1(records),
        migrated_from=_source(),
    )

    with pytest.raises(ValueError, match="JSON is invalid"):
        load_pdf_judgment_set(path, _dataset(), records)


def test_version_2_judgments_keep_migration_provenance(
    tmp_path: Path,
) -> None:
    records = (_record(),)
    path = _write_judgments(
        tmp_path, records,
        migrated_from=_source(),
        judgments=[{
            "case_id": "pdf-001",
            "evidence_index": 0,
            "record_id": records[0].record_id,
            "relevance": "required",
            "provenance": {
                "source_record_id": "pdf:old-record",
                "mapping": "whitespace_only",
            },
        }],
    )

    result = load_pdf_judgment_set(path, _dataset(), records)

    provenance = result.judgments[0].provenance
    assert provenance is not None
    assert provenance.mapping is PdfJudgmentMapping.WHITESPACE_ONLY
    assert result.migrated_from == PdfJudgmentSource.model_validate(_source())


def _source() -> dict[str, object]:
    return {
        "dataset_id": "golden-v0",
        "judgments_sha256": "a" * 64,
        "record_set_fingerprint": "b" * 64,
        "record_set_fingerprint_version": 1,
    }


def test_loads_judgments_for_exact_record_set(tmp_path: Path) -> None:
    records = (_record(),)
    path = _write_judgments(tmp_path, records)

    result = load_pdf_judgment_set(path, _dataset(), records)

    assert result.judgments[0].relevance is PdfRecordRelevance.REQUIRED


def test_record_fingerprint_is_stable_for_record_order() -> None:
    first = _record("pdf:first")
    second = _record("pdf:second", text="More evidence.")

    assert fingerprint_pdf_records((first, second)) == (
        fingerprint_pdf_records((second, first))
    )


def test_rejects_labels_after_section_interval_changes(
    tmp_path: Path,
) -> None:
    records = (_record(),)
    path = _write_judgments(tmp_path, records)
    regrouped = (replace(records[0], section_interval_id="pdf-section:b"),)

    with pytest.raises(ValueError, match="fingerprint differs"):
        load_pdf_judgment_set(path, _dataset(), regrouped)


def test_rejects_labels_after_record_content_changes(tmp_path: Path) -> None:
    records = (_record(),)
    path = _write_judgments(tmp_path, records)
    changed = (_record(text="Changed evidence."),)

    with pytest.raises(ValueError, match="fingerprint differs"):
        load_pdf_judgment_set(path, _dataset(), changed)


def test_rejects_record_outside_evidence_pages(tmp_path: Path) -> None:
    records = (_record(page=8),)
    path = _write_judgments(tmp_path, records)

    with pytest.raises(ValueError, match="does not overlap"):
        load_pdf_judgment_set(path, _dataset(), records)


def test_rejects_duplicate_judgments(tmp_path: Path) -> None:
    records = (_record(),)
    judgment = {
        "case_id": "pdf-001",
        "evidence_index": 0,
        "record_id": records[0].record_id,
        "relevance": "required",
    }
    path = _write_judgments(
        tmp_path,
        records,
        judgments=[judgment, judgment],
    )

    with pytest.raises(ValueError, match="JSON is invalid"):
        load_pdf_judgment_set(path, _dataset(), records)
