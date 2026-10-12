"""Tests for the deterministic PDF BM25 baseline adapter."""

import json

import pytest

from src.evaluation.pdf.bm25_baseline import PdfBm25Baseline
from src.evaluation.pdf.models import PdfGoldenDataset
from src.ingestion.pdf.chunks import PdfChunkPageSpan
from src.ingestion.pdf.index_records import PdfIndexRecord


def _record(
    record_id: str,
    text: str,
    *,
    section_path: tuple[str, ...] = ("Chapter",),
    flow_index: int = 0,
) -> PdfIndexRecord:
    return PdfIndexRecord(
        record_id=record_id,
        text=text,
        book_path="books/example.pdf",
        section_path=section_path,
        flow_index=flow_index,
        flow_start=0,
        flow_end=len(text),
        page_spans=(PdfChunkPageSpan(1, 0, len(text), 0, len(text)),),
    )


def _dataset() -> PdfGoldenDataset:
    return PdfGoldenDataset.model_validate_json(json.dumps({
        "schema_version": 1,
        "dataset_id": "golden-v1",
        "book_path": "books/example.pdf",
        "page_numbering": "physical_pdf_page",
        "language": "en",
        "cases": [
            {
                "id": "match",
                "question": "How should generated code be reviewed?",
                "why_now": "I need an engineering practice.",
                "expected_answer": "A human reviews it.",
                "answerability": "complete",
                "evidence": [{
                    "physical_pages": [1],
                    "role": "required",
                    "modality": "text",
                }],
                "tags": ["positive"],
            },
            {
                "id": "no-match",
                "question": "quantum entanglement",
                "why_now": "I need a safe refusal.",
                "expected_answer": "The source does not say.",
                "answerability": "not_found",
                "evidence": [],
                "tags": ["negative"],
            },
        ],
    }))


def test_ranks_pdf_content_and_preserves_record_identity() -> None:
    baseline = PdfBm25Baseline((
        _record("review", "A human must review generated code."),
        _record("other", "A database stores application state.", flow_index=1),
    ))

    hits = baseline.search("review generated code")

    assert baseline.record_count == 2
    assert hits[0].record.record_id == "review"
    assert hits[0].score > 0
    assert hits[0].content_score > 0


def test_section_titles_contribute_separate_metadata_score() -> None:
    baseline = PdfBm25Baseline((
        _record(
            "workflow",
            "The process has four stages.",
            section_path=("AI-assisted engineering workflow",),
        ),
    ))

    hit = baseline.search("engineering workflow")[0]

    assert hit.content_score == 0
    assert hit.metadata_score > 0


def test_dataset_run_ranks_all_matches_and_abstains_only_at_zero() -> None:
    records = (
        _record("review", "A human must review generated code."),
        _record("other", "A database stores application state.", flow_index=1),
    )
    results = PdfBm25Baseline(records).run_dataset(_dataset())

    assert results[0].ranked_record_ids[0] == "review"
    assert results[0].abstained is False
    assert results[1].ranked_record_ids == ()
    assert results[1].abstained is True


def test_rejects_duplicate_record_ids() -> None:
    records = (
        _record("duplicate", "First record."),
        _record("duplicate", "Second record.", flow_index=1),
    )

    with pytest.raises(ValueError, match="unique record IDs"):
        PdfBm25Baseline(records)
