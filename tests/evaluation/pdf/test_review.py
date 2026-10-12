"""Tests for private PDF evidence review reports."""

import json
from pathlib import Path

from src.evaluation.pdf import (
    PdfGoldenDataset,
    render_pdf_evidence_review,
    write_pdf_evidence_review,
)
from src.ingestion.pdf.chunks import PdfChunkPageSpan
from src.ingestion.pdf.index_records import PdfIndexRecord


def _dataset() -> PdfGoldenDataset:
    data = {
        "schema_version": 1,
        "dataset_id": "review-v1",
        "book_path": "books/example.pdf",
        "page_numbering": "physical_pdf_page",
        "language": "en",
        "cases": [
            {
                "id": "pdf-001",
                "question": "What does the diagram show?",
                "why_now": "I need visual evidence.",
                "expected_answer": "The diagram shows a verified workflow.",
                "answerability": "complete",
                "evidence": [
                    {
                        "physical_pages": [7],
                        "role": "required",
                        "modality": "figure",
                    }
                ],
                "tags": ["visual"],
            },
            {
                "id": "pdf-002",
                "question": "What is absent?",
                "why_now": "I need a negative control.",
                "expected_answer": "The book does not contain this answer.",
                "answerability": "not_found",
                "evidence": [],
                "tags": ["negative"],
            },
        ],
    }
    return PdfGoldenDataset.model_validate_json(json.dumps(data))


def _record() -> PdfIndexRecord:
    return PdfIndexRecord(
        record_id="pdf:record-1",
        text="A private example passage.",
        book_path="books/example.pdf",
        section_path=("Chapter", "Workflow"),
        flow_index=0,
        flow_start=10,
        flow_end=36,
        page_spans=(
            PdfChunkPageSpan(
                page_number=7,
                chunk_start=0,
                chunk_end=26,
                page_source_start=5,
                page_source_end=31,
            ),
        ),
    )


def test_renders_candidates_without_preselecting_relevance() -> None:
    report = render_pdf_evidence_review(_dataset(), (_record(),))

    assert "Page overlap creates candidates" in report
    assert "#### pdf:record-1" in report
    assert "[ ] required  [ ] supporting  [ ] irrelevant" in report
    assert "Chapter > Workflow" in report
    assert "A private example passage." in report


def test_renders_visual_and_not_found_guidance() -> None:
    report = render_pdf_evidence_review(_dataset(), (_record(),))

    assert "Visual review required" in report
    assert "No relevant record is expected" in report


def test_writes_review_to_explicit_local_path(tmp_path: Path) -> None:
    path = tmp_path / "review.md"

    write_pdf_evidence_review(path, _dataset(), (_record(),))

    assert path.read_text(encoding="utf-8").startswith(
        "# PDF evidence review: review-v1"
    )
