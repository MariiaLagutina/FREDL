"""Tests for page-level PDF golden evidence candidate mapping."""

import json

import pytest

from src.evaluation.pdf import (
    find_pdf_evidence_candidates,
    PdfGoldenDataset,
)
from src.ingestion.pdf.chunks import PdfChunkPageSpan
from src.ingestion.pdf.index_records import PdfIndexRecord


def _record(
    record_id: str,
    pages: tuple[int, ...],
    *,
    book_path: str = "books/example.pdf",
) -> PdfIndexRecord:
    spans = tuple(
        PdfChunkPageSpan(
            page_number=page,
            chunk_start=index * 10,
            chunk_end=(index + 1) * 10,
            page_source_start=0,
            page_source_end=10,
        )
        for index, page in enumerate(pages)
    )
    return PdfIndexRecord(
        record_id=record_id,
        text="x" * (10 * len(pages)),
        book_path=book_path,
        section_path=("Section",),
        flow_index=0,
        flow_start=0,
        flow_end=10 * len(pages),
        page_spans=spans,
    )


def _dataset(
    pages: list[int],
    *,
    modality: str = "text",
) -> PdfGoldenDataset:
    data = {
        "schema_version": 1,
        "dataset_id": "pdf-golden-v1",
        "book_path": "books/example.pdf",
        "page_numbering": "physical_pdf_page",
        "language": "en",
        "cases": [{
            "id": "pdf-001",
            "question": "Where is the evidence?",
            "why_now": "I need an exact source.",
            "expected_answer": "The evidence is on labelled pages.",
            "answerability": "complete",
            "evidence": [{
                "physical_pages": pages,
                "role": "required",
                "modality": modality,
            }],
            "tags": ["provenance"],
        }],
    }
    return PdfGoldenDataset.model_validate_json(json.dumps(data))


def test_finds_every_record_touching_labelled_pages() -> None:
    records = (
        _record("record-1", (6, 7)),
        _record("record-2", (7, 8)),
        _record("record-3", (9,)),
    )

    result = find_pdf_evidence_candidates(_dataset([7, 8]), records)[0]

    assert result.page_record_ids == ("record-1", "record-2")
    assert result.missing_pages == ()
    assert result.requires_visual_processing is False


def test_reports_labelled_pages_missing_from_index_records() -> None:
    result = find_pdf_evidence_candidates(
        _dataset([7, 8]),
        (_record("record-1", (7,)),),
    )[0]

    assert result.page_record_ids == ("record-1",)
    assert result.missing_pages == (8,)


def test_marks_figure_evidence_for_separate_visual_processing() -> None:
    result = find_pdf_evidence_candidates(
        _dataset([7], modality="figure"),
        (_record("record-1", (7,)),),
    )[0]

    assert result.page_record_ids == ("record-1",)
    assert result.requires_visual_processing is True


def test_ignores_records_from_other_books() -> None:
    records = (
        _record("target", (7,)),
        _record("other", (7,), book_path="books/other.pdf"),
    )

    result = find_pdf_evidence_candidates(_dataset([7]), records)[0]

    assert result.page_record_ids == ("target",)


def test_rejects_records_without_golden_book() -> None:
    records = (_record("other", (7,), book_path="books/other.pdf"),)

    with pytest.raises(ValueError, match="golden book"):
        find_pdf_evidence_candidates(_dataset([7]), records)


def test_rejects_duplicate_record_ids() -> None:
    record = _record("duplicate", (7,))

    with pytest.raises(ValueError, match="unique record IDs"):
        find_pdf_evidence_candidates(_dataset([7]), (record, record))
