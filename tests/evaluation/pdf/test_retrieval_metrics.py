"""Tests for PDF retrieval metrics backed by human judgments."""

import json

import pytest

from src.evaluation.pdf.judgments import PdfJudgmentSet
from src.evaluation.pdf.models import PdfGoldenDataset
from src.evaluation.pdf.retrieval_metrics import (
    evaluate_pdf_retrieval,
    PdfRetrievalResult,
)
from src.ingestion.pdf.chunks import PdfChunkPageSpan
from src.ingestion.pdf.index_records import PdfIndexRecord


def _record(record_id: str) -> PdfIndexRecord:
    return PdfIndexRecord(
        record_id=record_id,
        text=record_id,
        book_path="books/example.pdf",
        section_path=("Section",),
        flow_index=0,
        flow_start=0,
        flow_end=len(record_id),
        page_spans=(PdfChunkPageSpan(1, 0, 1, 0, 1),),
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
                "id": "answerable",
                "question": "What is supported?",
                "why_now": "I need evidence.",
                "expected_answer": "Required and supporting evidence.",
                "answerability": "complete",
                "evidence": [{
                    "physical_pages": [1],
                    "role": "required",
                    "modality": "text",
                }],
                "tags": ["positive"],
            },
            {
                "id": "negative",
                "question": "What is absent?",
                "why_now": "I need a safe refusal.",
                "expected_answer": "The source does not say.",
                "answerability": "not_found",
                "evidence": [],
                "tags": ["negative"],
            },
        ],
    }))


def _judgments() -> PdfJudgmentSet:
    return PdfJudgmentSet.model_validate_json(json.dumps({
        "schema_version": 1,
        "dataset_id": "golden-v1",
        "book_path": "books/example.pdf",
        "record_set_fingerprint": "0" * 64,
        "judgments": [
            {
                "case_id": "answerable",
                "evidence_index": 0,
                "record_id": "required",
                "relevance": "required",
            },
            {
                "case_id": "answerable",
                "evidence_index": 0,
                "record_id": "required-alternative",
                "relevance": "required",
            },
            {
                "case_id": "answerable",
                "evidence_index": 0,
                "record_id": "supporting",
                "relevance": "supporting",
            },
            {
                "case_id": "answerable",
                "evidence_index": 0,
                "record_id": "irrelevant",
                "relevance": "irrelevant",
            },
        ],
    }))


def test_scores_ranked_evidence_and_negative_abstention() -> None:
    records = tuple(
        _record(record_id)
        for record_id in (
            "required",
            "required-alternative",
            "supporting",
            "irrelevant",
            "unjudged",
        )
    )
    results = (
        PdfRetrievalResult(
            case_id="answerable",
            ranked_record_ids=("supporting", "required", "unjudged"),
        ),
        PdfRetrievalResult(
            case_id="negative",
            ranked_record_ids=("unjudged",),
            abstained=True,
        ),
    )

    evaluation = evaluate_pdf_retrieval(
        _dataset(), _judgments(), records, results, top_k=3
    )

    answerable = evaluation.case_scores[0]
    assert answerable.required_recall_at_k == 0.5
    assert answerable.all_required_at_k is False
    assert answerable.ranked_required_evidence_coverage_at_k == 1.0
    assert answerable.all_ranked_required_evidence_at_k is True
    assert answerable.context_required_evidence_coverage_at_k == 1.0
    assert answerable.first_required_rank == 2
    assert answerable.reciprocal_rank == 0.5
    assert answerable.judged_irrelevant_at_k == 0
    assert answerable.unjudged_at_k == 1
    assert evaluation.negative_abstention_accuracy == 1.0
    assert evaluation.mean_ranked_required_evidence_coverage_at_k == 1.0
    assert evaluation.all_ranked_required_evidence_at_k_rate == 1.0


def test_does_not_treat_unjudged_records_as_irrelevant() -> None:
    records = (_record("required"), _record("unjudged"))
    results = (
        PdfRetrievalResult("answerable", ("unjudged",)),
        PdfRetrievalResult("negative", (), abstained=True),
    )

    evaluation = evaluate_pdf_retrieval(
        _dataset(), _judgments(), records, results, top_k=1
    )

    score = evaluation.case_scores[0]
    assert score.required_recall_at_k == 0.0
    assert score.judged_irrelevant_at_k == 0
    assert score.unjudged_at_k == 1


def test_reports_context_coverage_separately_from_ranked_coverage() -> None:
    records = (_record("required"), _record("unjudged"))
    results = (
        PdfRetrievalResult(
            "answerable",
            ("unjudged",),
            context_record_ids=("required",),
        ),
        PdfRetrievalResult("negative", (), abstained=True),
    )

    evaluation = evaluate_pdf_retrieval(
        _dataset(), _judgments(), records, results, top_k=1
    )

    score = evaluation.case_scores[0]
    assert score.required_recall_at_k == 0.0
    assert score.first_required_rank is None
    assert score.ranked_required_evidence_coverage_at_k == 0.0
    assert score.all_ranked_required_evidence_at_k is False
    assert score.context_required_evidence_coverage_at_k == 1.0
    assert score.all_context_required_evidence_at_k is True
    assert evaluation.mean_ranked_required_evidence_coverage_at_k == 0.0
    assert evaluation.mean_context_required_evidence_coverage_at_k == 1.0


def test_evidence_counts_only_where_the_record_is_required() -> None:
    dataset = _dataset().model_copy(deep=True)
    case = dataset.cases[0]
    two_units = case.model_copy(update={
        "evidence": (case.evidence[0], case.evidence[0].model_copy(
            update={"physical_pages": (2,)},
        )),
    })
    dataset = dataset.model_copy(
        update={"cases": (two_units, dataset.cases[1])},
    )
    judgments = PdfJudgmentSet.model_validate_json(json.dumps({
        "schema_version": 2,
        "dataset_id": "golden-v1",
        "book_path": "books/example.pdf",
        "record_set_fingerprint": "0" * 64,
        "judgments": [
            {"case_id": "answerable", "evidence_index": 0,
             "record_id": "mixed", "relevance": "required"},
            {"case_id": "answerable", "evidence_index": 1,
             "record_id": "mixed", "relevance": "supporting"},
            {"case_id": "answerable", "evidence_index": 1,
             "record_id": "second", "relevance": "required"},
        ],
    }))
    records = (_record("mixed"), _record("second"))
    results = (
        PdfRetrievalResult("answerable", ("mixed",)),
        PdfRetrievalResult("negative", (), abstained=True),
    )

    score = evaluate_pdf_retrieval(
        dataset, judgments, records, results, top_k=1
    ).case_scores[0]

    assert score.required_evidence_total == 2
    assert score.ranked_required_evidence_covered_at_k == 1
    assert score.ranked_required_evidence_coverage_at_k == 0.5
    assert score.all_ranked_required_evidence_at_k is False
    assert score.required_recall_at_k == 0.5


def test_requires_exactly_one_result_for_every_case() -> None:
    records = (_record("required"),)
    results = (PdfRetrievalResult("answerable", ("required",)),)

    with pytest.raises(ValueError, match="cover every golden case"):
        evaluate_pdf_retrieval(_dataset(), _judgments(), records, results)


def test_rejects_duplicate_and_unknown_ranked_records() -> None:
    records = (_record("required"),)
    negative = PdfRetrievalResult("negative", (), abstained=True)

    with pytest.raises(ValueError, match="must be unique"):
        evaluate_pdf_retrieval(
            _dataset(),
            _judgments(),
            records,
            (
                PdfRetrievalResult("answerable", ("required", "required")),
                negative,
            ),
        )
    with pytest.raises(ValueError, match="unknown record"):
        evaluate_pdf_retrieval(
            _dataset(),
            _judgments(),
            records,
            (PdfRetrievalResult("answerable", ("missing",)), negative),
        )
