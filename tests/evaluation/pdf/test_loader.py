"""Tests for strict private PDF golden-dataset loading."""

import json
from pathlib import Path

import pytest

from src.evaluation.pdf import (
    load_pdf_golden_dataset,
    PdfGoldenAnswerability,
    PdfGoldenEvidenceRole,
    PdfGoldenModality,
)


def _case(
    case_id: str = "pdf-001",
    *,
    answerability: str = "complete",
) -> dict[str, object]:
    evidence: list[dict[str, object]] = [
        {
            "physical_pages": [7, 8],
            "role": "required",
            "modality": "text",
        }
    ]
    if answerability == "not_found":
        evidence = []
    return {
        "id": case_id,
        "question": "What keeps the source verifiable?",
        "why_now": "I need to inspect retrieved evidence.",
        "expected_answer": "The record preserves its source pages.",
        "answerability": answerability,
        "evidence": evidence,
        "tags": ["provenance"],
    }


def _dataset(cases: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_id": "private-pdf-golden-v1",
        "book_path": "books/example.pdf",
        "page_numbering": "physical_pdf_page",
        "language": "en",
        "cases": cases,
    }


def _write_json(tmp_path: Path, data: dict[str, object]) -> Path:
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_loads_answerable_and_not_found_cases(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path,
        _dataset([
            _case(),
            _case("pdf-002", answerability="not_found"),
        ]),
    )

    dataset = load_pdf_golden_dataset(path)

    assert dataset.dataset_id == "private-pdf-golden-v1"
    assert dataset.cases[0].answerability is PdfGoldenAnswerability.COMPLETE
    assert dataset.cases[0].evidence[0].role is (
        PdfGoldenEvidenceRole.REQUIRED
    )
    assert dataset.cases[0].evidence[0].modality is PdfGoldenModality.TEXT
    assert dataset.cases[1].evidence == ()


def test_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    path = _write_json(tmp_path, _dataset([_case(), _case()]))

    with pytest.raises(ValueError, match="JSON is invalid"):
        load_pdf_golden_dataset(path)


def test_rejects_evidence_for_not_found_case(tmp_path: Path) -> None:
    case = _case(answerability="not_found")
    case["evidence"] = [{
        "physical_pages": [7],
        "role": "supporting",
        "modality": "text",
    }]
    path = _write_json(tmp_path, _dataset([case]))

    with pytest.raises(ValueError, match="JSON is invalid"):
        load_pdf_golden_dataset(path)


@pytest.mark.parametrize("pages", [[0], [8, 7], [7, 7]])
def test_rejects_invalid_physical_pages(
    tmp_path: Path,
    pages: list[int],
) -> None:
    case = _case()
    case["evidence"] = [{
        "physical_pages": pages,
        "role": "required",
        "modality": "text",
    }]
    path = _write_json(tmp_path, _dataset([case]))

    with pytest.raises(ValueError, match="JSON is invalid"):
        load_pdf_golden_dataset(path)


def test_rejects_unknown_fields(tmp_path: Path) -> None:
    data = _dataset([_case()])
    data["private_book_text"] = "must not be accepted"
    path = _write_json(tmp_path, data)

    with pytest.raises(ValueError, match="JSON is invalid"):
        load_pdf_golden_dataset(path)
