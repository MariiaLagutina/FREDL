"""Tests for immutable retrieved-source validation reports."""

import pytest

from src.retrieval.validation import (
    SourceValidationIssue,
    SourceValidationIssueKind,
    SourceValidationReport,
)


def make_issue(kind: SourceValidationIssueKind) -> SourceValidationIssue:
    """Create one representative issue for the first retrieved source."""
    return SourceValidationIssue(
        kind=kind,
        result_index=0,
        source_index=0,
        question_id="q-1",
        file_path="data/raw/vllm/example.py",
        detail="Representative validation failure",
    )


def test_successful_report_exposes_valid_source_count() -> None:
    """A report without issues marks every checked source as valid."""
    report = SourceValidationReport(result_count=2, source_count=5)

    assert report.valid_source_count == 5
    assert report.invalid_source_count == 0
    assert report.passed


def test_multiple_issues_count_one_invalid_source_once() -> None:
    """Several failures on one source do not inflate invalid-source count."""
    report = SourceValidationReport(
        result_count=1,
        source_count=2,
        issues=(
            make_issue(SourceValidationIssueKind.INVALID_RANGE),
            make_issue(SourceValidationIssueKind.OVERSIZED),
        ),
    )

    assert report.invalid_source_count == 1
    assert report.valid_source_count == 1
    assert not report.passed


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("result_index", -1, "result index"),
        ("source_index", -1, "source index"),
        ("question_id", "", "question ID"),
        ("file_path", "", "file path"),
        ("detail", "", "detail"),
    ],
)
def test_issue_rejects_missing_source_context(
    field: str,
    value: int | str,
    message: str,
) -> None:
    """Every issue must identify an actionable retrieved source."""
    values: dict[str, object] = {
        "kind": SourceValidationIssueKind.UNKNOWN_PATH,
        "result_index": 0,
        "source_index": 0,
        "question_id": "q-1",
        "file_path": "data/raw/vllm/example.py",
        "detail": "Representative validation failure",
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        SourceValidationIssue(**values)  # type: ignore[arg-type]


def test_report_rejects_negative_result_count() -> None:
    """A report cannot describe a negative number of retrieval results."""
    with pytest.raises(ValueError, match="count must not be negative"):
        SourceValidationReport(result_count=-1, source_count=0)


def test_report_rejects_negative_source_count() -> None:
    """A report cannot describe a negative number of retrieved sources."""
    with pytest.raises(ValueError, match="count must not be negative"):
        SourceValidationReport(result_count=0, source_count=-1)
