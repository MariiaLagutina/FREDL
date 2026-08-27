"""Tests for complete retrieval-results source validation."""

from src.models import MinimalSource, QuerySearchResult, RetrievalResults
from src.retrieval.validation import (
    SourceValidationIssueKind,
    validate_retrieval_results,
)


KNOWN_PATH = "data/raw/vllm/known.py"


def source(path: str, start: int, end: int) -> MinimalSource:
    """Create one retrieved source for runner tests."""
    return MinimalSource(
        file_path=path,
        first_character_index=start,
        last_character_index=end,
    )


def result(
    question_id: str,
    sources: list[MinimalSource],
) -> QuerySearchResult:
    """Create one query result with representative question text."""
    return QuerySearchResult(
        question_id=question_id,
        question="Where is the implementation?",
        retrieved_sources=sources,
    )


def test_runner_reports_every_source_as_valid() -> None:
    """A complete valid batch produces a successful count summary."""
    results = RetrievalResults(
        search_results=[
            result("q-1", [source(KNOWN_PATH, 0, 5)]),
            result("q-2", [source(KNOWN_PATH, 6, 10)]),
        ],
        k=1,
    )

    report = validate_retrieval_results(
        results,
        {KNOWN_PATH: "cache data"},
    )

    assert report.result_count == 2
    assert report.source_count == 2
    assert report.valid_source_count == 2
    assert report.passed


def test_runner_accumulates_failures_in_stable_result_order() -> None:
    """Later sources are checked after earlier failures are discovered."""
    results = RetrievalResults(
        search_results=[
            result("q-1", [source("missing.py", 0, 5)]),
            result(
                "q-2",
                [
                    source(KNOWN_PATH, 5, 5),
                    source("also-missing.py", -1, 0),
                ],
            ),
        ],
        k=2,
    )

    report = validate_retrieval_results(
        results,
        {KNOWN_PATH: "cache data"},
    )

    assert report.result_count == 2
    assert report.source_count == 3
    assert report.invalid_source_count == 3
    assert [issue.kind for issue in report.issues] == [
        SourceValidationIssueKind.UNKNOWN_PATH,
        SourceValidationIssueKind.INVALID_RANGE,
        SourceValidationIssueKind.UNKNOWN_PATH,
        SourceValidationIssueKind.INVALID_RANGE,
    ]
    assert [issue.result_index for issue in report.issues] == [0, 1, 1, 1]
    assert [issue.source_index for issue in report.issues] == [0, 0, 1, 1]


def test_runner_accepts_an_empty_result_batch() -> None:
    """An empty retrieval file has a valid zero-count report."""
    report = validate_retrieval_results(
        RetrievalResults(search_results=[], k=5),
        {},
    )

    assert report.result_count == 0
    assert report.source_count == 0
    assert report.passed
