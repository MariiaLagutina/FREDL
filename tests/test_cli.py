"""Tests for the assignment-compatible Python Fire CLI."""

import json
from pathlib import Path
from unittest.mock import ANY, patch

import pytest

from src.__main__ import main
from src.retrieval.validation import SourceValidationReport


FINGERPRINT = "a" * 64
PIPELINE_FINGERPRINT = "b" * 64


def test_index_command_builds_schema_v2_snapshot(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public command runs production ingestion and persists its result."""
    corpus_root = tmp_path / "data" / "raw"
    corpus_root.mkdir(parents=True)
    (corpus_root / "guide.md").write_text(
        "# Cache\n\nThe cache stores chunks.\n",
        encoding="utf-8",
    )

    main(
        [
            "index",
            "--project_root",
            str(tmp_path),
            "--corpus_root",
            "data/raw",
            "--index_path",
            "data/processed/test-index.json",
        ]
    )

    index_path = tmp_path / "data" / "processed" / "test-index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert len(payload["corpus_fingerprint"]) == 64
    assert len(payload["pipeline_fingerprint"]) == 64
    captured = capsys.readouterr()
    assert "document_count:" in captured.out
    assert "schema_version:       2" in captured.out


def test_search_command_routes_one_raw_query() -> None:
    """The Fire search command reaches the stored single-query workflow."""
    with (
        patch(
            "src.cli._current_corpus_fingerprint",
            return_value=FINGERPRINT,
        ),
        patch(
            "src.cli._current_pipeline_fingerprint",
            return_value=PIPELINE_FINGERPRINT,
        ),
        patch("src.cli.run_stored_search", return_value=[]) as run_search,
    ):
        main(["search", "Where is the cache?", "--k", "3"])

    run_search.assert_called_once_with(
        Path("data/processed/bm25-index.json"),
        FINGERPRINT,
        PIPELINE_FINGERPRINT,
        "Where is the cache?",
        3,
    )


def test_search_dataset_uses_assignment_paths_and_output_name() -> None:
    """The Fire batch command preserves the dataset filename in its output."""
    with (
        patch(
            "src.cli._current_corpus_fingerprint",
            return_value=FINGERPRINT,
        ),
        patch(
            "src.cli._current_pipeline_fingerprint",
            return_value=PIPELINE_FINGERPRINT,
        ),
        patch("src.cli.run_stored_retrieval") as run_retrieval,
    ):
        main(
            [
                "search_dataset",
                "--dataset_path",
                "data/datasets/questions.json",
                "--save_directory",
                "data/output/search_results/Public",
                "--k",
                "3",
            ]
        )

    run_retrieval.assert_called_once_with(
        Path("data/processed/bm25-index.json"),
        FINGERPRINT,
        PIPELINE_FINGERPRINT,
        Path("data/datasets/questions.json"),
        Path("data/output/search_results/Public/questions.json"),
        3,
        progress=ANY,
    )


def test_search_rejects_non_positive_k_before_corpus_scan(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An invalid limit fails before fingerprint or index work begins."""
    with patch("src.cli._current_corpus_fingerprint") as fingerprint:
        with pytest.raises(SystemExit) as exit_info:
            main(["search", "cache", "--k", "0"])

    fingerprint.assert_not_called()
    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.err == "Error: Search k must be greater than zero\n"
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (FileNotFoundError("questions.json"), "File not found"),
        (ValueError("Question dataset JSON is invalid"), "JSON is invalid"),
        (
            ValueError(
                "Stored BM25 index schema is incompatible; reindex required"
            ),
            "reindex required",
        ),
    ],
)
def test_search_dataset_reports_expected_failures_without_traceback(
    failure: Exception,
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expected file and validation failures remain concise for users."""
    with (
        patch(
            "src.cli._current_corpus_fingerprint",
            return_value=FINGERPRINT,
        ),
        patch("src.cli.run_stored_retrieval", side_effect=failure),
    ):
        with pytest.raises(SystemExit) as exit_info:
            main(
                [
                    "search_dataset",
                    "--dataset_path",
                    "questions.json",
                    "--save_directory",
                    "results",
                ]
            )

    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert message in captured.err
    assert "Traceback" not in captured.err


def test_validate_sources_command_returns_audit_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The Fire validation command exposes stable report counts."""
    report = SourceValidationReport(result_count=100, source_count=500)
    with patch(
        "src.cli.validate_retrieval_file",
        return_value=report,
    ) as validate_file:
        main(
            [
                "validate_sources",
                "--results_path",
                "results/docs.json",
            ]
        )

    validate_file.assert_called_once_with(
        Path("results/docs.json"),
        Path("."),
        Path("data/raw"),
        max_source_length=2000,
    )
    captured = capsys.readouterr()
    assert "result_count:         100" in captured.out
    assert "source_count:         500" in captured.out
    assert "invalid_source_count: 0" in captured.out
    assert "passed:               true" in captured.out
