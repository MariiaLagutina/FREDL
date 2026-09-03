"""Tests for the assignment-compatible Python Fire CLI."""

import json
from pathlib import Path
from unittest.mock import ANY, call, patch

import pytest

from src.__main__ import main
from src.evaluation.retrieval import (
    RetrievalDatasetKind,
    RetrievalEvaluationReport,
    RetrievalMetrics,
)
from src.retrieval.bm25 import BM25Parameters
from src.retrieval.index_store import (
    PipelineConfig,
    SCHEMA_VERSION,
    fingerprint_pipeline,
)
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


def test_index_command_persists_requested_bm25_parameters(
    tmp_path: Path,
) -> None:
    """A controlled experiment records its BM25 parameters in the index."""
    corpus_root = tmp_path / "data" / "raw"
    corpus_root.mkdir(parents=True)
    (corpus_root / "guide.md").write_text("# Cache\n", encoding="utf-8")

    main(
        [
            "index",
            "--project_root",
            str(tmp_path),
            "--metadata_weight",
            "1.5",
        ]
    )

    payload = json.loads(
        (tmp_path / "data" / "processed" / "bm25-index.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["parameters"] == {
        "b": 0.65,
        "k1": 1.4,
        "metadata_weight": 1.5,
    }


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


def test_search_uses_requested_bm25_pipeline_fingerprint() -> None:
    """Search rejects accidental reuse of an index from another experiment."""
    parameters = BM25Parameters(metadata_weight=1.5)
    expected = fingerprint_pipeline(
        PipelineConfig(parameters=parameters),
        index_schema_version=SCHEMA_VERSION,
    )
    with (
        patch(
            "src.cli._current_corpus_fingerprint",
            return_value=FINGERPRINT,
        ),
        patch("src.cli.run_stored_search", return_value=[]) as run_search,
    ):
        main(["search", "cache", "--metadata_weight", "1.5"])

    assert run_search.call_args.args[2] == expected


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


def test_evaluate_command_reports_docs_and_code_separately(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The public evaluator loads and labels both required datasets."""
    metrics = RetrievalMetrics(2, 0.25, 1.0, 1.0, 1.0, 0.75)
    with (
        patch(
            "src.cli.load_evaluation_cases",
            side_effect=[(), ()],
        ) as load_cases,
        patch(
            "src.cli.evaluate_cases",
            side_effect=[
                RetrievalEvaluationReport(
                    RetrievalDatasetKind.DOCS,
                    metrics,
                ),
                RetrievalEvaluationReport(
                    RetrievalDatasetKind.CODE,
                    metrics,
                ),
            ],
        ) as evaluate_loaded_cases,
    ):
        main(
            [
                "evaluate",
                "--docs_ground_truth_path",
                "datasets/docs.json",
                "--docs_results_path",
                "results/docs.json",
                "--code_ground_truth_path",
                "datasets/code.json",
                "--code_results_path",
                "results/code.json",
                "--project_root",
                "/project",
            ]
        )

    assert load_cases.call_args_list == [
        call(
            Path("/project/datasets/docs.json"),
            Path("/project/results/docs.json"),
        ),
        call(
            Path("/project/datasets/code.json"),
            Path("/project/results/code.json"),
        ),
    ]
    assert evaluate_loaded_cases.call_args_list == [
        call(RetrievalDatasetKind.DOCS, ()),
        call(RetrievalDatasetKind.CODE, ()),
    ]
    output = capsys.readouterr().out
    assert "Docs:" in output
    assert "Code:" in output
    assert output.count("query_count:  2") == 2
    assert output.count("recall_at_1:  0.250000") == 2
    assert output.count("mrr:          0.750000") == 2


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (FileNotFoundError("docs.json"), "File not found"),
        (ValueError("Retrieval results JSON is invalid"), "JSON is invalid"),
        (
            ValueError(
                "Ground truth and retrieval results must contain the same IDs"
            ),
            "same IDs",
        ),
    ],
)
def test_evaluate_command_reports_expected_failures_without_traceback(
    failure: Exception,
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expected evaluator input failures remain concise for users."""
    with patch(
        "src.cli.load_evaluation_cases",
        side_effect=failure,
    ):
        with pytest.raises(SystemExit) as exit_info:
            main(
                [
                    "evaluate",
                    "--docs_ground_truth_path",
                    "docs-ground-truth.json",
                    "--docs_results_path",
                    "docs-results.json",
                    "--code_ground_truth_path",
                    "code-ground-truth.json",
                    "--code_results_path",
                    "code-results.json",
                ]
            )

    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert message in captured.err
    assert "Traceback" not in captured.err


def test_analyze_retrieval_errors_writes_docs_and_code_report(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The CLI exposes the generated report path and miss counts."""
    docs_cases = ("docs-case",)
    code_cases = ("code-case",)
    with (
        patch(
            "src.cli.load_evaluation_cases",
            side_effect=[docs_cases, code_cases],
        ),
        patch("src.cli.write_error_analysis_markdown") as write_report,
        patch(
            "src.cli.collect_top_five_misses",
            side_effect=[("docs-miss",), ("code-1", "code-2")],
        ),
    ):
        main(
            [
                "analyze_retrieval_errors",
                "--docs_ground_truth_path",
                "datasets/docs.json",
                "--docs_results_path",
                "results/docs.json",
                "--code_ground_truth_path",
                "datasets/code.json",
                "--code_results_path",
                "results/code.json",
                "--output_path",
                "reports/errors.md",
                "--project_root",
                "/project",
            ]
        )

    write_report.assert_called_once_with(
        Path("/project/reports/errors.md"),
        (
            (RetrievalDatasetKind.DOCS, docs_cases),
            (RetrievalDatasetKind.CODE, code_cases),
        ),
        Path("/project"),
        {},
    )
    output = capsys.readouterr().out
    assert "docs_top_5_misses: 1" in output
    assert "code_top_5_misses: 2" in output
    assert "/project/reports/errors.md" in output
