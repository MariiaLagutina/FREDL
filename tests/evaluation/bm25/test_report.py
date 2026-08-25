"""Tests for BM25 suite fingerprints and machine-readable reports."""

import json
from pathlib import Path

import pytest

from src.evaluation.bm25.models import ExperimentResult
from src.evaluation.bm25.report import write_json_report
from src.evaluation.bm25.reporter import print_results
from src.evaluation.bm25.runner import run_experiment
from src.evaluation.bm25.suite import (
    build_suite_documents,
    fingerprint_suite,
    load_suite,
)
from src.retrieval.bm25 import BM25Parameters


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUITE_ROOT = PROJECT_ROOT / "evals" / "bm25" / "mini"


def _result() -> ExperimentResult:
    """Run the small fixed control for report tests."""
    suite = load_suite(SUITE_ROOT)
    documents = build_suite_documents(PROJECT_ROOT, SUITE_ROOT, suite)
    return run_experiment(
        suite,
        documents,
        "M0",
        BM25Parameters(metadata_weight=1.0),
        latency_repetitions=1,
    )


def test_suite_fingerprint_is_stable_and_complete() -> None:
    """The fixed suite has one repeatable SHA-256 identity."""
    first = fingerprint_suite(SUITE_ROOT)
    second = fingerprint_suite(SUITE_ROOT)

    assert first == second
    assert len(first) == 64


def test_json_report_keeps_reproducibility_and_ranking_evidence(
    tmp_path: Path,
) -> None:
    """A saved report contains context, memory, metrics, and ranked hits."""
    output_path = tmp_path / "reports" / "m0.json"

    write_json_report(output_path, SUITE_ROOT, [_result()])
    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["schema_version"] == 1
    assert report["suite_fingerprint"] == fingerprint_suite(SUITE_ROOT)
    assert report["environment"]["python"]
    assert isinstance(report["git_dirty"], bool)
    assert report["runs"][0]["index_size_bytes"] > 0
    assert report["runs"][0]["peak_build_memory_bytes"] > 0
    assert report["runs"][0]["queries"][0]["hits"][0]["rank"] == 1


def test_terminal_report_displays_corpus_and_memory_counts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The concise terminal view exposes size evidence without verbose mode."""
    print_results([_result()])

    output = capsys.readouterr().out
    assert "files=8; chunks=9" in output
    assert "IndexKB" in output
    assert "PeakKB" in output
