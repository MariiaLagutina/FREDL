"""Tests for file-based retrieved-source validation."""

from pathlib import Path

import pytest

from src.retrieval.validation import (
    SourceValidationIssueKind,
    validate_retrieval_file,
)


def test_validate_retrieval_file_checks_complete_local_boundaries(
    tmp_path: Path,
) -> None:
    """One workflow loads result JSON and exact corpus text for validation."""
    corpus_root = tmp_path / "data" / "raw"
    corpus_root.mkdir(parents=True)
    source_path = corpus_root / "cache.py"
    source_path.write_text("cache = build_cache()\n", encoding="utf-8")
    results_path = tmp_path / "results.json"
    results_path.write_text(
        """
        {
          "search_results": [
            {
              "question_id": "q-1",
              "question": "Where is cache?",
              "retrieved_sources": [
                {
                  "file_path": "data/raw/cache.py",
                  "first_character_index": 0,
                  "last_character_index": 5
                },
                {
                  "file_path": "data/raw/missing.py",
                  "first_character_index": 0,
                  "last_character_index": 5
                }
              ]
            }
          ],
          "k": 2
        }
        """,
        encoding="utf-8",
    )

    report = validate_retrieval_file(
        results_path,
        tmp_path,
        corpus_root,
    )

    assert report.result_count == 1
    assert report.source_count == 2
    assert report.valid_source_count == 1
    assert [issue.kind for issue in report.issues] == [
        SourceValidationIssueKind.UNKNOWN_PATH
    ]


def test_invalid_results_json_fails_before_corpus_loading(
    tmp_path: Path,
) -> None:
    """Malformed result input is rejected even when corpus is missing."""
    results_path = tmp_path / "results.json"
    results_path.write_text("not JSON", encoding="utf-8")

    with pytest.raises(ValueError, match="Retrieval results JSON is invalid"):
        validate_retrieval_file(
            results_path,
            tmp_path,
            tmp_path / "missing-corpus",
        )
