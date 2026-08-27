"""Tests for the assignment-compatible Python Fire CLI."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.__main__ import main


FINGERPRINT = "a" * 64


def test_search_command_routes_one_raw_query() -> None:
    """The Fire search command reaches the stored single-query workflow."""
    with (
        patch(
            "src.cli._current_corpus_fingerprint",
            return_value=FINGERPRINT,
        ),
        patch("src.cli.run_stored_search", return_value=[]) as run_search,
    ):
        main(["search", "Where is the cache?", "--k", "3"])

    run_search.assert_called_once_with(
        Path("data/processed/bm25-index.json"),
        FINGERPRINT,
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
        Path("data/datasets/questions.json"),
        Path("data/output/search_results/Public/questions.json"),
        3,
    )


def test_search_rejects_non_positive_k_before_corpus_scan() -> None:
    """An invalid limit fails before fingerprint or index work begins."""
    with patch("src.cli._current_corpus_fingerprint") as fingerprint:
        with pytest.raises(ValueError, match="greater than zero"):
            main(["search", "cache", "--k", "0"])

    fingerprint.assert_not_called()
