"""Tests for the RAG command-line boundary."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.__main__ import main


def test_search_command_passes_explicit_paths_and_parameters() -> None:
    """CLI arguments reach the stored-index workflow without changes."""
    with patch("src.__main__.run_stored_retrieval") as run_retrieval:
        main(
            [
                "search",
                "--index",
                "cache/bm25-index.json",
                "--fingerprint",
                "a" * 64,
                "--input",
                "questions.json",
                "--output",
                "search-results.json",
                "--k",
                "3",
            ]
        )

    run_retrieval.assert_called_once_with(
        Path("cache/bm25-index.json"),
        "a" * 64,
        Path("questions.json"),
        Path("search-results.json"),
        k=3,
    )


def test_search_command_rejects_non_positive_k_before_loading_index() -> None:
    """Invalid limits fail at parsing without opening a large snapshot."""
    with patch("src.__main__.run_stored_retrieval") as run_retrieval:
        with pytest.raises(SystemExit):
            main(
                [
                    "search",
                    "--index",
                    "cache/bm25-index.json",
                    "--fingerprint",
                    "a" * 64,
                    "--input",
                    "questions.json",
                    "--output",
                    "search-results.json",
                    "--k",
                    "0",
                ]
            )

    run_retrieval.assert_not_called()
