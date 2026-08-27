"""Tests for the RAG command-line boundary."""

from pathlib import Path
from unittest.mock import patch

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
