"""Tests for one-pass validation corpus loading."""

from pathlib import Path

import pytest

from src.retrieval.validation import load_source_texts


def test_load_source_texts_preserves_exact_discovered_content(
    tmp_path: Path,
) -> None:
    """Validation uses canonical paths and text without newline changes."""
    corpus_root = tmp_path / "data" / "raw"
    corpus_root.mkdir(parents=True)
    (corpus_root / "module.py").write_text(
        "größen_limit = 1\r\n",
        encoding="utf-8",
        newline="",
    )
    (corpus_root / "notes.md").write_text(
        "# Cache\n",
        encoding="utf-8",
    )
    (corpus_root / "ignored.json").write_text(
        '{"not": "source text"}',
        encoding="utf-8",
    )

    texts = load_source_texts(tmp_path, corpus_root)

    assert texts == {
        "data/raw/module.py": "größen_limit = 1\r\n",
        "data/raw/notes.md": "# Cache\n",
    }


def test_load_source_texts_reports_missing_corpus(tmp_path: Path) -> None:
    """A missing corpus remains a visible filesystem boundary error."""
    with pytest.raises(FileNotFoundError):
        load_source_texts(tmp_path, tmp_path / "missing")
