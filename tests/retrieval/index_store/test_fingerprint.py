"""Tests for deterministic corpus identity calculation."""

from pathlib import Path

import pytest

from src.ingestion import CorpusFile, FileKind
from src.retrieval.index_store import fingerprint_corpus


def _corpus_files() -> list[CorpusFile]:
    """Return a deliberately unsorted two-file manifest."""
    return [
        CorpusFile(file_path="corpus/b.md", kind=FileKind.TEXT),
        CorpusFile(file_path="corpus/a.py", kind=FileKind.PYTHON),
    ]


def test_fingerprint_is_stable_for_manifest_order(tmp_path: Path) -> None:
    """Discovery order cannot change the corpus identity."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.py").write_text("value = 1\n", encoding="utf-8")
    (corpus / "b.md").write_text("# Cache\n", encoding="utf-8")
    files = _corpus_files()

    assert fingerprint_corpus(tmp_path, files) == fingerprint_corpus(
        tmp_path,
        list(reversed(files)),
    )


def test_fingerprint_changes_with_source_bytes(tmp_path: Path) -> None:
    """Any source content change invalidates the stored index identity."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.py").write_text("value = 1\n", encoding="utf-8")
    (corpus / "b.md").write_text("# Cache\n", encoding="utf-8")
    before = fingerprint_corpus(tmp_path, _corpus_files())

    (corpus / "a.py").write_text("value = 2\n", encoding="utf-8")

    assert fingerprint_corpus(tmp_path, _corpus_files()) != before


def test_fingerprint_rejects_duplicate_paths(tmp_path: Path) -> None:
    """An ambiguous manifest cannot produce a trusted identity."""
    source = tmp_path / "cache.py"
    source.write_text("cache = True\n", encoding="utf-8")
    corpus_file = CorpusFile(file_path="cache.py", kind=FileKind.PYTHON)

    with pytest.raises(ValueError, match="paths must be unique"):
        fingerprint_corpus(tmp_path, [corpus_file, corpus_file])
