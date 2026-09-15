"""Tests for deterministic corpus identity calculation."""

from pathlib import Path

import pytest

from src.ingestion import CorpusFile, FileKind
from src.retrieval.index_store import fingerprint_corpus, fingerprint_file


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


def test_fingerprint_file_changes_only_for_its_own_content(
    tmp_path: Path,
) -> None:
    """One file's identity is independent from a sibling file's content."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.py").write_text("value = 1\n", encoding="utf-8")
    (corpus / "b.md").write_text("# Cache\n", encoding="utf-8")
    a_file = CorpusFile(file_path="corpus/a.py", kind=FileKind.PYTHON)
    b_file = CorpusFile(file_path="corpus/b.md", kind=FileKind.TEXT)
    a_before = fingerprint_file(tmp_path, a_file)
    b_before = fingerprint_file(tmp_path, b_file)

    (corpus / "b.md").write_text("# Cache updated\n", encoding="utf-8")

    assert fingerprint_file(tmp_path, a_file) == a_before
    assert fingerprint_file(tmp_path, b_file) != b_before


def test_fingerprint_file_distinguishes_identical_content_by_path(
    tmp_path: Path,
) -> None:
    """Two files with the same bytes at different paths hash differently."""
    corpus = tmp_path / "corpus"
    (corpus / "docs").mkdir(parents=True)
    (corpus / "src").mkdir()
    (corpus / "docs" / "note.txt").write_text("same", encoding="utf-8")
    (corpus / "src" / "note.txt").write_text("same", encoding="utf-8")

    docs_file = CorpusFile(
        file_path="corpus/docs/note.txt", kind=FileKind.TEXT
    )
    src_file = CorpusFile(
        file_path="corpus/src/note.txt", kind=FileKind.TEXT
    )
    docs_fingerprint = fingerprint_file(tmp_path, docs_file)
    src_fingerprint = fingerprint_file(tmp_path, src_file)

    assert docs_fingerprint != src_fingerprint
