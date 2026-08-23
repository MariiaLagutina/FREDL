"""Tests for safe and deterministic corpus file discovery."""

from pathlib import Path

import pytest

from src.ingestion.files import discover_files


def write_file(path: Path, content: str = "content") -> None:
    """Create a text fixture and any missing parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_discover_files_returns_canonical_paths_and_kinds(
    tmp_path: Path,
) -> None:
    """Discovery returns project-relative POSIX paths and chunking kinds."""
    corpus_root = tmp_path / "data" / "raw" / "vllm-0.10.1"
    write_file(corpus_root / "vllm" / "engine.py")
    write_file(corpus_root / "docs" / "guide.md")
    write_file(corpus_root / "CMakeLists.txt")

    manifest = discover_files(tmp_path, corpus_root)

    assert [item.model_dump(mode="json") for item in manifest] == [
        {
            "file_path": "data/raw/vllm-0.10.1/CMakeLists.txt",
            "kind": "text",
        },
        {
            "file_path": "data/raw/vllm-0.10.1/docs/guide.md",
            "kind": "text",
        },
        {
            "file_path": "data/raw/vllm-0.10.1/vllm/engine.py",
            "kind": "python",
        },
    ]


def test_discover_files_ignores_unsupported_and_hidden_entries(
    tmp_path: Path,
) -> None:
    """Discovery excludes unsupported files and hidden directories."""
    corpus_root = tmp_path / "corpus"
    write_file(corpus_root / "module.py")
    write_file(corpus_root / "image.png")
    write_file(corpus_root / ".hidden.py")
    write_file(corpus_root / ".github" / "instructions.md")
    write_file(corpus_root / "__pycache__" / "cached.py")

    manifest = discover_files(tmp_path, corpus_root)

    assert [item.file_path for item in manifest] == ["corpus/module.py"]


def test_discover_files_does_not_follow_symlinks(tmp_path: Path) -> None:
    """Discovery excludes symlinked files and directories."""
    corpus_root = tmp_path / "corpus"
    outside_root = tmp_path / "outside"
    write_file(corpus_root / "real.py")
    write_file(outside_root / "secret.py")
    (corpus_root / "linked.py").symlink_to(outside_root / "secret.py")
    (corpus_root / "linked-directory").symlink_to(
        outside_root,
        target_is_directory=True,
    )

    manifest = discover_files(tmp_path, corpus_root)

    assert [item.file_path for item in manifest] == ["corpus/real.py"]


def test_discover_files_rejects_corpus_outside_project(
    tmp_path: Path,
) -> None:
    """Reject a corpus that cannot produce project-relative paths."""
    project_root = tmp_path / "project"
    corpus_root = tmp_path / "corpus"
    project_root.mkdir()
    corpus_root.mkdir()

    with pytest.raises(
        ValueError,
        match="Corpus root must be inside the project root",
    ):
        discover_files(project_root, corpus_root)


def test_discover_files_rejects_project_root_as_corpus(
    tmp_path: Path,
) -> None:
    """Discovery prevents accidentally indexing the whole project."""
    with pytest.raises(
        ValueError,
        match="Corpus root must be below the project root",
    ):
        discover_files(tmp_path, tmp_path)


def test_discover_files_rejects_non_directory_corpus(
    tmp_path: Path,
) -> None:
    """Discovery rejects an existing corpus path that is not a directory."""
    corpus_file = tmp_path / "corpus.txt"
    write_file(corpus_file)

    with pytest.raises(NotADirectoryError):
        discover_files(tmp_path, corpus_file)
