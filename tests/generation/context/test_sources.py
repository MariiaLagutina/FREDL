"""Tests for safely loading only retrieved context source files."""

from pathlib import Path

import pytest

from src.generation.context import load_ranked_source_texts
from src.models import MinimalSource


def _source(path: str, start: int = 0, end: int = 1) -> MinimalSource:
    """Create one retrieved source fixture."""
    return MinimalSource(
        file_path=path,
        first_character_index=start,
        last_character_index=end,
    )


def test_loader_reads_only_unique_retrieved_files_and_preserves_crlf(
    tmp_path: Path,
) -> None:
    """Unreferenced files stay unloaded and exact decoded text is retained."""
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    selected = corpus_root / "selected.md"
    selected.write_bytes("Grüße\r\n".encode("utf-8"))
    (corpus_root / "unselected.md").write_text("unused", encoding="utf-8")
    source = _source("corpus/selected.md")

    texts = load_ranked_source_texts(
        tmp_path,
        corpus_root,
        [source, source],
    )

    assert texts == {"corpus/selected.md": "Grüße\r\n"}


@pytest.mark.parametrize(
    "file_path",
    [
        "outside.md",
        "corpus/../outside.md",
        "corpus/./guide.md",
    ],
)
def test_loader_rejects_sources_outside_boundary_or_noncanonical_paths(
    tmp_path: Path,
    file_path: str,
) -> None:
    """Retrieved metadata cannot escape or disguise a corpus path."""
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (tmp_path / "outside.md").write_text("outside", encoding="utf-8")
    (corpus_root / "guide.md").write_text("guide", encoding="utf-8")

    with pytest.raises(ValueError, match="Context source"):
        load_ranked_source_texts(
            tmp_path,
            corpus_root,
            [_source(file_path)],
        )


def test_loader_rejects_symlink_that_leaves_corpus(tmp_path: Path) -> None:
    """A source symlink cannot bypass containment validation."""
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    (corpus_root / "link.md").symlink_to(outside)

    with pytest.raises(ValueError, match="inside the corpus root"):
        load_ranked_source_texts(
            tmp_path,
            corpus_root,
            [_source("corpus/link.md")],
        )
