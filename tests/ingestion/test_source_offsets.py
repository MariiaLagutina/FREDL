"""Tests for exact document text and source chunk offsets."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.ingestion import (
    Chunk,
    CorpusFile,
    FileKind,
    SourceDocument,
    make_chunk,
    read_document,
)


def make_document(text: str = "0123456789") -> SourceDocument:
    """Create a representative in-memory source document."""
    return SourceDocument(
        file_path="data/raw/example.md",
        kind=FileKind.TEXT,
        text=text,
    )


def test_read_document_preserves_unicode_and_crlf(tmp_path: Path) -> None:
    """Reading keeps every decoded character and original line ending."""
    source_path = tmp_path / "data" / "raw" / "guide.md"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes("Überschrift\r\nGrüße\r\n".encode("utf-8"))
    corpus_file = CorpusFile(
        file_path="data/raw/guide.md",
        kind=FileKind.TEXT,
    )

    document = read_document(tmp_path, corpus_file)

    assert document == SourceDocument(
        file_path="data/raw/guide.md",
        kind=FileKind.TEXT,
        text="Überschrift\r\nGrüße\r\n",
    )


def test_read_document_rejects_source_outside_project(
    tmp_path: Path,
) -> None:
    """A manifest-like traversal path cannot escape the project root."""
    project_root = tmp_path / "project"
    outside_file = tmp_path / "outside.md"
    project_root.mkdir()
    outside_file.write_text("outside", encoding="utf-8")
    corpus_file = CorpusFile(
        file_path="../outside.md",
        kind=FileKind.TEXT,
    )

    with pytest.raises(
        ValueError,
        match="Source file must be inside the project root",
    ):
        read_document(project_root, corpus_file)


def test_read_document_rejects_non_canonical_manifest_path(
    tmp_path: Path,
) -> None:
    """Reading requires the exact project-relative path used in outputs."""
    source_path = tmp_path / "data" / "raw" / "guide.md"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("guide", encoding="utf-8")
    corpus_file = CorpusFile(
        file_path="data/raw/../raw/guide.md",
        kind=FileKind.TEXT,
    )

    with pytest.raises(
        ValueError,
        match="Source path must be project-relative canonical POSIX",
    ):
        read_document(tmp_path, corpus_file)


def test_make_chunk_uses_exact_half_open_slice() -> None:
    """The factory derives text directly from document coordinates."""
    document = make_document()

    chunk = make_chunk(document, start=2, end=6)

    assert chunk == Chunk(
        file_path="data/raw/example.md",
        start=2,
        end=6,
        text="2345",
    )


def test_make_chunk_supports_exact_overlap() -> None:
    """Overlapping chunks remain exact independent source slices."""
    document = make_document()

    first = make_chunk(document, start=0, end=6)
    second = make_chunk(document, start=4, end=10)

    assert first.text == "012345"
    assert second.text == "456789"
    assert first.end - second.start == 2


def test_make_chunk_stores_immutable_section_path() -> None:
    """Retrieval metadata stays separate from exact source text."""
    chunk = make_chunk(
        make_document(),
        start=0,
        end=4,
        section_path=("Guide", "Linux"),
    )

    assert chunk.section_path == ("Guide", "Linux")


def test_chunk_rejects_invalid_section_path() -> None:
    """Section metadata cannot contain empty path elements."""
    with pytest.raises(
        ValueError,
        match="Chunk section path must contain non-empty titles",
    ):
        Chunk(
            file_path="data/raw/example.md",
            start=0,
            end=4,
            text="text",
            section_path=("Guide", ""),
        )


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [
        (-1, 2, "Chunk start must not be negative"),
        (2, 2, "Chunk end must be greater than start"),
        (4, 3, "Chunk end must be greater than start"),
        (0, 11, "Chunk end must not exceed document length"),
    ],
)
def test_make_chunk_rejects_invalid_range(
    start: int,
    end: int,
    message: str,
) -> None:
    """The factory rejects ranges that cannot slice the source exactly."""
    with pytest.raises(ValueError, match=message):
        make_chunk(make_document(), start=start, end=end)


def test_chunk_rejects_text_that_does_not_match_span_length() -> None:
    """Direct model construction still enforces coordinate length."""
    with pytest.raises(
        ValueError,
        match="Chunk text length must match its character span",
    ):
        Chunk(
            file_path="data/raw/example.md",
            start=0,
            end=4,
            text="wrong",
        )


def test_source_text_and_chunk_coordinates_are_immutable() -> None:
    """Later pipeline stages cannot silently invalidate source offsets."""
    document = make_document()
    chunk = make_chunk(document, start=0, end=4)

    with pytest.raises(FrozenInstanceError):
        document.text = document.text.lower()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        chunk.end = 3  # type: ignore[misc]
