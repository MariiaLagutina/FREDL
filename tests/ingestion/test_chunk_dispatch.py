"""Tests for selecting the document chunking strategy."""

import pytest

from src.ingestion import (
    FileKind,
    SourceDocument,
    chunk_document,
)


def test_dispatcher_uses_python_strategy() -> None:
    """Python documents retain structural function boundaries."""
    document = SourceDocument(
        "src/example.py",
        FileKind.PYTHON,
        "def value():\n    return 1\n",
    )

    chunks = chunk_document(document)

    assert chunks[0].text == "def value():\n    return 1"
    assert chunks[0].section_path == ()


def test_dispatcher_uses_markdown_strategy() -> None:
    """Markdown documents retain their active heading metadata."""
    document = SourceDocument(
        "docs/guide.md",
        FileKind.TEXT,
        "# Guide\nRead this.\n",
    )

    chunks = chunk_document(document)

    assert chunks[0].text == document.text
    assert chunks[0].section_path == ("Guide",)


@pytest.mark.parametrize("maximum", [0, 2001])
def test_dispatcher_forwards_invalid_size(maximum: int) -> None:
    """Strategy validation remains visible through the public dispatcher."""
    document = SourceDocument("notes.txt", FileKind.TEXT, "Text.\n")

    with pytest.raises(
        ValueError,
        match="Maximum chunk size must be between 1 and 2000",
    ):
        chunk_document(document, maximum)


def test_dispatcher_forwards_custom_maximum() -> None:
    """The selected strategy receives the caller's exact size limit."""
    document = SourceDocument(
        "notes.txt",
        FileKind.TEXT,
        "abcdefghij",
    )

    chunks = chunk_document(document, max_chunk_size=4)

    assert all(len(chunk.text) <= 4 for chunk in chunks)
