"""Tests for public Markdown and plain-text chunk construction."""

import pytest

from src.ingestion import (
    FileKind,
    SourceDocument,
    chunk_text_document,
)


def make_text_document(
    text: str,
    file_path: str = "docs/guide.md",
) -> SourceDocument:
    """Create one in-memory text fixture."""
    return SourceDocument(file_path=file_path, kind=FileKind.TEXT, text=text)


def assert_exact_chunks(document: SourceDocument, maximum: int) -> None:
    """Require exact, non-empty, bounded source slices."""
    chunks = chunk_text_document(document, maximum)

    assert all(chunk.text for chunk in chunks)
    assert all(len(chunk.text) <= maximum for chunk in chunks)
    assert all(
        chunk.text == document.text[chunk.start:chunk.end]
        for chunk in chunks
    )


def test_markdown_chunker_packs_blocks_inside_one_section() -> None:
    """A heading, paragraph, and list share their active section path."""
    source = "# Install\n\nUse uv.\n\n- Sync\n- Run\n"
    document = make_text_document(source)

    chunks = chunk_text_document(document)

    assert len(chunks) == 1
    assert chunks[0].text == source
    assert chunks[0].section_path == ("Install",)


def test_markdown_chunker_starts_a_chunk_for_each_heading_path() -> None:
    """Content from different Markdown sections is not packed together."""
    source = "# Guide\nIntro.\n## Linux\nUse uv.\n## macOS\nUse brew.\n"
    document = make_text_document(source)

    chunks = chunk_text_document(document)

    assert [chunk.section_path for chunk in chunks] == [
        ("Guide",),
        ("Guide", "Linux"),
        ("Guide", "macOS"),
    ]
    assert [chunk.text.splitlines()[0] for chunk in chunks] == [
        "# Guide",
        "## Linux",
        "## macOS",
    ]


def test_plain_text_does_not_interpret_hash_lines_as_headings() -> None:
    """TXT files use paragraph boundaries without Markdown semantics."""
    source = "# Literal hash line\nOrdinary text.\n"
    document = make_text_document(source, file_path="notes/readme.txt")

    chunks = chunk_text_document(document)

    assert len(chunks) == 1
    assert chunks[0].text == source
    assert chunks[0].section_path == ()


def test_oversized_text_block_uses_exact_line_fallback() -> None:
    """A long paragraph prefers complete lines before character limits."""
    source = "First line is long.\nSecond line is long.\nThird line.\n"
    document = make_text_document(source, file_path="notes/readme.txt")

    chunks = chunk_text_document(document, max_chunk_size=42)

    assert chunks[0].text == "First line is long.\nSecond line is long.\n"
    assert_exact_chunks(document, maximum=42)


def test_single_character_limit_makes_progress_across_crlf() -> None:
    """An impossible one-character CRLF limit never creates an empty span."""
    document = make_text_document("a\r\nb", file_path="notes/readme.txt")

    chunks = chunk_text_document(document, max_chunk_size=1)

    assert [chunk.text for chunk in chunks] == ["a", "b"]
    assert_exact_chunks(document, maximum=1)


@pytest.mark.parametrize("maximum", [0, -1, 2001])
def test_text_chunker_rejects_invalid_maximum(maximum: int) -> None:
    """Text chunks obey the evaluator-facing maximum size contract."""
    with pytest.raises(
        ValueError,
        match="Maximum chunk size must be between 1 and 2000",
    ):
        chunk_text_document(make_text_document("Text."), maximum)


def test_text_chunker_rejects_python_document() -> None:
    """The text strategy cannot silently process Python source."""
    document = SourceDocument("src/app.py", FileKind.PYTHON, "value = 1\n")

    with pytest.raises(
        ValueError,
        match="Text chunker requires a text source document",
    ):
        chunk_text_document(document)


def test_text_chunker_returns_no_chunks_for_empty_source() -> None:
    """An empty file does not create an empty retrieval record."""
    assert chunk_text_document(make_text_document("")) == []
