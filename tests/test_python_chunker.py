"""Tests for bounded Python chunks with exact source offsets."""

import pytest

from src.ingestion import (
    FileKind,
    SourceDocument,
    chunk_python_document,
)


def make_python_document(text: str) -> SourceDocument:
    """Create one in-memory Python source fixture."""
    return SourceDocument(
        file_path="data/raw/example.py",
        kind=FileKind.PYTHON,
        text=text,
    )


def assert_exact_chunks(
    document: SourceDocument,
    max_chunk_size: int,
) -> None:
    """Require every result to be a non-empty exact bounded slice."""
    chunks = chunk_python_document(document, max_chunk_size)

    assert all(chunk.text for chunk in chunks)
    assert all(len(chunk.text) <= max_chunk_size for chunk in chunks)
    assert all(
        chunk.text == document.text[chunk.start:chunk.end]
        for chunk in chunks
    )


def test_chunker_keeps_small_top_level_structure() -> None:
    """Small module text and a function remain separate exact chunks."""
    source = "import os\n\ndef hello():\n    return os.name\n"
    document = make_python_document(source)

    chunks = chunk_python_document(document)

    assert [chunk.text for chunk in chunks] == [
        "import os\n\n",
        "def hello():\n    return os.name",
    ]


def test_chunker_keeps_decorators_with_their_definitions() -> None:
    """Decorated functions and classes start at their first at-sign."""
    source = (
        "@cache\n"
        "def value():\n"
        "    return 1\n\n"
        "@dataclass\n"
        "class Server:\n"
        "    port: int\n"
    )
    document = make_python_document(source)

    chunks = chunk_python_document(document)

    assert [chunk.text for chunk in chunks] == [
        "@cache\ndef value():\n    return 1",
        "@dataclass\nclass Server:\n    port: int",
    ]


def test_chunker_keeps_signature_docstring_and_body_together() -> None:
    """A bounded function retains its complete semantic definition."""
    source = (
        "def describe(value):\n"
        "    \"\"\"Return a readable value.\"\"\"\n"
        "    return str(value)\n"
    )
    document = make_python_document(source)

    chunks = chunk_python_document(document)

    assert chunks[0].text == source.rstrip("\n")


def test_chunker_splits_large_section_at_line_boundary() -> None:
    """Oversized module text prefers complete source lines."""
    source = "first = 1\nsecond = 2\nthird = 3\n"
    document = make_python_document(source)

    chunks = chunk_python_document(document, max_chunk_size=22)

    assert [chunk.text for chunk in chunks] == [
        "first = 1\nsecond = 2\n",
        "third = 3\n",
    ]
    assert_exact_chunks(document, max_chunk_size=22)


def test_chunker_falls_back_to_character_limit_for_long_line() -> None:
    """A line above the limit is split without changing source offsets."""
    source = "value = '" + "x" * 40 + "'\n"
    document = make_python_document(source)

    chunks = chunk_python_document(document, max_chunk_size=16)

    assert len(chunks) > 1
    assert_exact_chunks(document, max_chunk_size=16)


def test_chunker_splits_large_class_at_method_boundaries() -> None:
    """Direct methods become separate chunks when their class is oversized."""
    source = (
        "class Calculator:\n"
        "    factor = 2\n\n"
        "    def add(self, value):\n"
        "        return value + self.factor\n\n"
        "    def subtract(self, value):\n"
        "        return value - self.factor\n"
    )
    document = make_python_document(source)

    chunks = chunk_python_document(document, max_chunk_size=70)

    assert [chunk.text.lstrip().splitlines()[0] for chunk in chunks] == [
        "class Calculator:",
        "def add(self, value):",
        "def subtract(self, value):",
    ]
    assert chunks[1].text.startswith("    def add")
    assert chunks[2].text.startswith("    def subtract")
    assert_exact_chunks(document, max_chunk_size=70)


def test_chunker_splits_large_function_at_statement_boundaries() -> None:
    """Small direct statements are packed without splitting AST blocks."""
    source = (
        "def process(value):\n"
        "    validate(value)\n"
        "    normalize(value)\n"
        "    if value.ready:\n"
        "        save(value)\n"
        "    return value\n"
    )
    document = make_python_document(source)

    chunks = chunk_python_document(document, max_chunk_size=65)

    assert chunks[0].text == (
        "def process(value):\n"
        "    validate(value)\n"
        "    normalize(value)\n"
    )
    assert chunks[1].text.startswith("    if value.ready:")
    assert "        save(value)" in chunks[1].text
    assert_exact_chunks(document, max_chunk_size=65)


def test_chunker_falls_back_for_invalid_python() -> None:
    """A syntax error uses safe line splitting instead of losing the file."""
    source = "def broken(:\n    value = 1\n    return value\n"
    document = make_python_document(source)

    chunks = chunk_python_document(document, max_chunk_size=20)

    assert chunks
    assert_exact_chunks(document, max_chunk_size=20)


def test_chunker_preserves_unicode_and_crlf_offsets() -> None:
    """Unicode and original newlines survive chunk construction exactly."""
    source = "grüße = 'Hallo'\r\ndef value():\r\n    return grüße\r\n"
    document = make_python_document(source)

    first_run = chunk_python_document(document, max_chunk_size=30)
    second_run = chunk_python_document(document, max_chunk_size=30)

    assert first_run == second_run
    assert_exact_chunks(document, max_chunk_size=30)


def test_chunker_never_splits_a_crlf_pair() -> None:
    """A size boundary between CR and LF moves before the newline pair."""
    document = make_python_document("aaaa\r\nbbbb")

    chunks = chunk_python_document(document, max_chunk_size=5)

    assert [chunk.text for chunk in chunks] == ["aaaa", "\r\nbbb", "b"]
    assert_exact_chunks(document, max_chunk_size=5)


@pytest.mark.parametrize("max_chunk_size", [0, -1, 2001])
def test_chunker_rejects_invalid_maximum_size(
    max_chunk_size: int,
) -> None:
    """Chunk size stays within the Moulinette contract."""
    with pytest.raises(
        ValueError,
        match="Maximum chunk size must be between 1 and 2000",
    ):
        chunk_python_document(
            make_python_document("value = 1\n"),
            max_chunk_size=max_chunk_size,
        )


def test_chunker_rejects_non_python_document() -> None:
    """The Python strategy cannot silently process a text document."""
    document = SourceDocument(
        file_path="data/raw/example.md",
        kind=FileKind.TEXT,
        text="# Heading\n",
    )

    with pytest.raises(
        ValueError,
        match="Python chunker requires a Python source document",
    ):
        chunk_python_document(document)


def test_chunker_returns_no_chunks_for_empty_source() -> None:
    """An empty file does not create an empty retrieval record."""
    assert chunk_python_document(make_python_document("")) == []
