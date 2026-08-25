"""Tests for qualified Python symbol extraction."""

from src.ingestion import (
    extract_python_symbol_spans,
    FileKind,
    SourceDocument,
)


def _python_document(text: str) -> SourceDocument:
    """Return a Python document with stable test metadata."""
    return SourceDocument(
        file_path="data/raw/corpus/cache.py",
        kind=FileKind.PYTHON,
        text=text,
    )


def test_symbol_spans_preserve_nested_qualified_names() -> None:
    """Classes, methods, and nested functions retain their parent path."""
    source = (
        "class CacheStore:\n"
        "    def get_item(self):\n"
        "        def normalize(value):\n"
        "            return value\n"
        "        return normalize(self.value)\n"
    )

    symbols = extract_python_symbol_spans(_python_document(source))

    assert [symbol.qualified_name for symbol in symbols] == [
        "CacheStore",
        "CacheStore.get_item",
        "CacheStore.get_item.normalize",
    ]
    assert all(
        source[symbol.start:symbol.end].strip()
        for symbol in symbols
    )


def test_symbol_spans_use_character_offsets_for_unicode_source() -> None:
    """UTF-8 AST columns do not leak into character-based chunk positions."""
    source = "grüße = 1\n\nclass Größe:\n    pass\n"

    symbol = extract_python_symbol_spans(_python_document(source))[0]

    assert symbol.qualified_name == "Größe"
    assert source[symbol.start:symbol.end] == "class Größe:\n    pass"


def test_symbol_extraction_returns_empty_on_invalid_python() -> None:
    """A syntax-error fallback document has no speculative symbol metadata."""
    assert extract_python_symbol_spans(
        _python_document("def broken(:\n")
    ) == ()


def test_symbol_extraction_rejects_text_documents() -> None:
    """Format-specific extraction fails clearly for documentation input."""
    document = SourceDocument(
        file_path="data/raw/corpus/readme.md",
        kind=FileKind.TEXT,
        text="# Readme\n",
    )

    try:
        extract_python_symbol_spans(document)
    except ValueError as error:
        assert str(error) == "Python symbol extraction requires Python source"
    else:
        raise AssertionError("Expected text document rejection")
