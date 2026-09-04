"""Tests for extracting identifiers from structural Python sites."""

from src.ingestion import (
    extract_python_identifier_spans,
    FileKind,
    SourceDocument,
)


def _python_document(text: str) -> SourceDocument:
    """Return a Python document with stable test metadata."""
    return SourceDocument(
        file_path="data/raw/corpus/example.py",
        kind=FileKind.PYTHON,
        text=text,
    )


def test_extracts_parameters_assignments_and_keyword_arguments() -> None:
    """Only identifiers at selected structural sites become metadata."""
    source = (
        "LIMIT = 4\n"
        "def run(value, *, enabled=False):\n"
        "    self.result = call(option=value)\n"
    )

    spans = extract_python_identifier_spans(_python_document(source))

    assert [span.identifier for span in spans] == [
        "LIMIT",
        "value",
        "enabled",
        "result",
        "option",
    ]
    assert all(source[span.start:span.end] for span in spans)


def test_extracts_destructured_and_variable_arguments() -> None:
    """Nested targets and both variable parameter forms remain visible."""
    source = "def collect(*items, **options):\n    first, *rest = items\n"

    spans = extract_python_identifier_spans(_python_document(source))

    assert [span.identifier for span in spans] == [
        "items",
        "options",
        "first",
        "rest",
    ]


def test_invalid_python_has_no_speculative_identifiers() -> None:
    """Syntax errors follow the existing symbol-extraction fallback."""
    assert extract_python_identifier_spans(
        _python_document("def broken(:\n")
    ) == ()
