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


def test_extracts_assignment_targets_without_other_identifiers() -> None:
    """Parameters and keyword arguments do not enter the focused field."""
    source = (
        "LIMIT = 4\n"
        "def run(value, *, enabled=False):\n"
        "    self.result = call(option=value)\n"
    )

    spans = extract_python_identifier_spans(_python_document(source))

    assert [span.identifier for span in spans] == [
        "LIMIT",
        "result",
    ]
    assert all(source[span.start:span.end] for span in spans)


def test_extracts_destructured_assignment_targets() -> None:
    """Nested assignment targets remain visible without parameters."""
    source = "def collect(*items, **options):\n    first, *rest = items\n"

    spans = extract_python_identifier_spans(_python_document(source))

    assert [span.identifier for span in spans] == [
        "first",
        "rest",
    ]


def test_invalid_python_has_no_speculative_identifiers() -> None:
    """Syntax errors follow the existing symbol-extraction fallback."""
    assert extract_python_identifier_spans(
        _python_document("def broken(:\n")
    ) == ()
