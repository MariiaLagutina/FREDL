"""Tests for exact Python AST position conversion."""

import ast

import pytest

from src.ingestion.python import (
    _node_span,
    _PythonSourceMap,
    _StructuralSpan,
)


def test_source_map_converts_ascii_ast_position() -> None:
    """ASCII AST columns match ordinary character columns."""
    source = "first = 1\nsecond = 2\n"
    assignment = ast.parse(source).body[1]

    offset = _PythonSourceMap(source).character_offset(
        assignment.lineno,
        assignment.col_offset,
    )

    assert offset == source.index("second")


def test_source_map_converts_utf8_byte_column() -> None:
    """A multi-byte character before a node does not shift its position."""
    source = "é = 1; value = 2\n"
    assignment = ast.parse(source).body[1]

    offset = _PythonSourceMap(source).character_offset(
        assignment.lineno,
        assignment.col_offset,
    )

    assert assignment.col_offset == 8
    assert offset == 7
    assert source[offset:].startswith("value")


def test_source_map_preserves_crlf_absolute_offsets() -> None:
    """Line starts count original CRLF characters exactly."""
    source = "grüße = 1\r\nvalue = 2\r\n"
    assignment = ast.parse(source).body[1]

    offset = _PythonSourceMap(source).character_offset(
        assignment.lineno,
        assignment.col_offset,
    )

    assert offset == source.index("value")


def test_source_map_rejects_column_inside_line_ending() -> None:
    """A byte position cannot point into CRLF separator characters."""
    source_map = _PythonSourceMap("value\r\nnext\r\n")

    with pytest.raises(
        ValueError,
        match="AST UTF-8 column is outside its source line",
    ):
        source_map.character_offset(line=1, utf8_column=6)


@pytest.mark.parametrize(
    ("line", "column", "message"),
    [
        (0, 0, "AST line is outside the source text"),
        (2, 0, "AST line is outside the source text"),
        (1, -1, "AST UTF-8 column must not be negative"),
        (1, 100, "AST UTF-8 column is outside its source line"),
        (1, 1, "AST UTF-8 column splits a source character"),
    ],
)
def test_source_map_rejects_invalid_position(
    line: int,
    column: int,
    message: str,
) -> None:
    """Invalid AST coordinates fail before producing a wrong source span."""
    with pytest.raises(ValueError, match=message):
        _PythonSourceMap("é = 1").character_offset(line, column)


def test_node_span_maps_plain_function() -> None:
    """A function node maps to its exact half-open source range."""
    source = "prefix = 'grüße'\n\ndef hello():\n    return 1\n"
    function = ast.parse(source).body[1]

    span = _node_span(function, _PythonSourceMap(source))

    assert span == _StructuralSpan(
        start=source.index("def hello"),
        end=source.index("1\n", source.index("def hello")) + 1,
    )
    assert source[span.start:span.end] == "def hello():\n    return 1"


@pytest.mark.parametrize(
    "source",
    [
        "@cache\n@validate(arg=True)\ndef hello():\n    return 1\n",
        "@cache\nasync def hello():\n    return 1\n",
        "@dataclass\nclass Server:\n    port = 8000\n",
    ],
)
def test_node_span_includes_definition_decorators(source: str) -> None:
    """Decorated definitions begin at the first source @ marker."""
    definition = ast.parse(source).body[0]

    span = _node_span(definition, _PythonSourceMap(source))

    assert span.start == 0
    assert source[span.start] == "@"
    assert source[span.start:span.end].startswith("@")


def test_node_span_includes_indented_method_decorator() -> None:
    """A method span starts at its indented decorator marker."""
    source = (
        "class Server:\n"
        "    @staticmethod\n"
        "    def port():\n"
        "        return 8000\n"
    )
    class_node = ast.parse(source).body[0]
    assert isinstance(class_node, ast.ClassDef)
    method = class_node.body[0]

    span = _node_span(method, _PythonSourceMap(source))

    assert span.start == source.index("@staticmethod")
    assert source[span.start:span.end].startswith("@staticmethod")


def test_node_span_rejects_node_without_positions() -> None:
    """Synthetic or operator nodes cannot produce invented source ranges."""
    with pytest.raises(
        ValueError,
        match="AST node does not provide a complete source range",
    ):
        _node_span(ast.Add(), _PythonSourceMap("1 + 2"))
