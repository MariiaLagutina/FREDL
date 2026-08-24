"""Tests for structural sections inside oversized Python functions."""

import ast

from src.ingestion.python_positions import _PythonSourceMap
from src.ingestion.python_sections import (
    _function_sections,
    _SectionKind,
    _top_level_sections,
)


def test_function_sections_partition_direct_statements_exactly() -> None:
    """A function header and direct statements cover its exact source."""
    source = (
        "def process(value):\n"
        "    # Validate first.\n"
        "    validate(value)\n"
        "    if value.ready:\n"
        "        save(value)\n"
        "    return value"
    )
    source_map = _PythonSourceMap(source)
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.FunctionDef)
    function_span = _top_level_sections(
        ast.parse(source),
        source_map,
    )[0].span

    sections = _function_sections(function, function_span, source_map)

    assert [section.kind for section in sections] == [
        _SectionKind.GAP,
        _SectionKind.STATEMENT,
        _SectionKind.GAP,
        _SectionKind.STATEMENT,
        _SectionKind.GAP,
        _SectionKind.STATEMENT,
    ]
    assert "".join(
        source[section.span.start:section.span.end]
        for section in sections
    ) == source
    assert source[
        sections[1].span.start:sections[1].span.end
    ].startswith("    # Validate first.\n    validate")


def test_function_sections_keep_nested_block_together() -> None:
    """Nested statements remain inside their direct compound statement."""
    source = (
        "def process(items):\n"
        "    for item in items:\n"
        "        if item.ready:\n"
        "            save(item)"
    )
    module = ast.parse(source)
    function = module.body[0]
    assert isinstance(function, ast.FunctionDef)
    source_map = _PythonSourceMap(source)
    function_span = _top_level_sections(module, source_map)[0].span

    sections = _function_sections(function, function_span, source_map)

    statements = [
        section for section in sections
        if section.kind is _SectionKind.STATEMENT
    ]
    assert len(statements) == 1
    statement_text = source[
        statements[0].span.start:statements[0].span.end
    ]
    assert "if item.ready:" in statement_text
