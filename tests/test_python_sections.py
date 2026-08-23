"""Tests for exact top-level Python source partitioning."""

import ast

from src.ingestion.python import (
    _PythonSourceMap,
    _SectionKind,
    _SourceSection,
    _top_level_sections,
)


def partition(source: str) -> list[_SourceSection]:
    """Return internal top-level sections for a source fixture."""
    return _top_level_sections(ast.parse(source), _PythonSourceMap(source))


def test_sections_cover_single_function_and_trailing_newline() -> None:
    """A definition and final newline cover the complete source exactly."""
    source = "def hello():\n    return 1\n"

    sections = partition(source)

    assert [section.kind for section in sections] == [
        _SectionKind.DEFINITION,
        _SectionKind.GAP,
    ]
    assert "".join(
        source[section.span.start:section.span.end]
        for section in sections
    ) == source


def test_sections_preserve_text_around_multiple_definitions() -> None:
    """Prefix, middle, and suffix gaps remain in their source order."""
    source = (
        '"""Module docs."""\n\n'
        "import os\n\n"
        "def first():\n"
        "    return 1\n\n"
        "# Between definitions.\n"
        "async def second():\n"
        "    return 2\n\n"
        "RESULT = first()\n"
    )

    sections = partition(source)

    assert [section.kind for section in sections] == [
        _SectionKind.GAP,
        _SectionKind.DEFINITION,
        _SectionKind.GAP,
        _SectionKind.DEFINITION,
        _SectionKind.GAP,
    ]
    assert "".join(
        source[section.span.start:section.span.end]
        for section in sections
    ) == source


def test_sections_keep_comment_before_decorator_in_gap() -> None:
    """Comment attachment is deferred without losing source characters."""
    source = "# Explain cache.\n@cache\ndef value():\n    return 1\n"

    sections = partition(source)

    first_text = source[
        sections[0].span.start:sections[0].span.end
    ]
    definition_text = source[
        sections[1].span.start:sections[1].span.end
    ]
    assert sections[0].kind is _SectionKind.GAP
    assert first_text == "# Explain cache.\n"
    assert definition_text.startswith("@cache\n")


def test_sections_treat_top_level_class_as_one_definition() -> None:
    """Methods remain inside their containing top-level class span."""
    source = (
        "class Server:\n"
        "    def start(self):\n"
        "        return True\n"
    )

    sections = partition(source)

    assert sections[0].kind is _SectionKind.DEFINITION
    assert source[
        sections[0].span.start:sections[0].span.end
    ].startswith("class Server:")
    assert sum(
        section.kind is _SectionKind.DEFINITION
        for section in sections
    ) == 1


def test_sections_return_one_gap_when_module_has_no_definitions() -> None:
    """Module-level statements remain available when no definitions exist."""
    source = "import os\n\nVALUE = 42\n"

    sections = partition(source)

    assert len(sections) == 1
    assert sections[0].kind is _SectionKind.GAP
    assert sections[0].span.start == 0
    assert sections[0].span.end == len(source)


def test_sections_return_empty_list_for_empty_module() -> None:
    """An empty source has no non-empty ranges to partition."""
    assert partition("") == []
