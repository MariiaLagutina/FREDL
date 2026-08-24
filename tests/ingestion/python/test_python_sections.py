"""Tests for exact top-level Python source partitioning."""

import ast

from src.ingestion.chunking.python.positions import _PythonSourceMap
from src.ingestion.chunking.python.sections import (
    _class_sections,
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


def test_sections_attach_comment_before_decorator() -> None:
    """A directly preceding comment becomes part of the definition."""
    source = "# Explain cache.\n@cache\ndef value():\n    return 1\n"

    sections = partition(source)

    definition_text = source[
        sections[0].span.start:sections[0].span.end
    ]
    assert sections[0].kind is _SectionKind.DEFINITION
    assert definition_text.startswith("# Explain cache.\n@cache\n")


def test_sections_attach_consecutive_comment_lines() -> None:
    """One uninterrupted comment block stays with its definition."""
    source = "# First.\n# Second.\ndef value():\n    return 1\n"

    sections = partition(source)

    definition = sections[0]
    assert definition.kind is _SectionKind.DEFINITION
    assert source[
        definition.span.start:definition.span.end
    ].startswith("# First.\n# Second.\n")


def test_sections_do_not_attach_comment_across_blank_line() -> None:
    """A blank line separates a module comment from a definition."""
    source = "# Module note.\n\ndef value():\n    return 1\n"

    sections = partition(source)

    assert sections[0].kind is _SectionKind.GAP
    assert source[
        sections[0].span.start:sections[0].span.end
    ] == "# Module note.\n\n"


def test_sections_attach_comment_but_not_previous_statement() -> None:
    """Only comments after module code move into the next definition."""
    source = "VALUE = 1\n# Explain function.\ndef value():\n    return VALUE\n"

    sections = partition(source)

    gap_text = source[sections[0].span.start:sections[0].span.end]
    definition_text = source[
        sections[1].span.start:sections[1].span.end
    ]
    assert gap_text == "VALUE = 1\n"
    assert definition_text.startswith("# Explain function.\ndef value")


def test_sections_keep_shebang_and_encoding_comment_in_gap() -> None:
    """Interpreter and encoding declarations remain module metadata."""
    source = (
        "#!/usr/bin/env python3\n"
        "# -*- coding: utf-8 -*-\n"
        "def value():\n"
        "    return 1\n"
    )

    sections = partition(source)

    gap_text = source[sections[0].span.start:sections[0].span.end]
    assert sections[0].kind is _SectionKind.GAP
    assert gap_text == (
        "#!/usr/bin/env python3\n"
        "# -*- coding: utf-8 -*-\n"
    )


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


def test_class_sections_partition_direct_methods_exactly() -> None:
    """A class header and its methods form exact ordered source slices."""
    source = (
        "class Server:\n"
        "    port = 8000\n\n"
        "    # Start the service.\n"
        "    def start(self):\n"
        "        return True\n\n"
        "    async def stop(self):\n"
        "        return False"
    )
    source_map = _PythonSourceMap(source)
    class_node = ast.parse(source).body[0]
    assert isinstance(class_node, ast.ClassDef)
    class_span = partition(source)[0].span

    sections = _class_sections(class_node, class_span, source_map)

    assert [section.kind for section in sections] == [
        _SectionKind.GAP,
        _SectionKind.DEFINITION,
        _SectionKind.GAP,
        _SectionKind.DEFINITION,
    ]
    assert "".join(
        source[section.span.start:section.span.end]
        for section in sections
    ) == source
    assert source[
        sections[1].span.start:sections[1].span.end
    ].startswith("    # Start the service.\n    def start")


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
