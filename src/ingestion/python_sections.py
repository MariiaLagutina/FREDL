"""Partition Python modules into definitions and surrounding source text."""

import ast
from bisect import bisect_right
from dataclasses import dataclass
from enum import Enum
import re

from src.ingestion.python_positions import (
    _node_span,
    _PythonSourceMap,
    _StructuralSpan,
)


class _SectionKind(str, Enum):
    """Identify whether a module section is structure or surrounding text."""

    GAP = "gap"
    DEFINITION = "definition"


@dataclass(frozen=True, slots=True)
class _SourceSection:
    """Classify one non-empty, exact range in a Python module."""

    kind: _SectionKind
    span: _StructuralSpan


def _top_level_sections(
    module: ast.Module,
    source_map: _PythonSourceMap,
) -> list[_SourceSection]:
    """Partition a module into top-level definitions and exact text gaps."""
    definition_types = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
    )
    definition_spans = [
        _span_with_attached_comments(_node_span(node, source_map), source_map)
        for node in module.body
        if isinstance(node, definition_types)
    ]

    sections: list[_SourceSection] = []
    cursor = 0

    for definition_span in definition_spans:
        if definition_span.start < cursor:
            message = "Top-level definition spans must not overlap"
            raise ValueError(message)
        if cursor < definition_span.start:
            sections.append(
                _SourceSection(
                    kind=_SectionKind.GAP,
                    span=_StructuralSpan(cursor, definition_span.start),
                )
            )
        sections.append(
            _SourceSection(
                kind=_SectionKind.DEFINITION,
                span=definition_span,
            )
        )
        cursor = definition_span.end

    if cursor < len(source_map.text):
        sections.append(
            _SourceSection(
                kind=_SectionKind.GAP,
                span=_StructuralSpan(cursor, len(source_map.text)),
            )
        )

    return sections


def _span_with_attached_comments(
    span: _StructuralSpan,
    source_map: _PythonSourceMap,
) -> _StructuralSpan:
    """Extend a definition over its directly preceding comment block."""
    line_index = bisect_right(source_map._line_starts, span.start) - 1
    definition_line_start = source_map._line_starts[line_index]
    indentation = source_map.text[definition_line_start:span.start]
    if indentation.strip():
        message = "Definition span must begin after indentation only"
        raise ValueError(message)

    attached_start = span.start

    for candidate_index in range(line_index - 1, -1, -1):
        candidate_start = source_map._line_starts[candidate_index]
        if candidate_index + 1 < len(source_map._line_starts):
            candidate_end = source_map._line_starts[candidate_index + 1]
        else:
            candidate_end = len(source_map.text)
        candidate = source_map.text[candidate_start:candidate_end]
        candidate = candidate.removesuffix("\r\n")
        candidate = candidate.removesuffix("\r").removesuffix("\n")

        if not candidate.startswith(indentation + "#"):
            break
        stripped_candidate = candidate[len(indentation):]
        if candidate_index == 0 and stripped_candidate.startswith("#!"):
            break
        if candidate_index <= 1 and re.match(
            r"^#.*?coding[:=][ \t]*[-\w.]+",
            stripped_candidate,
        ):
            break
        attached_start = candidate_start

    return _StructuralSpan(start=attached_start, end=span.end)
