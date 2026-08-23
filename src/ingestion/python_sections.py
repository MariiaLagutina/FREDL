"""Partition Python modules into definitions and surrounding source text."""

import ast
from bisect import bisect_right
from collections.abc import Sequence
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
    STATEMENT = "statement"


@dataclass(frozen=True, slots=True)
class _SourceSection:
    """Classify one non-empty, exact range in a Python module."""

    kind: _SectionKind
    span: _StructuralSpan
    node: ast.AST | None = None


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
        (
            _span_with_attached_comments(
                _node_span(node, source_map),
                source_map,
            ),
            node,
        )
        for node in module.body
        if isinstance(node, definition_types)
    ]

    return _partition_span(
        _StructuralSpan(0, len(source_map.text)),
        definition_spans,
        _SectionKind.DEFINITION,
    ) if source_map.text else []


def _class_sections(
    node: ast.ClassDef,
    span: _StructuralSpan,
    source_map: _PythonSourceMap,
) -> list[_SourceSection]:
    """Partition one class span around its direct methods."""
    method_types = (ast.FunctionDef, ast.AsyncFunctionDef)
    method_spans = [
        (
            _span_with_attached_comments(
                _node_span(child, source_map),
                source_map,
            ),
            child,
        )
        for child in node.body
        if isinstance(child, method_types)
    ]
    return _partition_span(span, method_spans, _SectionKind.DEFINITION)


def _function_sections(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    span: _StructuralSpan,
    source_map: _PythonSourceMap,
) -> list[_SourceSection]:
    """Partition one function span around its direct body statements."""
    statement_spans = [
        (
            _span_with_attached_comments(
                _node_span(statement, source_map),
                source_map,
            ),
            statement,
        )
        for statement in node.body
    ]
    return _partition_span(span, statement_spans, _SectionKind.STATEMENT)


def _partition_span(
    container: _StructuralSpan,
    definitions: Sequence[tuple[_StructuralSpan, ast.AST]],
    structure_kind: _SectionKind,
) -> list[_SourceSection]:
    """Partition one container around ordered structural definitions."""

    sections: list[_SourceSection] = []
    cursor = container.start

    for definition_span, node in definitions:
        if definition_span.start < cursor:
            message = "Definition spans must not overlap"
            raise ValueError(message)
        if definition_span.end > container.end:
            message = "Definition span must stay inside its container"
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
                kind=structure_kind,
                span=definition_span,
                node=node,
            )
        )
        cursor = definition_span.end

    if cursor < container.end:
        sections.append(
            _SourceSection(
                kind=_SectionKind.GAP,
                span=_StructuralSpan(cursor, container.end),
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

    attached_start = definition_line_start

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
