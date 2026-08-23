"""Map Python AST positions to exact source character offsets."""

import ast
from bisect import bisect_right
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import cast


class _SectionKind(str, Enum):
    """Identify whether a module section is structure or surrounding text."""

    GAP = "gap"
    DEFINITION = "definition"


@dataclass(frozen=True, slots=True)
class _StructuralSpan:
    """Store one exact half-open range selected from Python structure."""

    start: int
    end: int

    def __post_init__(self) -> None:
        """Reject ranges that cannot identify source text."""
        if self.start < 0:
            message = "Structural span start must not be negative"
            raise ValueError(message)
        if self.end <= self.start:
            message = "Structural span end must be greater than start"
            raise ValueError(message)


@dataclass(frozen=True, slots=True)
class _SourceSection:
    """Classify one non-empty, exact range in a Python module."""

    kind: _SectionKind
    span: _StructuralSpan


@dataclass(frozen=True, slots=True)
class _PythonSourceMap:
    """Translate one-based AST lines and UTF-8 columns into string indexes."""

    text: str
    _line_starts: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Precompute the absolute character index of every source line."""
        line_starts = [0]
        for match in re.finditer(r"\r\n|\r|\n", self.text):
            if match.end() < len(self.text):
                line_starts.append(match.end())
        object.__setattr__(self, "_line_starts", tuple(line_starts))

    def character_offset(self, line: int, utf8_column: int) -> int:
        """Return an absolute character index for one AST source position."""
        if line < 1 or line > len(self._line_starts):
            message = "AST line is outside the source text"
            raise ValueError(message)
        if utf8_column < 0:
            message = "AST UTF-8 column must not be negative"
            raise ValueError(message)

        line_start = self._line_starts[line - 1]
        if line < len(self._line_starts):
            line_end = self._line_starts[line]
        else:
            line_end = len(self.text)
        line_text = self.text[line_start:line_end]
        if line_text.endswith("\r\n"):
            line_text = line_text[:-2]
        elif line_text.endswith(("\r", "\n")):
            line_text = line_text[:-1]
        encoded_line = line_text.encode("utf-8")

        if utf8_column > len(encoded_line):
            message = "AST UTF-8 column is outside its source line"
            raise ValueError(message)

        encoded_prefix = encoded_line[:utf8_column]
        try:
            character_prefix = encoded_prefix.decode("utf-8")
        except UnicodeDecodeError as error:
            message = "AST UTF-8 column splits a source character"
            raise ValueError(message) from error

        return line_start + len(character_prefix)


def _node_span(node: ast.AST, source_map: _PythonSourceMap) -> _StructuralSpan:
    """Return the exact source range described by one positional AST node."""
    position_names = (
        "lineno",
        "col_offset",
        "end_lineno",
        "end_col_offset",
    )
    positions = [getattr(node, name, None) for name in position_names]
    if not all(isinstance(position, int) for position in positions):
        message = "AST node does not provide a complete source range"
        raise ValueError(message)

    line, column, end_line, end_column = cast(
        tuple[int, int, int, int],
        tuple(positions),
    )
    start = source_map.character_offset(line, column)
    end = source_map.character_offset(end_line, end_column)

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        if node.decorator_list:
            decorator = node.decorator_list[0]
            decorator_expression_start = source_map.character_offset(
                decorator.lineno,
                decorator.col_offset,
            )
            decorator_line_start = source_map.character_offset(
                decorator.lineno,
                0,
            )
            at_sign = source_map.text.rfind(
                "@",
                decorator_line_start,
                decorator_expression_start,
            )
            if at_sign < decorator_line_start:
                message = "Decorated AST node has no source @ marker"
                raise ValueError(message)
            start = at_sign

    return _StructuralSpan(start=start, end=end)


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
