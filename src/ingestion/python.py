"""Map Python AST positions to exact source character offsets."""

import ast
from dataclasses import dataclass, field
import re
from typing import cast


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
