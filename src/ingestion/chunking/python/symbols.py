"""Extract qualified Python symbol names with exact source spans."""

import ast
from dataclasses import dataclass

from src.ingestion.chunking.python.positions import (
    _node_span,
    _PythonSourceMap,
)
from src.ingestion.documents import SourceDocument
from src.ingestion.files import FileKind


_SYMBOL_NODE_TYPES = (
    ast.ClassDef,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
)


@dataclass(frozen=True, slots=True)
class PythonSymbolSpan:
    """Store one qualified Python symbol and its exact source range."""

    qualified_name: str
    start: int
    end: int

    def __post_init__(self) -> None:
        """Reject incomplete names and invalid half-open ranges."""
        if not self.qualified_name:
            raise ValueError("Python symbol name must not be empty")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Python symbol span must be a positive range")


def extract_python_symbol_spans(
    document: SourceDocument,
) -> tuple[PythonSymbolSpan, ...]:
    """Return qualified class and function names in source order."""
    if document.kind is not FileKind.PYTHON:
        raise ValueError("Python symbol extraction requires Python source")

    try:
        module = ast.parse(document.text)
    except SyntaxError:
        return ()

    symbols: list[PythonSymbolSpan] = []
    source_map = _PythonSourceMap(document.text)
    _collect_symbols(module, (), source_map, symbols)
    return tuple(symbols)


def _collect_symbols(
    node: ast.AST,
    parents: tuple[str, ...],
    source_map: _PythonSourceMap,
    symbols: list[PythonSymbolSpan],
) -> None:
    """Collect nested definitions while preserving their qualified paths."""
    child_parents = parents
    if isinstance(node, _SYMBOL_NODE_TYPES):
        child_parents = (*parents, node.name)
        span = _node_span(node, source_map)
        symbols.append(
            PythonSymbolSpan(
                qualified_name=".".join(child_parents),
                start=span.start,
                end=span.end,
            )
        )

    for child in ast.iter_child_nodes(node):
        _collect_symbols(child, child_parents, source_map, symbols)
