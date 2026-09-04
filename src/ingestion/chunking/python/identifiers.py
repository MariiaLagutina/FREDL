"""Extract identifiers used at structurally meaningful Python sites."""

import ast
from dataclasses import dataclass

from src.ingestion.chunking.python.positions import (
    _node_span,
    _PythonSourceMap,
)
from src.ingestion.documents import SourceDocument
from src.ingestion.files import FileKind


@dataclass(frozen=True, slots=True)
class PythonIdentifierSpan:
    """Store one structural identifier and its containing source range."""

    identifier: str
    start: int
    end: int

    def __post_init__(self) -> None:
        """Reject incomplete identifiers and invalid half-open ranges."""
        if not self.identifier:
            raise ValueError("Python identifier must not be empty")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Python identifier span must be a positive range")


def extract_python_identifier_spans(
    document: SourceDocument,
) -> tuple[PythonIdentifierSpan, ...]:
    """Return parameters, assignment targets, and keyword argument names."""
    if document.kind is not FileKind.PYTHON:
        raise ValueError("Python identifier extraction requires Python source")

    try:
        module = ast.parse(document.text)
    except SyntaxError:
        return ()

    source_map = _PythonSourceMap(document.text)
    identifiers: list[PythonIdentifierSpan] = []
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for argument in _function_arguments(node.args):
                _append_identifier(
                    identifiers,
                    argument.arg,
                    argument,
                    source_map,
                )
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else (node.target,)
            )
            for target in targets:
                target_identifiers = _target_identifiers(target)
                for identifier, target_node in target_identifiers:
                    _append_identifier(
                        identifiers,
                        identifier,
                        target_node,
                        source_map,
                    )
        elif isinstance(node, ast.NamedExpr):
            for identifier, target_node in _target_identifiers(node.target):
                _append_identifier(
                    identifiers,
                    identifier,
                    target_node,
                    source_map,
                )
        elif isinstance(node, ast.keyword) and node.arg is not None:
            _append_identifier(identifiers, node.arg, node, source_map)

    return tuple(identifiers)


def _function_arguments(arguments: ast.arguments) -> tuple[ast.arg, ...]:
    """Return every declared function parameter in source categories."""
    values = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
    if arguments.vararg is not None:
        values.append(arguments.vararg)
    if arguments.kwarg is not None:
        values.append(arguments.kwarg)
    return tuple(values)


def _target_identifiers(node: ast.AST) -> tuple[tuple[str, ast.AST], ...]:
    """Return searchable names represented by one assignment target."""
    if isinstance(node, ast.Name):
        return ((node.id, node),)
    if isinstance(node, ast.Attribute):
        return ((node.attr, node),)
    if isinstance(node, ast.Starred):
        return _target_identifiers(node.value)
    if isinstance(node, (ast.List, ast.Tuple)):
        return tuple(
            identifier
            for element in node.elts
            for identifier in _target_identifiers(element)
        )
    return ()


def _append_identifier(
    identifiers: list[PythonIdentifierSpan],
    identifier: str,
    node: ast.AST,
    source_map: _PythonSourceMap,
) -> None:
    """Append a structural identifier with its containing node range."""
    span = _node_span(node, source_map)
    identifiers.append(
        PythonIdentifierSpan(
            identifier=identifier,
            start=span.start,
            end=span.end,
        )
    )
