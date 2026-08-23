"""Build exact source chunks from Python documents."""

import ast
import re

from src.ingestion.documents import Chunk, SourceDocument, make_chunk
from src.ingestion.files import FileKind
from src.ingestion.python_positions import (
    _PythonSourceMap,
    _StructuralSpan,
)
from src.ingestion.python_sections import (
    _class_sections,
    _top_level_sections,
)


MAX_CHUNK_SIZE = 2000


def chunk_python_document(
    document: SourceDocument,
    max_chunk_size: int = MAX_CHUNK_SIZE,
) -> list[Chunk]:
    """Split a Python document into bounded chunks with exact source text."""
    if document.kind is not FileKind.PYTHON:
        message = "Python chunker requires a Python source document"
        raise ValueError(message)
    if max_chunk_size <= 0 or max_chunk_size > MAX_CHUNK_SIZE:
        message = "Maximum chunk size must be between 1 and 2000"
        raise ValueError(message)
    if not document.text:
        return []

    section_spans: list[_StructuralSpan] = []
    try:
        module = ast.parse(document.text)
    except SyntaxError:
        section_spans.append(_StructuralSpan(0, len(document.text)))
    else:
        source_map = _PythonSourceMap(document.text)
        for section in _top_level_sections(module, source_map):
            if (
                isinstance(section.node, ast.ClassDef)
                and section.span.end - section.span.start > max_chunk_size
            ):
                section_spans.extend(
                    child.span
                    for child in _class_sections(
                        section.node,
                        section.span,
                        source_map,
                    )
                )
            else:
                section_spans.append(section.span)

    chunks: list[Chunk] = []
    for section_span in section_spans:
        for span in _split_span(document.text, section_span, max_chunk_size):
            if document.text[span.start:span.end].strip():
                chunks.append(
                    make_chunk(document, start=span.start, end=span.end)
                )
    return chunks


def _split_span(
    text: str,
    span: _StructuralSpan,
    max_chunk_size: int,
) -> list[_StructuralSpan]:
    """Split a source range at line boundaries, then character limits."""
    if span.end - span.start <= max_chunk_size:
        return [span]

    spans: list[_StructuralSpan] = []
    cursor = span.start

    while cursor < span.end:
        proposed_end = min(cursor + max_chunk_size, span.end)
        if proposed_end < span.end:
            if text[proposed_end - 1:proposed_end + 1] == "\r\n":
                proposed_end -= 1
            line_end = _last_line_boundary(text, cursor, proposed_end)
            if (
                line_end > cursor
                and text[cursor:line_end].strip()
            ):
                proposed_end = line_end
        spans.append(_StructuralSpan(cursor, proposed_end))
        cursor = proposed_end

    return spans


def _last_line_boundary(text: str, start: int, end: int) -> int:
    """Return the last complete newline boundary inside a source range."""
    boundary = start
    for match in re.finditer(r"\r\n|\r|\n", text[start:end]):
        boundary = start + match.end()
    return boundary
