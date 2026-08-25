"""Prepare exact source chunks for two-field BM25 retrieval."""

from collections.abc import Sequence

from src.ingestion import (
    Chunk,
    extract_python_symbol_spans,
    FileKind,
    PythonSymbolSpan,
    SourceDocument,
)
from src.retrieval.bm25.models import BM25Document
from src.retrieval.tokenization import CodeTokenizer, TextTokenizer


def build_bm25_documents(
    document: SourceDocument,
    chunks: Sequence[Chunk],
) -> list[BM25Document]:
    """Create searchable documents without changing exact chunk evidence."""
    symbols = (
        extract_python_symbol_spans(document)
        if document.kind is FileKind.PYTHON
        else ()
    )
    content_tokenizer = (
        CodeTokenizer()
        if document.kind is FileKind.PYTHON
        else TextTokenizer()
    )

    documents: list[BM25Document] = []
    for chunk in chunks:
        _validate_chunk(document, chunk)
        content_terms = tuple(content_tokenizer.tokenize(chunk.text))
        if not content_terms:
            continue
        documents.append(
            BM25Document(
                chunk=chunk,
                content_terms=content_terms,
                metadata_terms=_metadata_terms(chunk, symbols),
            )
        )
    return documents


def _metadata_terms(
    chunk: Chunk,
    symbols: Sequence[PythonSymbolSpan],
) -> tuple[str, ...]:
    """Return stable unique path, section, and symbol terms."""
    values = [chunk.file_path, *chunk.section_path]
    for symbol in symbols:
        if symbol.start < chunk.end and chunk.start < symbol.end:
            values.append(symbol.qualified_name)
            values.append(symbol.qualified_name.rsplit(".", 1)[-1])

    terms: list[str] = []
    tokenizer = CodeTokenizer()
    for value in values:
        for term in tokenizer.tokenize(value):
            if term not in terms:
                terms.append(term)
    return tuple(terms)


def _validate_chunk(document: SourceDocument, chunk: Chunk) -> None:
    """Require every retrieval chunk to be exact evidence from its document."""
    if chunk.file_path != document.file_path:
        raise ValueError("Chunk and document paths must match")
    if chunk.end > len(document.text):
        raise ValueError("Chunk must stay inside its source document")
    if document.text[chunk.start:chunk.end] != chunk.text:
        raise ValueError("Chunk text must match its exact document slice")
