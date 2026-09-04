"""Corpus ingestion utilities."""

from src.ingestion.audit.models import (
    ChunkAuditIssue,
    ChunkAuditIssueKind,
    ChunkAuditReport,
    ChunkSizeSummary,
)
from src.ingestion.audit.runner import audit_corpus, audit_documents
from src.ingestion.chunking.orchestrator import chunk_document
from src.ingestion.documents import (
    Chunk,
    SourceDocument,
    make_chunk,
    read_document,
)
from src.ingestion.files import CorpusFile, FileKind, discover_files
from src.ingestion.chunking.python.chunker import chunk_python_document
from src.ingestion.chunking.python.identifiers import (
    extract_python_identifier_spans,
    PythonIdentifierSpan,
)
from src.ingestion.chunking.python.symbols import (
    extract_python_symbol_spans,
    PythonSymbolSpan,
)
from src.ingestion.chunking.text.chunker import chunk_text_document

__all__ = [
    "Chunk",
    "ChunkAuditIssue",
    "ChunkAuditIssueKind",
    "ChunkAuditReport",
    "ChunkSizeSummary",
    "CorpusFile",
    "FileKind",
    "PythonSymbolSpan",
    "PythonIdentifierSpan",
    "SourceDocument",
    "audit_corpus",
    "audit_documents",
    "chunk_document",
    "discover_files",
    "extract_python_symbol_spans",
    "extract_python_identifier_spans",
    "chunk_python_document",
    "chunk_text_document",
    "make_chunk",
    "read_document",
]
