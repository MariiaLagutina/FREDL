"""Select the chunking strategy required by one source document."""

from src.ingestion.documents import Chunk, SourceDocument
from src.ingestion.files import FileKind
from src.ingestion.python_chunks import chunk_python_document
from src.ingestion.text_chunks import chunk_text_document


MAX_CHUNK_SIZE = 2000


def chunk_document(
    document: SourceDocument,
    max_chunk_size: int = MAX_CHUNK_SIZE,
) -> list[Chunk]:
    """Dispatch one source document to its matching chunking strategy."""
    if document.kind is FileKind.PYTHON:
        return chunk_python_document(document, max_chunk_size)
    if document.kind is FileKind.TEXT:
        return chunk_text_document(document, max_chunk_size)

    message = f"Unsupported file kind: {document.kind}"
    raise ValueError(message)
