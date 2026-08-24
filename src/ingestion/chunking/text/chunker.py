"""Build bounded chunks from Markdown and plain text documents."""

from dataclasses import dataclass
from pathlib import PurePosixPath
import re

from src.ingestion.documents import Chunk, SourceDocument, make_chunk
from src.ingestion.files import FileKind
from src.ingestion.chunking.text.markdown import _markdown_blocks
from src.ingestion.chunking.text.models import _TextBlock
from src.ingestion.chunking.text.plain_text import _plain_text_blocks


MAX_CHUNK_SIZE = 2000
MAX_DEFAULT_OVERLAP_SIZE = 100


@dataclass(frozen=True, slots=True)
class _ChunkCandidate:
    """Store one exact candidate range before Chunk construction."""

    start: int
    end: int
    section_path: tuple[str, ...]


def chunk_text_document(
    document: SourceDocument,
    max_chunk_size: int = MAX_CHUNK_SIZE,
    overlap_size: int | None = None,
) -> list[Chunk]:
    """Split a Markdown or plain text document into bounded chunks."""
    if document.kind is not FileKind.TEXT:
        message = "Text chunker requires a text source document"
        raise ValueError(message)
    if max_chunk_size <= 0 or max_chunk_size > MAX_CHUNK_SIZE:
        message = "Maximum chunk size must be between 1 and 2000"
        raise ValueError(message)
    if overlap_size is None:
        overlap_size = min(
            MAX_DEFAULT_OVERLAP_SIZE,
            max_chunk_size // 10,
        )
    if overlap_size < 0 or overlap_size >= max_chunk_size:
        message = "Overlap size must be smaller than maximum chunk size"
        raise ValueError(message)
    if not document.text:
        return []

    suffix = PurePosixPath(document.file_path).suffix.lower()
    blocks = (
        _markdown_blocks(document.text)
        if suffix == ".md"
        else _plain_text_blocks(document.text)
    )
    candidates = _pack_blocks(
        document.text,
        blocks,
        max_chunk_size,
        overlap_size,
    )

    return [
        make_chunk(
            document,
            candidate.start,
            candidate.end,
            section_path=candidate.section_path,
        )
        for candidate in candidates
        if document.text[candidate.start:candidate.end].strip()
    ]


def _pack_blocks(
    text: str,
    blocks: list[_TextBlock],
    max_chunk_size: int,
    overlap_size: int,
) -> list[_ChunkCandidate]:
    """Pack adjacent blocks within one section and the size limit."""
    candidates: list[_ChunkCandidate] = []

    for block in blocks:
        if block.end - block.start > max_chunk_size:
            candidates.extend(
                _split_range(
                    text,
                    block.start,
                    block.end,
                    block.section_path,
                    max_chunk_size,
                    overlap_size,
                )
            )
        elif (
            candidates
            and candidates[-1].end == block.start
            and candidates[-1].section_path == block.section_path
            and block.end - candidates[-1].start <= max_chunk_size
        ):
            previous = candidates[-1]
            candidates[-1] = _ChunkCandidate(
                previous.start,
                block.end,
                previous.section_path,
            )
        else:
            candidates.append(
                _ChunkCandidate(
                    block.start,
                    block.end,
                    block.section_path,
                )
            )

    return candidates


def _split_range(
    text: str,
    start: int,
    end: int,
    section_path: tuple[str, ...],
    max_chunk_size: int,
    overlap_size: int,
) -> list[_ChunkCandidate]:
    """Split one oversized block at line and then character boundaries."""
    candidates: list[_ChunkCandidate] = []
    cursor = start

    while cursor < end:
        proposed_end = min(cursor + max_chunk_size, end)
        if proposed_end < end:
            if (
                proposed_end - 1 > cursor
                and text[proposed_end - 1:proposed_end + 1] == "\r\n"
            ):
                proposed_end -= 1
            line_end = _last_line_boundary(text, cursor, proposed_end)
            if line_end > cursor and text[cursor:line_end].strip():
                proposed_end = line_end
        candidates.append(_ChunkCandidate(cursor, proposed_end, section_path))
        if proposed_end == end or overlap_size == 0:
            cursor = proposed_end
        else:
            cursor = _overlap_start(
                text,
                cursor,
                proposed_end,
                overlap_size,
            )

    return candidates


def _last_line_boundary(text: str, start: int, end: int) -> int:
    """Return the final newline boundary inside an exact range."""
    boundary = start
    for match in re.finditer(r"\r\n|\r|\n", text[start:end]):
        boundary = start + match.end()
    return boundary


def _overlap_start(
    text: str,
    chunk_start: int,
    chunk_end: int,
    overlap_size: int,
) -> int:
    """Prefer a complete trailing source line as the next chunk overlap."""
    target = max(chunk_start + 1, chunk_end - overlap_size)
    line_start = chunk_start

    for match in re.finditer(r"\r\n|\r|\n", text[chunk_start:target]):
        candidate = chunk_start + match.end()
        if candidate < chunk_end:
            line_start = candidate

    complete_line_overlap = chunk_end - line_start
    if (
        line_start > chunk_start
        and complete_line_overlap <= overlap_size * 2
    ):
        return line_start
    return target
