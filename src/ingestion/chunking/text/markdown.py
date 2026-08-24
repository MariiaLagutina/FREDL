"""Partition Markdown sources into exact structural blocks."""

import re

from src.ingestion.chunking.text.blocks import (
    _heading_from_line,
    _MarkdownHeading,
    _opening_fence_from_line,
    _is_closing_fence,
    _section_path,
    _update_heading_stack,
)
from src.ingestion.chunking.text.models import _BlockKind, _TextBlock


_LIST_ITEM = re.compile(r"^ {0,3}(?:[-+*]|\d+[.)])[ \t]+")


def _markdown_blocks(text: str) -> list[_TextBlock]:
    """Partition a Markdown document without changing any source text."""
    lines = _source_lines(text)
    blocks: list[_TextBlock] = []
    headings: tuple[_MarkdownHeading, ...] = ()
    index = 0

    while index < len(lines):
        start, end, line = lines[index]

        if not line.strip():
            index = _consume_whitespace(lines, index)
            blocks.append(
                _TextBlock(
                    _BlockKind.WHITESPACE,
                    start,
                    lines[index - 1][1],
                    _section_path(headings),
                )
            )
            continue

        fence = _opening_fence_from_line(line, start)
        if fence is not None:
            index += 1
            while index < len(lines):
                _, closing_end, candidate = lines[index]
                index += 1
                end = closing_end
                if _is_closing_fence(candidate, fence):
                    break
            blocks.append(
                _TextBlock(
                    _BlockKind.FENCED_CODE,
                    start,
                    end,
                    _section_path(headings),
                )
            )
            continue

        heading = _heading_from_line(line, start)
        if heading is not None:
            headings = _update_heading_stack(headings, heading)
            blocks.append(
                _TextBlock(
                    _BlockKind.HEADING,
                    start,
                    end,
                    _section_path(headings),
                )
            )
            index += 1
            continue

        kind = (
            _BlockKind.LIST
            if _LIST_ITEM.match(line)
            else _BlockKind.PARAGRAPH
        )
        index += 1
        while index < len(lines) and not _starts_new_block(
            lines[index],
            allow_list=kind is _BlockKind.LIST,
        ):
            end = lines[index][1]
            index += 1
        blocks.append(
            _TextBlock(
                kind,
                start,
                end,
                _section_path(headings),
            )
        )

    return blocks


def _source_lines(text: str) -> list[tuple[int, int, str]]:
    """Return exact lines using only CRLF, CR, and LF as boundaries."""
    lines: list[tuple[int, int, str]] = []
    cursor = 0
    for match in re.finditer(r"\r\n|\r|\n", text):
        end = match.end()
        lines.append((cursor, end, text[cursor:end]))
        cursor = end
    if cursor < len(text):
        lines.append((cursor, len(text), text[cursor:]))
    return lines


def _consume_whitespace(
    lines: list[tuple[int, int, str]],
    index: int,
) -> int:
    """Return the first line index after consecutive blank lines."""
    while index < len(lines) and not lines[index][2].strip():
        index += 1
    return index


def _starts_new_block(
    line: tuple[int, int, str],
    *,
    allow_list: bool,
) -> bool:
    """Return whether a source line starts another structural block."""
    start, _, text = line
    return (
        not text.strip()
        or _opening_fence_from_line(text, start) is not None
        or _heading_from_line(text, start) is not None
        or (not allow_list and _LIST_ITEM.match(text) is not None)
    )
