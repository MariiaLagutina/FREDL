"""Partition plain text sources into exact paragraph ranges."""

from src.ingestion.chunking.text.markdown import _source_lines
from src.ingestion.chunking.text.models import _BlockKind, _TextBlock


def _plain_text_blocks(text: str) -> list[_TextBlock]:
    """Return alternating paragraph and whitespace blocks."""
    lines = _source_lines(text)
    blocks: list[_TextBlock] = []
    index = 0

    while index < len(lines):
        start = lines[index][0]
        is_whitespace = not lines[index][2].strip()
        index += 1

        while (
            index < len(lines)
            and (not lines[index][2].strip()) is is_whitespace
        ):
            index += 1

        blocks.append(
            _TextBlock(
                kind=(
                    _BlockKind.WHITESPACE
                    if is_whitespace
                    else _BlockKind.PARAGRAPH
                ),
                start=start,
                end=lines[index - 1][1],
                section_path=(),
            )
        )

    return blocks
