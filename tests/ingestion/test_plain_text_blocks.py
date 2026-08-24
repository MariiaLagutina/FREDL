"""Tests for exact paragraph blocks in plain text sources."""

from src.ingestion.plain_text_blocks import _plain_text_blocks
from src.ingestion.text_models import _BlockKind


def test_plain_text_blocks_alternate_paragraphs_and_whitespace() -> None:
    """Blank lines separate exact plain-text paragraph ranges."""
    source = "First paragraph.\r\nStill first.\r\n\r\nSecond paragraph."

    blocks = _plain_text_blocks(source)

    assert [block.kind for block in blocks] == [
        _BlockKind.PARAGRAPH,
        _BlockKind.WHITESPACE,
        _BlockKind.PARAGRAPH,
    ]
    assert "".join(source[block.start:block.end] for block in blocks) == source
    assert all(block.section_path == () for block in blocks)


def test_empty_plain_text_has_no_blocks() -> None:
    """An empty text source creates no artificial paragraph."""
    assert _plain_text_blocks("") == []
