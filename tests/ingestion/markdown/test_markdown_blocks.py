"""Tests for exact structural Markdown block partitioning."""

from src.ingestion.markdown_blocks import (
    _BlockKind,
    _markdown_blocks,
)


def test_markdown_blocks_preserve_structure_and_section_paths() -> None:
    """Headings, prose, lists, and code retain exact ordered ranges."""
    source = (
        "# Guide\n\n"
        "Read the *important* notes.\n\n"
        "- Install Python\n"
        "- Create an environment\n\n"
        "```python\n"
        "# Not a heading\n"
        "print('ready')\n"
        "```\n\n"
        "## Linux\n"
        "Use uv.\n"
    )

    blocks = _markdown_blocks(source)

    assert [block.kind for block in blocks] == [
        _BlockKind.HEADING,
        _BlockKind.WHITESPACE,
        _BlockKind.PARAGRAPH,
        _BlockKind.WHITESPACE,
        _BlockKind.LIST,
        _BlockKind.WHITESPACE,
        _BlockKind.FENCED_CODE,
        _BlockKind.WHITESPACE,
        _BlockKind.HEADING,
        _BlockKind.PARAGRAPH,
    ]
    assert "".join(source[block.start:block.end] for block in blocks) == source
    assert blocks[2].section_path == ("Guide",)
    assert blocks[6].section_path == ("Guide",)
    assert blocks[9].section_path == ("Guide", "Linux")


def test_unclosed_fence_becomes_one_complete_code_block() -> None:
    """An incomplete document keeps fenced content together through EOF."""
    source = "# Guide\n```python\nprint('unfinished')\n# Code\n"

    blocks = _markdown_blocks(source)

    assert [block.kind for block in blocks] == [
        _BlockKind.HEADING,
        _BlockKind.FENCED_CODE,
    ]
    assert source[blocks[1].start:blocks[1].end] == (
        "```python\nprint('unfinished')\n# Code\n"
    )


def test_plain_text_without_markdown_is_one_paragraph() -> None:
    """Consecutive plain-text lines stay in one paragraph block."""
    source = "First line.\r\nSecond line.\r\n"

    blocks = _markdown_blocks(source)

    assert len(blocks) == 1
    assert blocks[0].kind is _BlockKind.PARAGRAPH
    assert source[blocks[0].start:blocks[0].end] == source


def test_empty_document_has_no_blocks() -> None:
    """An empty source produces no artificial ranges."""
    assert _markdown_blocks("") == []
