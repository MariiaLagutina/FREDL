"""Tests for Markdown headings and hierarchical section paths."""

import pytest

from src.ingestion.text_blocks import (
    _heading_from_line,
    _MarkdownHeading,
    _section_path,
    _update_heading_stack,
)


@pytest.mark.parametrize(
    ("line", "level", "title"),
    [
        ("# Install\n", 1, "Install"),
        ("  ### Linux ###\r\n", 3, "Linux"),
        ("###### Details", 6, "Details"),
    ],
)
def test_heading_parser_accepts_atx_headings(
    line: str,
    level: int,
    title: str,
) -> None:
    """Valid ATX headings retain their complete source line range."""
    heading = _heading_from_line(line, start=10)

    assert heading == _MarkdownHeading(
        level=level,
        title=title,
        start=10,
        end=10 + len(line),
    )


@pytest.mark.parametrize(
    "line",
    [
        "#Missing space\n",
        "    # Too deeply indented\n",
        "####### Too many marks\n",
        "### ###\n",
        "Plain paragraph\n",
    ],
)
def test_heading_parser_rejects_non_headings(line: str) -> None:
    """Heading-like text outside ATX rules remains ordinary content."""
    assert _heading_from_line(line, start=0) is None


def test_heading_stack_replaces_siblings_and_descendants() -> None:
    """A new heading keeps only ancestors with a lower level number."""
    stack: tuple[_MarkdownHeading, ...] = ()

    for line in ("# Guide\n", "## Linux\n", "### GPU\n"):
        heading = _heading_from_line(line, start=0)
        assert heading is not None
        stack = _update_heading_stack(stack, heading)

    replacement = _heading_from_line("## macOS\n", start=0)
    assert replacement is not None
    stack = _update_heading_stack(stack, replacement)

    assert _section_path(stack) == ("Guide", "macOS")


def test_heading_stack_supports_skipped_levels() -> None:
    """A missing intermediate level does not create synthetic titles."""
    first = _heading_from_line("# Guide\n", start=0)
    third = _heading_from_line("### GPU\n", start=8)
    assert first is not None
    assert third is not None

    stack = _update_heading_stack((), first)
    stack = _update_heading_stack(stack, third)

    assert _section_path(stack) == ("Guide", "GPU")
