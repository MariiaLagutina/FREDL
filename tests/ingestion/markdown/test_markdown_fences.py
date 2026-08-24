"""Tests for fenced Markdown code block recognition."""

from src.ingestion.chunking.text.blocks import (
    _headings_outside_fences,
    _is_closing_fence,
    _opening_fence_from_line,
)


def test_backtick_fence_hides_code_headings() -> None:
    """Hash-prefixed code inside a backtick fence is not a heading."""
    source = (
        "# Guide\n\n"
        "```python\n"
        "# This is Python code.\n"
        "print('hello')\n"
        "```\n\n"
        "## Usage\n"
    )

    headings = _headings_outside_fences(source)

    assert [(heading.level, heading.title) for heading in headings] == [
        (1, "Guide"),
        (2, "Usage"),
    ]


def test_tilde_fence_hides_markdown_examples() -> None:
    """Tilde fences protect Markdown examples from structural parsing."""
    source = "~~~markdown\n# Example heading\n~~~\n# Real heading\n"

    headings = _headings_outside_fences(source)

    assert [heading.title for heading in headings] == ["Real heading"]


def test_shorter_marker_sequence_does_not_close_fence() -> None:
    """A closing fence must be at least as long as its opening fence."""
    source = (
        "````python\n"
        "```\n"
        "# Still code\n"
        "````\n"
        "# Outside\n"
    )

    headings = _headings_outside_fences(source)

    assert [heading.title for heading in headings] == ["Outside"]


def test_unclosed_fence_protects_the_rest_of_document() -> None:
    """An incomplete fence keeps all following lines as code content."""
    source = "# Before\n```text\n# Still code\n"

    headings = _headings_outside_fences(source)

    assert [heading.title for heading in headings] == ["Before"]


def test_fence_parsers_require_matching_marker_and_length() -> None:
    """Closing recognition uses the opening marker contract."""
    fence = _opening_fence_from_line("  ~~~~python\r\n", start=12)
    assert fence is not None

    assert _is_closing_fence(" ~~~~~\r\n", fence)
    assert not _is_closing_fence(" ````,\r\n", fence)
    assert not _is_closing_fence(" ~~~\r\n", fence)


def test_backtick_in_info_string_invalidates_opening() -> None:
    """A backtick fence info string cannot itself contain backticks."""
    assert _opening_fence_from_line("```py`thon\n", start=0) is None
