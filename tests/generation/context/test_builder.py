"""Tests for deterministic token-budgeted context construction."""

import pytest

from src.generation import ContextBuilder
from src.models import MinimalSource


class CharacterTokenCounter:
    """Provide exact deterministic counts for context-builder tests."""

    def count_tokens(self, text: str) -> int:
        """Treat each character as one synthetic token."""
        return len(text)


def _source(path: str, start: int, end: int) -> MinimalSource:
    """Create one compact exact source fixture."""
    return MinimalSource(
        file_path=path,
        first_character_index=start,
        last_character_index=end,
    )


def test_context_preserves_rank_and_labels_exact_spans() -> None:
    """Included source text and labels follow retrieval ranking."""
    texts = {
        "docs/guide.md": "zero cache three",
        "src/cache.py": "cache = build()",
    }
    sources = [
        _source("docs/guide.md", 5, 10),
        _source("src/cache.py", 0, 15),
    ]

    result = ContextBuilder(texts, CharacterTokenCounter()).build(
        sources,
        token_budget=1000,
    )

    assert result.sources == tuple(sources)
    assert result.used_tokens == len(result.context)
    assert result.skipped_source_count == 0
    assert result.context == (
        "[Source 1]\n"
        "Path: docs/guide.md\n"
        "Range: [5, 10)\n\n"
        "cache\n\n---\n\n"
        "[Source 2]\n"
        "Path: src/cache.py\n"
        "Range: [0, 15)\n\n"
        "cache = build()"
    )


def test_context_deduplicates_exact_source_coordinates() -> None:
    """Repeated retrieval hits appear once without renumbering gaps."""
    source = _source("docs/guide.md", 0, 5)

    result = ContextBuilder(
        {"docs/guide.md": "cache"},
        CharacterTokenCounter(),
    ).build([source, source], token_budget=1000)

    assert result.sources == (source,)
    assert result.context.count("[Source 1]") == 1
    assert "[Source 2]" not in result.context
    assert result.skipped_source_count == 1


def test_context_skips_complete_block_that_exceeds_budget() -> None:
    """A large source is not truncated and a later small source may fit."""
    texts = {
        "docs/large.md": "x" * 100,
        "docs/small.md": "ok",
    }
    small = _source("docs/small.md", 0, 2)
    small_only = ContextBuilder(
        texts,
        CharacterTokenCounter(),
    ).build([small], token_budget=1000)

    result = ContextBuilder(texts, CharacterTokenCounter()).build(
        [_source("docs/large.md", 0, 100), small],
        token_budget=small_only.used_tokens,
    )

    assert result.context == small_only.context
    assert result.sources == (small,)
    assert result.used_tokens <= small_only.used_tokens
    assert result.skipped_source_count == 1


def test_context_rejects_non_positive_budget() -> None:
    """A caller must allocate a positive context budget explicitly."""
    builder = ContextBuilder({}, CharacterTokenCounter())

    with pytest.raises(ValueError, match="greater than zero"):
        builder.build([], token_budget=0)


@pytest.mark.parametrize(
    "source",
    [
        _source("missing.md", 0, 1),
        _source("docs/guide.md", 2, 6),
    ],
)
def test_context_rejects_unknown_or_invalid_source(
    source: MinimalSource,
) -> None:
    """Context never hides a stale path or an invalid coordinate range."""
    builder = ContextBuilder(
        {"docs/guide.md": "text"},
        CharacterTokenCounter(),
    )

    with pytest.raises(ValueError, match="context source"):
        builder.build([source], token_budget=1000)
