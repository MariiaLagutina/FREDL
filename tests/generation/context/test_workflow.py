"""Tests for the retrieval-to-context filesystem workflow."""

from pathlib import Path

from src.generation.context import build_context
from src.models import MinimalSource


class CharacterTokenCounter:
    """Provide deterministic synthetic token counts."""

    def count_tokens(self, text: str) -> int:
        """Treat each character as one token."""
        return len(text)


def test_workflow_builds_ranked_context_from_exact_file_spans(
    tmp_path: Path,
) -> None:
    """The boundary and builder compose without changing source metadata."""
    corpus_root = tmp_path / "docs"
    corpus_root.mkdir()
    (corpus_root / "guide.md").write_text(
        "zero cache three",
        encoding="utf-8",
    )
    source = MinimalSource(
        file_path="docs/guide.md",
        first_character_index=5,
        last_character_index=10,
    )

    result = build_context(
        tmp_path,
        corpus_root,
        [source],
        CharacterTokenCounter(),
        token_budget=100,
    )

    assert result.sources == (source,)
    assert result.used_tokens == len(result.context)
    assert result.context.endswith("\n\ncache")
