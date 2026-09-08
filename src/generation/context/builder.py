"""Construct deterministic grounded context within a token budget."""

from collections.abc import Mapping, Sequence

from src.generation.context.models import ContextBuildResult, TokenCounter
from src.models import MinimalSource


_SOURCE_SEPARATOR = "\n\n---\n\n"


class ContextBuilder:
    """Format ranked exact source spans without exceeding a token budget."""

    def __init__(
        self,
        source_texts: Mapping[str, str],
        token_counter: TokenCounter,
    ) -> None:
        """Store exact corpus text and the tokenizer-specific counter."""
        self._source_texts = source_texts
        self._token_counter = token_counter

    def build(
        self,
        ranked_sources: Sequence[MinimalSource],
        token_budget: int,
    ) -> ContextBuildResult:
        """Return complete unique source blocks that fit the budget."""
        if token_budget <= 0:
            raise ValueError("Context token budget must be greater than zero")

        blocks: list[str] = []
        included: list[MinimalSource] = []
        seen: set[tuple[str, int, int]] = set()
        skipped_count = 0
        used_tokens = self._count_tokens("")

        for source in ranked_sources:
            key = _source_key(source)
            if key in seen:
                skipped_count += 1
                continue
            seen.add(key)

            source_text = self._source_text(source)
            block = _format_source(len(included) + 1, source, source_text)
            candidate = _SOURCE_SEPARATOR.join([*blocks, block])
            candidate_tokens = self._count_tokens(candidate)
            if candidate_tokens > token_budget:
                skipped_count += 1
                continue

            blocks.append(block)
            included.append(source)
            used_tokens = candidate_tokens

        return ContextBuildResult(
            context=_SOURCE_SEPARATOR.join(blocks),
            sources=tuple(included),
            used_tokens=used_tokens,
            skipped_source_count=skipped_count,
        )

    def _source_text(self, source: MinimalSource) -> str:
        """Load one exact validated half-open span from the corpus mapping."""
        document = self._source_texts.get(source.file_path)
        if document is None:
            message = f"Unknown context source path: {source.file_path}"
            raise ValueError(message)

        start = source.first_character_index
        end = source.last_character_index
        if start < 0 or end <= start or end > len(document):
            raise ValueError(
                f"Invalid context source range [{start}, {end}) for "
                f"document length {len(document)}"
            )
        return document[start:end]

    def _count_tokens(self, text: str) -> int:
        """Reject a broken tokenizer adapter before trusting its budget."""
        count = self._token_counter.count_tokens(text)
        if count < 0:
            raise ValueError("Token counter must not return a negative count")
        return count


def _source_key(source: MinimalSource) -> tuple[str, int, int]:
    """Return the exact source identity used for stable deduplication."""
    return (
        source.file_path,
        source.first_character_index,
        source.last_character_index,
    )


def _format_source(
    number: int,
    source: MinimalSource,
    source_text: str,
) -> str:
    """Render one complete source with an explicit citation label."""
    return (
        f"[Source {number}]\n"
        f"Path: {source.file_path}\n"
        f"Range: [{source.first_character_index}, "
        f"{source.last_character_index})\n\n"
        f"{source_text}"
    )
