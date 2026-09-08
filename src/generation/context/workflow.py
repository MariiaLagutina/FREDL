"""Connect retrieved source locations to bounded context construction."""

from collections.abc import Sequence
from pathlib import Path

from src.generation.context.builder import ContextBuilder
from src.generation.context.models import ContextBuildResult, TokenCounter
from src.generation.context.sources import load_ranked_source_texts
from src.models import MinimalSource


def build_context(
    project_root: Path,
    corpus_root: Path,
    ranked_sources: Sequence[MinimalSource],
    token_counter: TokenCounter,
    token_budget: int,
) -> ContextBuildResult:
    """Load retrieved files and construct their token-bounded context."""
    source_texts = load_ranked_source_texts(
        project_root,
        corpus_root,
        ranked_sources,
    )
    return ContextBuilder(source_texts, token_counter).build(
        ranked_sources,
        token_budget,
    )
