"""Tests for the traceable single-query orchestration workflow."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.generation import (
    ContextBuildResult,
    GenerationConfig,
    GroundedAnswerResult,
    LoadedGenerationBackend,
    answer_query,
)
from src.models import MinimalSource
from src.retrieval.index_store import PipelineConfig


def _source(path: str, start: int, end: int) -> MinimalSource:
    return MinimalSource(
        file_path=path,
        first_character_index=start,
        last_character_index=end,
    )


def test_answer_query_composes_retrieval_context_and_generation() -> None:
    """The result distinguishes ranked hits from budgeted prompt sources."""
    retrieved = [
        _source("data/raw/guide.md", 0, 20),
        _source("data/raw/guide.md", 21, 50),
    ]
    context = ContextBuildResult(
        context="[Source 1] data/raw/guide.md:0-20\nCache uses LRU.",
        sources=(retrieved[0],),
        used_tokens=14,
        skipped_source_count=1,
    )
    grounded = GroundedAnswerResult(
        answer="The cache uses LRU. [Source 1]",
        sources=context.sources,
        prompt_version="v1",
        generation_attempts=2,
    )
    backend = LoadedGenerationBackend(object(), object(), "cpu")
    generation_config = GenerationConfig()
    pipeline_config = PipelineConfig()

    with (
        patch("src.generation.query.workflow.discover_files") as discover,
        patch(
            "src.generation.query.workflow.fingerprint_corpus",
            return_value="corpus-fingerprint",
        ) as corpus_fingerprint,
        patch(
            "src.generation.query.workflow.fingerprint_pipeline",
            return_value="pipeline-fingerprint",
        ) as pipeline_fingerprint,
        patch(
            "src.generation.query.workflow.run_stored_search",
            return_value=retrieved,
        ) as search,
        patch(
            "src.generation.query.workflow.build_context",
            return_value=context,
        ) as build,
        patch(
            "src.generation.query.workflow.generate_grounded_answer",
            return_value=grounded,
        ) as generate,
    ):
        result = answer_query(
            "Which cache policy is used?",
            Path("/project"),
            Path("data/raw"),
            Path("data/processed/index.json"),
            backend,
            generation_config,
            pipeline_config,
            k=2,
            context_token_budget=100,
        )

    discover.assert_called_once_with(
        Path("/project"), Path("/project/data/raw")
    )
    corpus_fingerprint.assert_called_once_with(
        Path("/project"), discover.return_value
    )
    pipeline_fingerprint.assert_called_once()
    search.assert_called_once_with(
        Path("/project/data/processed/index.json"),
        "corpus-fingerprint",
        "pipeline-fingerprint",
        "Which cache policy is used?",
        2,
        auxiliary_path_penalty=0.5,
        path_candidate_depth=20,
    )
    token_counter = build.call_args.args[3]
    assert token_counter.tokenizer is backend.tokenizer
    assert build.call_args.args[:3] == (
        Path("/project"),
        Path("/project/data/raw"),
        retrieved,
    )
    assert build.call_args.args[4] == 100
    generate.assert_called_once_with(
        "Which cache policy is used?",
        context,
        backend,
        generation_config,
    )
    assert result.answer == grounded.answer
    assert result.retrieved_sources == tuple(retrieved)
    assert result.context_sources == context.sources
    assert result.used_context_tokens == 14
    assert result.skipped_source_count == 1
    assert result.prompt_version == "v1"
    assert result.generation_attempts == 2


@pytest.mark.parametrize(
    ("question", "k", "token_budget", "message"),
    [
        ("  ", 5, 100, "Question must not be empty"),
        ("question", 0, 100, "Search k must be greater than zero"),
        (
            "question",
            5,
            0,
            "Context token budget must be greater than zero",
        ),
    ],
)
def test_answer_query_rejects_invalid_input_before_io(
    question: str,
    k: int,
    token_budget: int,
    message: str,
) -> None:
    """Cheap validation runs before corpus discovery or model generation."""
    with patch("src.generation.query.workflow.discover_files") as discover:
        with pytest.raises(ValueError, match=message):
            answer_query(
                question,
                Path("/project"),
                Path("data/raw"),
                Path("index.json"),
                LoadedGenerationBackend(object(), object(), "cpu"),
                GenerationConfig(),
                PipelineConfig(),
                k=k,
                context_token_budget=token_budget,
            )

    discover.assert_not_called()
