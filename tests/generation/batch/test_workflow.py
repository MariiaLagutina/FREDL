"""Tests for ordered batch generation from stored retrieval results."""

from pathlib import Path
from unittest.mock import call, patch

import pytest

from src.generation import (
    ContextBuildResult,
    GenerationConfig,
    GroundedAnswerResult,
    LoadedGenerationBackend,
    generate_dataset_answers,
)
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)


def _source(path: str, start: int, end: int) -> MinimalSource:
    return MinimalSource(
        file_path=path,
        first_character_index=start,
        last_character_index=end,
    )


def _search_results() -> StudentSearchResults:
    return StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="q-2",
                question="Second question?",
                retrieved_sources=[
                    _source("data/raw/second.md", 0, 20),
                    _source("data/raw/skipped.md", 0, 30),
                ],
            ),
            MinimalSearchResults(
                question_id="q-1",
                question="First question?",
                retrieved_sources=[
                    _source("data/raw/first.py", 10, 40)
                ],
            ),
        ],
        k=5,
    )


def test_batch_generates_in_order_and_preserves_retrieval_contract() -> None:
    """Prompt subsets do not alter assignment-facing retrieved sources."""
    search_results = _search_results()
    first_context = ContextBuildResult(
        context="[Source 1] data/raw/second.md:0-20\nSecond evidence.",
        sources=(search_results.search_results[0].retrieved_sources[0],),
        used_tokens=20,
        skipped_source_count=1,
    )
    second_context = ContextBuildResult(
        context="[Source 1] data/raw/first.py:10-40\nFirst evidence.",
        sources=tuple(search_results.search_results[1].retrieved_sources),
        used_tokens=18,
        skipped_source_count=0,
    )
    backend = LoadedGenerationBackend(object(), object(), "cpu")
    config = GenerationConfig()
    progress_calls: list[tuple[int, int]] = []

    with (
        patch(
            "src.generation.batch.workflow.build_context",
            side_effect=[first_context, second_context],
        ) as build,
        patch(
            "src.generation.batch.workflow.generate_grounded_answer",
            side_effect=[
                GroundedAnswerResult(
                    "Second answer. [Source 1]",
                    first_context.sources,
                    "v1",
                ),
                GroundedAnswerResult(
                    "First answer. [Source 1]",
                    second_context.sources,
                    "v1",
                ),
            ],
        ) as generate,
    ):
        result = generate_dataset_answers(
            Path("/project"),
            Path("/project/data/raw"),
            search_results,
            backend,
            config,
            context_token_budget=100,
            progress=lambda position, total: progress_calls.append(
                (position, total)
            ),
        )

    assert [answer.question_id for answer in result.search_results] == [
        "q-2",
        "q-1",
    ]
    assert result.search_results[0].retrieved_sources == (
        search_results.search_results[0].retrieved_sources
    )
    assert result.k == search_results.k
    token_counter = build.call_args_list[0].args[3]
    assert token_counter.tokenizer is backend.tokenizer
    assert build.call_args_list == [
        call(
            Path("/project"),
            Path("/project/data/raw"),
            search_results.search_results[0].retrieved_sources,
            token_counter,
            100,
        ),
        call(
            Path("/project"),
            Path("/project/data/raw"),
            search_results.search_results[1].retrieved_sources,
            token_counter,
            100,
        ),
    ]
    assert generate.call_args_list == [
        call("Second question?", first_context, backend, config),
        call("First question?", second_context, backend, config),
    ]
    assert progress_calls == [(1, 2), (2, 2)]


def test_empty_batch_preserves_k_without_generation() -> None:
    """An empty valid input returns an empty assignment output."""
    search_results = StudentSearchResults(search_results=[], k=5)

    with patch(
        "src.generation.batch.workflow.generate_grounded_answer"
    ) as generate:
        result = generate_dataset_answers(
            Path("/project"),
            Path("/project/data/raw"),
            search_results,
            LoadedGenerationBackend(object(), object(), "cpu"),
            GenerationConfig(),
            context_token_budget=100,
        )

    generate.assert_not_called()
    assert result.search_results == []
    assert result.k == 5


@pytest.mark.parametrize(
    ("search_results", "token_budget", "message"),
    [
        (
            StudentSearchResults(search_results=[], k=0),
            100,
            "Search result k must be greater than zero",
        ),
        (
            StudentSearchResults(search_results=[], k=5),
            0,
            "Context token budget must be greater than zero",
        ),
        (
            StudentSearchResults(
                search_results=[
                    MinimalSearchResults(
                        question_id="q-1",
                        question="First?",
                        retrieved_sources=[],
                    ),
                    MinimalSearchResults(
                        question_id="q-1",
                        question="Duplicate?",
                        retrieved_sources=[],
                    ),
                ],
                k=5,
            ),
            100,
            "question IDs must be unique",
        ),
    ],
)
def test_invalid_batch_fails_before_context_or_model_work(
    search_results: StudentSearchResults,
    token_budget: int,
    message: str,
) -> None:
    """Ambiguous batch input cannot produce a partial answer sequence."""
    with patch("src.generation.batch.workflow.build_context") as build:
        with pytest.raises(ValueError, match=message):
            generate_dataset_answers(
                Path("/project"),
                Path("/project/data/raw"),
                search_results,
                LoadedGenerationBackend(object(), object(), "cpu"),
                GenerationConfig(),
                context_token_budget=token_budget,
            )

    build.assert_not_called()
