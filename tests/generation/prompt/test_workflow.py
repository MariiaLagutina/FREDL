"""Tests for the internally traced grounded generation workflow."""

from unittest.mock import patch

import pytest

from src.generation import (
    GROUNDING_PROMPT_VERSION,
    ContextBuildResult,
    GenerationConfig,
    GroundedAnswerValidationError,
    LoadedGenerationBackend,
    generate_grounded_answer,
)
from src.models import MinimalSource


def _context() -> ContextBuildResult:
    """Build one labelled context with an inspectable source trace."""
    source = MinimalSource(
        file_path="docs/cache.md",
        first_character_index=10,
        last_character_index=40,
    )
    return ContextBuildResult(
        context=(
            "[Source 1] docs/cache.md:10-40\n"
            "The cache uses LRU eviction."
        ),
        sources=(source,),
        used_tokens=12,
        skipped_source_count=0,
    )


def test_workflow_generates_answer_and_preserves_grounding_trace() -> None:
    """The answer retains exact context sources and prompt version."""
    context = _context()
    backend = LoadedGenerationBackend(object(), object(), "cpu")
    config = GenerationConfig()

    with patch(
        "src.generation.prompt.workflow.generate_answer",
        return_value="The cache uses LRU eviction. [Source 1]",
    ) as generate:
        result = generate_grounded_answer(
            "Which eviction policy is used?",
            context,
            backend,
            config,
        )

    messages = generate.call_args.args[0]
    assert [message.role for message in messages] == ["system", "user"]
    assert "Which eviction policy is used?" in messages[1].content
    assert context.context in messages[1].content
    assert generate.call_args.args[1:] == (backend, config)
    assert result.answer == "The cache uses LRU eviction. [Source 1]"
    assert result.sources == context.sources
    assert result.prompt_version == GROUNDING_PROMPT_VERSION
    assert result.generation_attempts == 1


def test_workflow_rejects_empty_question_before_generation() -> None:
    """Invalid input cannot reach the expensive model boundary."""
    backend = LoadedGenerationBackend(object(), object(), "cpu")

    with patch(
        "src.generation.prompt.workflow.generate_answer"
    ) as generate:
        with pytest.raises(ValueError, match="question must not be empty"):
            generate_grounded_answer(
                "  ",
                _context(),
                backend,
                GenerationConfig(),
            )

    generate.assert_not_called()


def test_workflow_corrects_one_invalid_generated_answer() -> None:
    """One invalid answer receives a bounded corrective generation pass."""
    backend = LoadedGenerationBackend(object(), object(), "cpu")
    observed: list[tuple[int, str]] = []

    with patch(
        "src.generation.prompt.workflow.generate_answer",
        side_effect=[
            "The cache uses LRU eviction.",
            "The cache uses LRU eviction. [Source 1]",
        ],
    ) as generate:
        result = generate_grounded_answer(
            "Which eviction policy is used?",
            _context(),
            backend,
            GenerationConfig(),
            attempt_observer=lambda attempt, answer: observed.append(
                (attempt, answer)
            ),
        )

    correction_messages = generate.call_args_list[1].args[0]
    assert [message.role for message in correction_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert correction_messages[-2].content == "The cache uses LRU eviction."
    assert "Rewrite your complete answer" in correction_messages[-1].content
    assert result.answer.endswith("[Source 1]")
    assert result.generation_attempts == 2
    assert observed == [
        (1, "The cache uses LRU eviction."),
        (2, "The cache uses LRU eviction. [Source 1]"),
    ]


def test_workflow_rejects_second_invalid_generated_answer() -> None:
    """The corrective policy stops after one retry."""
    backend = LoadedGenerationBackend(object(), object(), "cpu")
    observed: list[tuple[int, str]] = []

    with patch(
        "src.generation.prompt.workflow.generate_answer",
        side_effect=[
            "The cache uses LRU eviction.",
            "The cache uses LRU eviction.",
        ],
    ) as generate:
        with pytest.raises(
            GroundedAnswerValidationError,
            match="does not cite any retrieved source",
        ):
            generate_grounded_answer(
                "Which eviction policy is used?",
                _context(),
                backend,
                GenerationConfig(),
                attempt_observer=lambda attempt, answer: observed.append(
                    (attempt, answer)
                ),
            )

    assert generate.call_count == 2
    assert observed == [
        (1, "The cache uses LRU eviction."),
        (2, "The cache uses LRU eviction."),
    ]
