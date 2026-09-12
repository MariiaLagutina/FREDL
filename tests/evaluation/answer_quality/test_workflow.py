"""Tests for failure-tolerant answer-quality diagnostics."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.evaluation.answer_quality import diagnose_dataset_answers
from src.generation import (
    GeneratedAnswerObserver,
    GenerationConfig,
    GroundedAnswerResult,
    GroundedAnswerValidationError,
    LoadedGenerationBackend,
)
from src.models import (
    MinimalSearchResults,
    MinimalSource,
    StudentSearchResults,
)


class FakeTokenizer:
    """Provide the token-counting surface required by the workflow."""

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> list[int]:
        """Count whitespace-separated tokens deterministically."""
        del add_special_tokens
        return list(range(len(text.split())))


def _search_results() -> StudentSearchResults:
    source_text = "The cache uses LRU."
    source = MinimalSource(
        file_path="data/raw/cache.md",
        first_character_index=0,
        last_character_index=len(source_text),
    )
    return StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="bad",
                question="Which policy fails validation?",
                retrieved_sources=[source],
            ),
            MinimalSearchResults(
                question_id="good",
                question="Which policy is supported?",
                retrieved_sources=[source],
            ),
        ],
        k=1,
    )


def test_diagnostic_keeps_attempts_and_continues_after_failure(
    tmp_path: Path,
) -> None:
    """One rejected answer cannot hide later review evidence."""
    corpus_root = tmp_path / "data" / "raw"
    corpus_root.mkdir(parents=True)
    (corpus_root / "cache.md").write_text(
        "The cache uses LRU.",
        encoding="utf-8",
    )
    progress_calls: list[tuple[int, int]] = []

    def generate(
        question: str,
        _context: object,
        _backend: object,
        _config: object,
        *,
        answer_observer: GeneratedAnswerObserver | None = None,
    ) -> GroundedAnswerResult:
        if question.startswith("Which policy fails"):
            if answer_observer is not None:
                answer_observer("The cache uses LRU.")
            raise GroundedAnswerValidationError(
                "Generated answer does not cite any retrieved source"
            )
        answer = "The cache uses LRU. [Source 1]"
        if answer_observer is not None:
            answer_observer(answer)
        return GroundedAnswerResult(
            answer=answer,
            sources=(
                _search_results().search_results[1].retrieved_sources[0],
            ),
            prompt_version="v1",
        )

    with patch(
        "src.evaluation.answer_quality.workflow.generate_grounded_answer",
        side_effect=generate,
    ) as generate_answer:
        report = diagnose_dataset_answers(
            tmp_path,
            corpus_root,
            _search_results(),
            LoadedGenerationBackend(FakeTokenizer(), object(), "cpu"),
            GenerationConfig(),
            context_token_budget=100,
            progress=lambda position, total: progress_calls.append(
                (position, total)
            ),
        )

    assert generate_answer.call_count == 2
    assert report.passed_count == 1
    assert report.failed_count == 1
    assert report.device == "cpu"
    assert [case.question_id for case in report.cases] == ["bad", "good"]
    assert [attempt.answer for attempt in report.cases[0].attempts] == [
        "The cache uses LRU.",
    ]
    assert report.cases[0].accepted_answer is None
    assert report.cases[0].error_type == "GroundedAnswerValidationError"
    assert report.cases[1].accepted_answer == (
        "The cache uses LRU. [Source 1]"
    )
    assert progress_calls == [(1, 2), (2, 2)]


def test_diagnostic_rejects_duplicate_ids_before_model_work() -> None:
    """Ambiguous report identities fail before loading source context."""
    results = _search_results()
    results.search_results[1].question_id = "bad"

    with patch(
        "src.evaluation.answer_quality.workflow.build_context"
    ) as build:
        with pytest.raises(
            ValueError,
            match="Search result question IDs must be unique",
        ):
            diagnose_dataset_answers(
                Path("."),
                Path("data/raw"),
                results,
                LoadedGenerationBackend(FakeTokenizer(), object(), "cpu"),
                GenerationConfig(),
                context_token_budget=100,
            )

    build.assert_not_called()


def test_diagnostic_continues_after_context_failure(tmp_path: Path) -> None:
    """A broken source is retained without suppressing later cases."""
    corpus_root = tmp_path / "data" / "raw"
    corpus_root.mkdir(parents=True)
    (corpus_root / "cache.md").write_text(
        "The cache uses LRU.",
        encoding="utf-8",
    )
    results = _search_results()
    results.search_results[0].retrieved_sources[0] = MinimalSource(
        file_path="data/raw/missing.md",
        first_character_index=0,
        last_character_index=1,
    )

    with patch(
        "src.evaluation.answer_quality.workflow.generate_grounded_answer",
        return_value=GroundedAnswerResult(
            answer="The cache uses LRU. [Source 1]",
            sources=tuple(results.search_results[1].retrieved_sources),
            prompt_version="v1",
        ),
    ) as generate_answer:
        report = diagnose_dataset_answers(
            tmp_path,
            corpus_root,
            results,
            LoadedGenerationBackend(FakeTokenizer(), object(), "cpu"),
            GenerationConfig(),
            context_token_budget=100,
        )

    assert generate_answer.call_count == 1
    assert report.failed_count == 1
    assert report.passed_count == 1
    assert report.cases[0].context_sources == ()
    assert report.cases[0].attempts == ()
    assert report.cases[0].error_type == "FileNotFoundError"
