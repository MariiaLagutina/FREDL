"""Tests for local evaluation file loading and question alignment."""

from pathlib import Path

import pytest

from src.evaluation.retrieval import load_evaluation_cases
from src.models import (
    AnsweredQuestion,
    MinimalSource,
    QuerySearchResult,
    RagDataset,
    RetrievalResults,
    UnansweredQuestion,
)


def _source(start: int = 0, end: int = 10) -> MinimalSource:
    """Create one labelled or retrieved source range."""
    return MinimalSource(
        file_path="data/raw/cache.py",
        first_character_index=start,
        last_character_index=end,
    )


def _answered(question_id: str, question: str) -> AnsweredQuestion:
    """Create one ground-truth question with a relevance label."""
    return AnsweredQuestion(
        question_id=question_id,
        question=question,
        answer="The cache is local.",
        sources=[_source()],
    )


def _result(question_id: str, question: str) -> QuerySearchResult:
    """Create one persisted retrieval result."""
    return QuerySearchResult(
        question_id=question_id,
        question=question,
        retrieved_sources=[_source()],
    )


def _write_files(
    tmp_path: Path,
    questions: list[AnsweredQuestion | UnansweredQuestion],
    results: list[QuerySearchResult],
) -> tuple[Path, Path]:
    """Write one validated ground-truth and results fixture pair."""
    ground_truth_path = tmp_path / "ground-truth.json"
    results_path = tmp_path / "results.json"
    ground_truth_path.write_text(
        RagDataset(rag_questions=questions).model_dump_json(),
        encoding="utf-8",
    )
    results_path.write_text(
        RetrievalResults(search_results=results, k=5).model_dump_json(),
        encoding="utf-8",
    )
    return ground_truth_path, results_path


def test_load_evaluation_cases_aligns_results_by_id(
    tmp_path: Path,
) -> None:
    """Results may arrive reordered but cases follow labelled order."""
    ground_truth_path, results_path = _write_files(
        tmp_path,
        [_answered("q-1", "First?"), _answered("q-2", "Second?")],
        [_result("q-2", "Second?"), _result("q-1", "First?")],
    )

    cases = load_evaluation_cases(ground_truth_path, results_path)

    assert [case.question_id for case in cases] == ["q-1", "q-2"]
    assert cases[0].references == (_source(),)
    assert cases[0].retrieved == (_source(),)


def test_load_evaluation_cases_requires_answered_ground_truth(
    tmp_path: Path,
) -> None:
    """Evaluation cannot calculate recall without relevance labels."""
    ground_truth_path, results_path = _write_files(
        tmp_path,
        [UnansweredQuestion(question_id="q-1", question="Cache?")],
        [_result("q-1", "Cache?")],
    )

    with pytest.raises(ValueError, match="only answered"):
        load_evaluation_cases(ground_truth_path, results_path)


def test_load_evaluation_cases_requires_identical_id_sets(
    tmp_path: Path,
) -> None:
    """Missing or unrelated results cannot be silently omitted."""
    ground_truth_path, results_path = _write_files(
        tmp_path,
        [_answered("q-1", "Cache?")],
        [_result("q-2", "Cache?")],
    )

    with pytest.raises(ValueError, match="same IDs"):
        load_evaluation_cases(ground_truth_path, results_path)


def test_load_evaluation_cases_requires_matching_question_text(
    tmp_path: Path,
) -> None:
    """An ID collision cannot join two different questions."""
    ground_truth_path, results_path = _write_files(
        tmp_path,
        [_answered("q-1", "Where is cache?")],
        [_result("q-1", "Where is scheduler?")],
    )

    with pytest.raises(ValueError, match="question text"):
        load_evaluation_cases(ground_truth_path, results_path)
