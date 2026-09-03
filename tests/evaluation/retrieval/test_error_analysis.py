"""Tests for collecting objective retrieval miss evidence."""

from src.evaluation.retrieval import (
    RetrievalEvaluationCase,
    collect_top_five_misses,
)
from src.models import MinimalSource


def _source(file_path: str, start: int = 0, end: int = 20) -> MinimalSource:
    return MinimalSource(
        file_path=file_path,
        first_character_index=start,
        last_character_index=end,
    )


def _case(
    question_id: str,
    relevant_rank: int | None,
) -> RetrievalEvaluationCase:
    reference = _source(f"docs/{question_id}.md")
    retrieved = [_source(f"docs/noise-{rank}.md") for rank in range(1, 11)]
    if relevant_rank is not None:
        retrieved[relevant_rank - 1] = reference
    return RetrievalEvaluationCase(
        question_id=question_id,
        question=f"Question {question_id}",
        references=(reference,),
        retrieved=tuple(retrieved),
    )


def test_collects_only_top_five_misses_in_input_order() -> None:
    cases = [
        _case("hit", 5),
        _case("below", 6),
        _case("absent", None),
        _case("top", 1),
    ]

    misses = collect_top_five_misses(cases)

    assert [miss.question_id for miss in misses] == ["below", "absent"]
    assert [miss.relevant_rank for miss in misses] == [6, None]


def test_miss_evidence_preserves_sources_for_manual_classification() -> None:
    case = _case("below", 8)

    miss = collect_top_five_misses([case])[0]

    assert miss.question == case.question
    assert miss.references == case.references
    assert miss.retrieved == case.retrieved
