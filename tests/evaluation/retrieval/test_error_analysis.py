"""Tests for collecting objective retrieval miss evidence."""

from src.evaluation.retrieval import (
    RetrievalErrorCategory,
    RetrievalEvaluationCase,
    RetrievalMissEvidence,
    classify_ranked_miss,
    classify_structural_miss,
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


def test_classifies_relevant_source_below_top_five() -> None:
    evidence = collect_top_five_misses([_case("below", 8)])[0]

    analysis = classify_ranked_miss(evidence)

    assert analysis is not None
    assert analysis.category is RetrievalErrorCategory.RELEVANT_BELOW_TOP_5
    assert analysis.relevant_rank == 8
    assert analysis.hypothesis
    assert analysis.proposed_fix
    assert analysis.next_test


def test_leaves_absent_relevant_source_for_content_review() -> None:
    evidence = collect_top_five_misses([_case("absent", None)])[0]

    assert classify_ranked_miss(evidence) is None


def test_classifies_three_chunks_from_one_file_as_duplicate_results() -> None:
    reference = _source("docs/reference.md", 100, 200)
    repeated = tuple(
        _source("docs/noise.md", start, start + 20)
        for start in (0, 10, 40)
    )
    evidence = RetrievalMissEvidence(
        question_id="duplicates",
        question="Where is the setting documented?",
        references=(reference,),
        retrieved=repeated + (_source("docs/other.md", 0, 20),),
        relevant_rank=None,
    )

    analysis = classify_structural_miss(evidence)

    assert analysis is not None
    assert analysis.category is RetrievalErrorCategory.DUPLICATE_RESULTS


def test_does_not_treat_distinct_same_file_chunks_as_duplicates() -> None:
    reference = _source("docs/reference.md", 100, 200)
    evidence = RetrievalMissEvidence(
        question_id="same-file",
        question="Where is the setting documented?",
        references=(reference,),
        retrieved=(
            _source("docs/noise.md", 0, 20),
            _source("docs/noise.md", 30, 50),
            _source("docs/noise.md", 60, 80),
        ),
        relevant_rank=None,
    )

    assert classify_structural_miss(evidence) is None


def test_classifies_adjacent_range_as_chunk_boundary() -> None:
    reference = _source("docs/reference.md", 100, 200)
    evidence = RetrievalMissEvidence(
        question_id="boundary",
        question="Where is the setting documented?",
        references=(reference,),
        retrieved=(_source("docs/reference.md", 20, 98),),
        relevant_rank=None,
    )

    analysis = classify_structural_miss(evidence)

    assert analysis is not None
    assert analysis.category is RetrievalErrorCategory.CHUNK_BOUNDARY


def test_leaves_distant_same_file_chunk_for_content_review() -> None:
    reference = _source("docs/reference.md", 1000, 1100)
    evidence = RetrievalMissEvidence(
        question_id="distant",
        question="Where is the setting documented?",
        references=(reference,),
        retrieved=(_source("docs/reference.md", 0, 100),),
        relevant_rank=None,
    )

    assert classify_structural_miss(evidence) is None
