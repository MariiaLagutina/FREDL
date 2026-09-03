"""Tests for retrieval error-analysis models."""

import pytest

from src.evaluation.retrieval.error_models import (
    RetrievalErrorAnalysisReport,
    RetrievalErrorCategory,
    RetrievalMissAnalysis,
    RetrievalMissEvidence,
)
from src.evaluation.retrieval.models import RetrievalDatasetKind
from src.models import MinimalSource


def _miss(
    question_id: str = "docs-001",
    category: RetrievalErrorCategory = RetrievalErrorCategory.WRONG_FILE,
    relevant_rank: int | None = None,
) -> RetrievalMissAnalysis:
    return RetrievalMissAnalysis(
        question_id=question_id,
        question="Where is the cache configured?",
        category=category,
        relevant_rank=relevant_rank,
        hypothesis="The query shares terms with a different file.",
        proposed_fix="Add a path-aware scoring signal.",
        next_test="Compare path-aware ranking on the same misses.",
    )


def test_error_categories_have_stable_serialized_values() -> None:
    assert [category.value for category in RetrievalErrorCategory] == [
        "wrong_file",
        "relevant_below_top_5",
        "chunk_boundary",
        "lost_identifier",
        "paraphrase",
        "duplicate_results",
        "noisy_metadata",
    ]


@pytest.mark.parametrize(
    "field_name",
    ["question_id", "question", "hypothesis", "proposed_fix", "next_test"],
)
def test_miss_rejects_empty_actionable_text(field_name: str) -> None:
    values = {
        "question_id": "docs-001",
        "question": "Where is the cache configured?",
        "category": RetrievalErrorCategory.WRONG_FILE,
        "relevant_rank": None,
        "hypothesis": "The query shares terms with a different file.",
        "proposed_fix": "Add a path-aware scoring signal.",
        "next_test": "Compare path-aware ranking on the same misses.",
    }
    values[field_name] = "  "

    with pytest.raises(ValueError, match="must not be empty"):
        RetrievalMissAnalysis(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("relevant_rank", [0, 1, 5])
def test_miss_rejects_relevant_rank_inside_top_five(
    relevant_rank: int,
) -> None:
    with pytest.raises(ValueError, match="greater than five"):
        _miss(relevant_rank=relevant_rank)


def test_report_rejects_duplicate_question_ids() -> None:
    with pytest.raises(ValueError, match="question IDs must be unique"):
        RetrievalErrorAnalysisReport(
            dataset=RetrievalDatasetKind.DOCS,
            misses=(_miss(), _miss()),
        )


def test_report_counts_categories_and_selects_dominant_category() -> None:
    report = RetrievalErrorAnalysisReport(
        dataset=RetrievalDatasetKind.CODE,
        misses=(
            _miss("code-001", RetrievalErrorCategory.PARAPHRASE),
            _miss("code-002", RetrievalErrorCategory.LOST_IDENTIFIER),
            _miss("code-003", RetrievalErrorCategory.PARAPHRASE),
        ),
    )

    assert dict(report.category_counts) == {
        RetrievalErrorCategory.WRONG_FILE: 0,
        RetrievalErrorCategory.RELEVANT_BELOW_TOP_5: 0,
        RetrievalErrorCategory.CHUNK_BOUNDARY: 0,
        RetrievalErrorCategory.LOST_IDENTIFIER: 1,
        RetrievalErrorCategory.PARAPHRASE: 2,
        RetrievalErrorCategory.DUPLICATE_RESULTS: 0,
        RetrievalErrorCategory.NOISY_METADATA: 0,
    }
    assert report.dominant_category is RetrievalErrorCategory.PARAPHRASE


def test_empty_report_has_no_dominant_category() -> None:
    report = RetrievalErrorAnalysisReport(
        dataset=RetrievalDatasetKind.DOCS,
        misses=(),
    )

    assert report.dominant_category is None


def test_miss_evidence_rejects_rank_outside_retrieved_sources() -> None:
    source = MinimalSource(
        file_path="docs/cache.md",
        first_character_index=0,
        last_character_index=20,
    )

    with pytest.raises(ValueError, match="refer to a retrieved source"):
        RetrievalMissEvidence(
            question_id="docs-001",
            question="Where is the cache configured?",
            references=(source,),
            retrieved=(source,) * 6,
            relevant_rank=7,
        )
