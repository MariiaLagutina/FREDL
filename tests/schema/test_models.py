"""Tests for the RAG pipeline data models."""

from pydantic import ValidationError
import pytest

from src.models import (
    AnsweredQuestion,
    MinimalSource,
    QueryAnswer,
    QuerySearchResult,
    RagDataset,
    RetrievalResults,
    RetrievalResultsWithAnswers,
    UnansweredQuestion,
)


def make_source() -> MinimalSource:
    """Create a representative source location for tests."""
    return MinimalSource(
        file_path="data/raw/vllm-0.10.1/docs/example.md",
        first_character_index=10,
        last_character_index=42,
    )


def test_minimal_source_has_required_output_fields() -> None:
    """MinimalSource serializes to the exact required field names."""
    source = make_source()

    assert source.model_dump() == {
        "file_path": "data/raw/vllm-0.10.1/docs/example.md",
        "first_character_index": 10,
        "last_character_index": 42,
    }


def test_minimal_source_requires_every_coordinate() -> None:
    """MinimalSource rejects input with a missing character coordinate."""
    with pytest.raises(ValidationError):
        MinimalSource.model_validate(
            {
                "file_path": "data/raw/vllm-0.10.1/docs/example.md",
                "first_character_index": 10,
            }
        )


def test_unanswered_question_generates_unique_string_ids() -> None:
    """Questions receive distinct string identifiers when none are supplied."""
    first = UnansweredQuestion(question="First question?")
    second = UnansweredQuestion(question="Second question?")

    assert isinstance(first.question_id, str)
    assert first.question_id != second.question_id


def test_rag_dataset_parses_answered_and_unanswered_questions() -> None:
    """RagDataset preserves both supported question variants."""
    dataset = RagDataset.model_validate(
        {
            "rag_questions": [
                {"question_id": "q1", "question": "Not answered yet?"},
                {
                    "question_id": "q2",
                    "question": "Already answered?",
                    "sources": [make_source().model_dump()],
                    "answer": "Yes.",
                },
            ]
        }
    )

    assert isinstance(dataset.rag_questions[0], UnansweredQuestion)
    assert isinstance(dataset.rag_questions[1], AnsweredQuestion)


def test_student_search_results_round_trip() -> None:
    """Batch search output survives JSON serialization and validation."""
    result = RetrievalResults(
        search_results=[
            QuerySearchResult(
                question_id="q1",
                question="Where is the example?",
                retrieved_sources=[make_source()],
            )
        ],
        k=5,
    )

    restored = RetrievalResults.model_validate_json(
        result.model_dump_json()
    )

    assert restored == result


def test_student_answers_round_trip() -> None:
    """Batch answer output survives JSON serialization and validation."""
    result = RetrievalResultsWithAnswers(
        search_results=[
            QueryAnswer(
                question_id="q1",
                question="Where is the example?",
                retrieved_sources=[make_source()],
                answer="It is in the example documentation.",
            )
        ],
        k=5,
    )

    restored = RetrievalResultsWithAnswers.model_validate_json(
        result.model_dump_json()
    )

    assert restored == result
