"""Load and align labelled questions with persisted retrieval results."""

from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from src.evaluation.retrieval.models import RetrievalEvaluationCase
from src.models import AnsweredQuestion, RetrievalResults, UnansweredQuestion
from src.retrieval import load_rag_dataset


def load_evaluation_cases(
    ground_truth_path: Path,
    results_path: Path,
) -> tuple[RetrievalEvaluationCase, ...]:
    """Validate two files and align results in ground-truth order."""
    dataset = load_rag_dataset(ground_truth_path)
    try:
        results = RetrievalResults.model_validate_json(
            results_path.read_text(encoding="utf-8")
        )
    except ValidationError as error:
        raise ValueError("Retrieval results JSON is invalid") from error

    references = _answered_questions(dataset.rag_questions)
    _require_unique_ids(
        [question.question_id for question in references],
        "Ground-truth",
    )
    _require_unique_ids(
        [result.question_id for result in results.search_results],
        "Retrieval result",
    )
    results_by_id = {
        result.question_id: result for result in results.search_results
    }
    reference_ids = {question.question_id for question in references}
    if set(results_by_id) != reference_ids:
        raise ValueError(
            "Ground truth and retrieval results must contain the same IDs"
        )

    cases: list[RetrievalEvaluationCase] = []
    for reference in references:
        result = results_by_id[reference.question_id]
        if result.question != reference.question:
            raise ValueError(
                "Ground truth and retrieval question text must match"
            )
        cases.append(
            RetrievalEvaluationCase(
                question_id=reference.question_id,
                question=reference.question,
                references=tuple(reference.sources),
                retrieved=tuple(result.retrieved_sources),
            )
        )
    return tuple(cases)


def _answered_questions(
    questions: Sequence[UnansweredQuestion],
) -> list[AnsweredQuestion]:
    """Require relevance labels for every evaluation question."""
    if any(
        not isinstance(question, AnsweredQuestion)
        for question in questions
    ):
        raise ValueError("Ground truth must contain only answered questions")
    return [
        question
        for question in questions
        if isinstance(question, AnsweredQuestion)
    ]


def _require_unique_ids(question_ids: list[str], label: str) -> None:
    """Reject ambiguous joins before building evaluation cases."""
    if len(question_ids) != len(set(question_ids)):
        raise ValueError(f"{label} question IDs must be unique")
