"""Generate assignment answers from already persisted retrieval results."""

from collections.abc import Callable
from pathlib import Path

from src.generation.backend import GenerationConfig, LoadedGenerationBackend
from src.generation.context import HuggingFaceTokenCounter, build_context
from src.generation.prompt import generate_grounded_answer
from src.models import (
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


BatchAnswerProgress = Callable[[int, int], None]


def generate_dataset_answers(
    project_root: Path,
    corpus_root: Path,
    search_results: StudentSearchResults,
    backend: LoadedGenerationBackend,
    generation_config: GenerationConfig,
    *,
    context_token_budget: int,
    progress: BatchAnswerProgress | None = None,
) -> StudentSearchResultsAndAnswer:
    """Generate answers in input order without repeating retrieval."""
    _validate_batch_input(search_results, context_token_budget)
    token_counter = HuggingFaceTokenCounter(backend.tokenizer)
    total = len(search_results.search_results)
    answers: list[MinimalAnswer] = []

    for position, search_result in enumerate(
        search_results.search_results,
        start=1,
    ):
        context = build_context(
            project_root,
            corpus_root,
            search_result.retrieved_sources,
            token_counter,
            context_token_budget,
        )
        grounded_answer = generate_grounded_answer(
            search_result.question,
            context,
            backend,
            generation_config,
        )
        answers.append(
            MinimalAnswer(
                question_id=search_result.question_id,
                question=search_result.question,
                retrieved_sources=search_result.retrieved_sources,
                answer=grounded_answer.answer,
            )
        )
        if progress is not None:
            progress(position, total)

    return StudentSearchResultsAndAnswer(
        search_results=answers,
        k=search_results.k,
    )


def _validate_batch_input(
    search_results: StudentSearchResults,
    context_token_budget: int,
) -> None:
    """Reject ambiguous batch values before source or model work."""
    if context_token_budget <= 0:
        raise ValueError("Context token budget must be greater than zero")
    if search_results.k <= 0:
        raise ValueError("Search result k must be greater than zero")

    question_ids = [
        result.question_id for result in search_results.search_results
    ]
    if any(not question_id.strip() for question_id in question_ids):
        raise ValueError("Search result question ID must not be empty")
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Search result question IDs must be unique")
    if any(
        not result.question.strip()
        for result in search_results.search_results
    ):
        raise ValueError("Search result question must not be empty")
