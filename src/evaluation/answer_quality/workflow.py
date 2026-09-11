"""Run grounded generation while preserving per-question failures."""

from pathlib import Path

from src.evaluation.answer_quality.models import (
    AnswerQualityCaseEvidence,
    AnswerQualityDiagnosticReport,
    GenerationAttemptEvidence,
)
from src.generation import (
    BatchAnswerProgress,
    GROUNDING_PROMPT_VERSION,
    GenerationConfig,
    HuggingFaceTokenCounter,
    LoadedGenerationBackend,
    build_context,
    generate_grounded_answer,
)
from src.models import MinimalSource, StudentSearchResults


def diagnose_dataset_answers(
    project_root: Path,
    corpus_root: Path,
    search_results: StudentSearchResults,
    backend: LoadedGenerationBackend,
    generation_config: GenerationConfig,
    *,
    context_token_budget: int,
    progress: BatchAnswerProgress | None = None,
) -> AnswerQualityDiagnosticReport:
    """Collect raw attempts and continue after controlled case failures."""
    _validate_inputs(search_results, context_token_budget)
    token_counter = HuggingFaceTokenCounter(backend.tokenizer)
    total = len(search_results.search_results)
    cases: list[AnswerQualityCaseEvidence] = []

    for position, search_result in enumerate(
        search_results.search_results,
        start=1,
    ):
        attempts: list[GenerationAttemptEvidence] = []
        context_sources: tuple[MinimalSource, ...] = ()
        used_context_tokens = 0
        skipped_source_count = 0

        def observe_attempt(attempt: int, answer: str) -> None:
            attempts.append(GenerationAttemptEvidence(attempt, answer))

        try:
            context = build_context(
                project_root,
                corpus_root,
                search_result.retrieved_sources,
                token_counter,
                context_token_budget,
            )
            context_sources = context.sources
            used_context_tokens = context.used_tokens
            skipped_source_count = context.skipped_source_count
            result = generate_grounded_answer(
                search_result.question,
                context,
                backend,
                generation_config,
                attempt_observer=observe_attempt,
            )
        except (OSError, UnicodeError, ValueError, RuntimeError) as error:
            case = AnswerQualityCaseEvidence(
                question_id=search_result.question_id,
                question=search_result.question,
                retrieved_sources=tuple(search_result.retrieved_sources),
                context_sources=context_sources,
                used_context_tokens=used_context_tokens,
                skipped_source_count=skipped_source_count,
                attempts=tuple(attempts),
                accepted_answer=None,
                error_type=error.__class__.__name__,
                error=str(error) or error.__class__.__name__,
            )
        else:
            case = AnswerQualityCaseEvidence(
                question_id=search_result.question_id,
                question=search_result.question,
                retrieved_sources=tuple(search_result.retrieved_sources),
                context_sources=result.sources,
                used_context_tokens=used_context_tokens,
                skipped_source_count=skipped_source_count,
                attempts=tuple(attempts),
                accepted_answer=result.answer,
                error_type=None,
                error=None,
            )
        cases.append(case)
        if progress is not None:
            progress(position, total)

    return AnswerQualityDiagnosticReport(
        model_name=generation_config.model_name,
        device=backend.device,
        prompt_version=GROUNDING_PROMPT_VERSION,
        max_new_tokens=generation_config.max_new_tokens,
        context_token_budget=context_token_budget,
        search_k=search_results.k,
        cases=tuple(cases),
    )


def _validate_inputs(
    search_results: StudentSearchResults,
    context_token_budget: int,
) -> None:
    """Reject ambiguous diagnostics before source or model work."""
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
