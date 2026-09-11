"""Connect bounded context, prompt construction, and local generation."""

from collections.abc import Callable

from src.generation.backend import (
    ChatMessage,
    GenerationConfig,
    LoadedGenerationBackend,
    generate_answer,
)
from src.generation.context import ContextBuildResult
from src.generation.prompt.models import GroundedAnswerResult
from src.generation.prompt.template import (
    GROUNDING_PROMPT_VERSION,
    build_grounded_messages,
)
from src.generation.prompt.validation import (
    GroundedAnswerValidationError,
    validate_grounded_answer,
)


CORRECTION_INSTRUCTION = (
    "Rewrite your complete answer because it violated the required output "
    "contract. Use only the original retrieved sources. Cite every factual "
    "sentence with valid [Source N] labels. Use 'The sources conflict:' only "
    "when at least two cited sources truly disagree. If the sources do not "
    "answer the question, return the exact insufficient-context response."
)


GenerationAttemptObserver = Callable[[int, str], None]


def generate_grounded_answer(
    question: str,
    context: ContextBuildResult,
    backend: LoadedGenerationBackend,
    config: GenerationConfig,
    *,
    attempt_observer: GenerationAttemptObserver | None = None,
) -> GroundedAnswerResult:
    """Generate one answer and preserve its exact grounding trace."""
    messages = build_grounded_messages(question, context)
    answer = generate_answer(messages, backend, config)
    generation_attempts = 1
    _observe_attempt(attempt_observer, generation_attempts, answer)
    try:
        validate_grounded_answer(answer, context.sources)
    except GroundedAnswerValidationError:
        correction_messages = [
            *messages,
            ChatMessage("assistant", answer),
            ChatMessage("user", CORRECTION_INSTRUCTION),
        ]
        answer = generate_answer(correction_messages, backend, config)
        generation_attempts = 2
        _observe_attempt(attempt_observer, generation_attempts, answer)
        validate_grounded_answer(answer, context.sources)
    return GroundedAnswerResult(
        answer=answer,
        sources=context.sources,
        prompt_version=GROUNDING_PROMPT_VERSION,
        generation_attempts=generation_attempts,
    )


def _observe_attempt(
    observer: GenerationAttemptObserver | None,
    attempt: int,
    answer: str,
) -> None:
    """Expose raw generation text without changing validation behavior."""
    if observer is not None:
        observer(attempt, answer)
