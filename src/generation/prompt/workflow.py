"""Connect bounded context, prompt construction, and local generation."""

from collections.abc import Callable

from src.generation.backend import (
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
from src.generation.prompt.validation import validate_grounded_answer


GeneratedAnswerObserver = Callable[[str], None]


def generate_grounded_answer(
    question: str,
    context: ContextBuildResult,
    backend: LoadedGenerationBackend,
    config: GenerationConfig,
    *,
    answer_observer: GeneratedAnswerObserver | None = None,
) -> GroundedAnswerResult:
    """Generate one answer and preserve its exact grounding trace."""
    messages = build_grounded_messages(question, context)
    answer = generate_answer(messages, backend, config)
    _observe_answer(answer_observer, answer)
    validate_grounded_answer(answer, context.sources)
    return GroundedAnswerResult(
        answer=answer,
        sources=context.sources,
        prompt_version=GROUNDING_PROMPT_VERSION,
    )


def _observe_answer(
    observer: GeneratedAnswerObserver | None,
    answer: str,
) -> None:
    """Expose raw generation text without changing validation behavior."""
    if observer is not None:
        observer(answer)
