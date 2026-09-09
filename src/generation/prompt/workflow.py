"""Connect bounded context, prompt construction, and local generation."""

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


def generate_grounded_answer(
    question: str,
    context: ContextBuildResult,
    backend: LoadedGenerationBackend,
    config: GenerationConfig,
) -> GroundedAnswerResult:
    """Generate one answer and preserve its exact grounding trace."""
    messages = build_grounded_messages(question, context)
    answer = generate_answer(messages, backend, config)
    return GroundedAnswerResult(
        answer=answer,
        sources=context.sources,
        prompt_version=GROUNDING_PROMPT_VERSION,
    )
