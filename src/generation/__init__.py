"""Grounded answer-generation components."""

from src.generation.backend import (
    ChatMessage,
    DEFAULT_MODEL_NAME,
    DevicePreference,
    GenerationBackendLoadError,
    GenerationConfig,
    GenerationError,
    GenerationRuntime,
    LoadedGenerationBackend,
    TransformersRuntime,
    generate_answer,
    load_generation_backend,
    select_device,
)
from src.generation.context import (
    ContextBuildResult,
    ContextBuilder,
    HuggingFaceTokenCounter,
    TokenCounter,
    build_context,
    load_ranked_source_texts,
)
from src.generation.prompt import (
    GROUNDING_PROMPT_VERSION,
    GroundedAnswerResult,
    INSUFFICIENT_CONTEXT_RESPONSE,
    build_grounded_messages,
    generate_grounded_answer,
)

__all__ = [
    "ChatMessage",
    "ContextBuildResult",
    "ContextBuilder",
    "DEFAULT_MODEL_NAME",
    "DevicePreference",
    "GenerationBackendLoadError",
    "GenerationConfig",
    "GenerationError",
    "GenerationRuntime",
    "GROUNDING_PROMPT_VERSION",
    "GroundedAnswerResult",
    "LoadedGenerationBackend",
    "HuggingFaceTokenCounter",
    "INSUFFICIENT_CONTEXT_RESPONSE",
    "TokenCounter",
    "TransformersRuntime",
    "build_context",
    "build_grounded_messages",
    "generate_answer",
    "generate_grounded_answer",
    "load_generation_backend",
    "load_ranked_source_texts",
    "select_device",
]
