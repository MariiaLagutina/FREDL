"""Configure local language-model generation backends."""

from src.generation.backend.config import (
    DEFAULT_MODEL_NAME,
    DevicePreference,
    GenerationConfig,
    select_device,
)
from src.generation.backend.generator import (
    ChatMessage,
    GenerationError,
    generate_answer,
)
from src.generation.backend.runtime import (
    GenerationBackendLoadError,
    GenerationRuntime,
    LoadedGenerationBackend,
    TransformersRuntime,
    load_generation_backend,
)

__all__ = [
    "ChatMessage",
    "DEFAULT_MODEL_NAME",
    "DevicePreference",
    "GenerationBackendLoadError",
    "GenerationConfig",
    "GenerationError",
    "GenerationRuntime",
    "LoadedGenerationBackend",
    "TransformersRuntime",
    "generate_answer",
    "load_generation_backend",
    "select_device",
]
