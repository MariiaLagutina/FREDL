"""Configure local language-model generation backends."""

from src.generation.backend.config import (
    DEFAULT_MODEL_NAME,
    DevicePreference,
    GenerationConfig,
    select_device,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "DevicePreference",
    "GenerationConfig",
    "select_device",
]
