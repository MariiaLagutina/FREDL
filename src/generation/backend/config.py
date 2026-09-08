"""Validated configuration for deterministic local Qwen generation."""

from dataclasses import dataclass
from enum import Enum


DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"


class DevicePreference(str, Enum):
    """Describe how the generation backend selects its compute device."""

    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """Store reproducible model-loading and decoding choices."""

    model_name: str = DEFAULT_MODEL_NAME
    device: DevicePreference = DevicePreference.AUTO
    max_new_tokens: int = 256
    local_files_only: bool = False
    enable_thinking: bool = False
    do_sample: bool = False

    def __post_init__(self) -> None:
        """Reject ambiguous or unsupported generation settings."""
        if not self.model_name.strip():
            raise ValueError("Generation model name must not be empty")
        if self.max_new_tokens <= 0:
            raise ValueError("Maximum new tokens must be greater than zero")
        if self.do_sample:
            raise ValueError("Deterministic generation must not use sampling")


def select_device(
    preference: DevicePreference,
    *,
    cuda_available: bool,
) -> str:
    """Resolve automatic CPU fallback or validate an explicit CUDA request."""
    if preference is DevicePreference.CPU:
        return "cpu"
    if cuda_available:
        return "cuda"
    if preference is DevicePreference.CUDA:
        raise RuntimeError("CUDA was requested but is not available")
    return "cpu"
