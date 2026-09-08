"""Tests for deterministic Qwen backend configuration."""

import pytest

from src.generation import (
    DEFAULT_MODEL_NAME,
    DevicePreference,
    GenerationConfig,
    select_device,
)


def test_generation_defaults_match_mandatory_qwen_model() -> None:
    """The assignment checkpoint is the stable production default."""
    config = GenerationConfig()

    assert config.model_name == "Qwen/Qwen3-0.6B"
    assert config.model_name == DEFAULT_MODEL_NAME
    assert config.device is DevicePreference.AUTO
    assert config.max_new_tokens == 256
    assert config.local_files_only is False
    assert config.enable_thinking is False
    assert config.do_sample is False


@pytest.mark.parametrize(
    ("preference", "cuda_available", "expected"),
    [
        (DevicePreference.AUTO, True, "cuda"),
        (DevicePreference.AUTO, False, "cpu"),
        (DevicePreference.CUDA, True, "cuda"),
        (DevicePreference.CPU, True, "cpu"),
        (DevicePreference.CPU, False, "cpu"),
    ],
)
def test_device_selection_is_explicit_and_predictable(
    preference: DevicePreference,
    cuda_available: bool,
    expected: str,
) -> None:
    """Automatic mode falls back while explicit CPU remains on CPU."""
    assert select_device(
        preference,
        cuda_available=cuda_available,
    ) == expected


def test_explicit_cuda_fails_when_unavailable() -> None:
    """An explicit CUDA request is not silently changed to CPU."""
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        select_device(
            DevicePreference.CUDA,
            cuda_available=False,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"model_name": "  "}, "model name"),
        ({"max_new_tokens": 0}, "greater than zero"),
        ({"do_sample": True}, "must not use sampling"),
    ],
)
def test_generation_config_rejects_unsupported_values(
    changes: dict[str, object],
    message: str,
) -> None:
    """Invalid settings fail before model loading begins."""
    with pytest.raises(ValueError, match=message):
        GenerationConfig(**changes)  # type: ignore[arg-type]
