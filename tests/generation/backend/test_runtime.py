"""Tests for loading Qwen through the framework runtime boundary."""

from dataclasses import dataclass, field
from typing import Any

import pytest

from src.generation import (
    DevicePreference,
    GenerationBackendLoadError,
    GenerationConfig,
    load_generation_backend,
)


class FakeModel:
    """Record model placement and evaluation-mode activation."""

    def __init__(self) -> None:
        self.devices: list[str] = []
        self.eval_called = False

    def to(self, device: str) -> None:
        """Record the selected compute device."""
        self.devices.append(device)

    def eval(self) -> None:
        """Record deterministic evaluation-mode activation."""
        self.eval_called = True


@dataclass
class FakeRuntime:
    """Provide controllable lightweight framework behavior."""

    cuda_available: bool
    tokenizer: object = field(default_factory=object)
    model: FakeModel = field(default_factory=FakeModel)
    fail_with: Exception | None = None
    calls: list[tuple[str, str, bool, object | None]] = field(
        default_factory=list
    )

    def cuda_is_available(self) -> bool:
        """Return the configured availability result."""
        return self.cuda_available

    def dtype_for(self, device: str) -> str:
        """Return an inspectable fake dtype."""
        return f"{device}-dtype"

    def load_tokenizer(
        self,
        model_name: str,
        *,
        local_files_only: bool,
    ) -> object:
        """Record tokenizer loading or raise the configured failure."""
        self.calls.append(
            ("tokenizer", model_name, local_files_only, None)
        )
        if self.fail_with is not None:
            raise self.fail_with
        return self.tokenizer

    def load_model(
        self,
        model_name: str,
        *,
        local_files_only: bool,
        dtype: Any,
    ) -> FakeModel:
        """Record model loading or raise the configured failure."""
        self.calls.append(("model", model_name, local_files_only, dtype))
        if self.fail_with is not None:
            raise self.fail_with
        return self.model


@pytest.mark.parametrize(
    ("cuda_available", "expected_device"),
    [(True, "cuda"), (False, "cpu")],
)
def test_loader_places_model_on_resolved_device(
    cuda_available: bool,
    expected_device: str,
) -> None:
    """Automatic mode loads one pair and activates model evaluation mode."""
    runtime = FakeRuntime(cuda_available=cuda_available)

    loaded = load_generation_backend(GenerationConfig(), runtime)

    assert loaded.tokenizer is runtime.tokenizer
    assert loaded.model is runtime.model
    assert loaded.device == expected_device
    assert runtime.model.devices == [expected_device]
    assert runtime.model.eval_called
    assert runtime.calls == [
        ("tokenizer", "Qwen/Qwen3-0.6B", False, None),
        (
            "model",
            "Qwen/Qwen3-0.6B",
            False,
            f"{expected_device}-dtype",
        ),
    ]


def test_loader_passes_offline_mode_to_both_artifacts() -> None:
    """Offline defense never asks either loader to use the network."""
    runtime = FakeRuntime(cuda_available=False)

    load_generation_backend(
        GenerationConfig(
            device=DevicePreference.CPU,
            local_files_only=True,
        ),
        runtime,
    )

    assert [call[2] for call in runtime.calls] == [True, True]


def test_offline_cache_failure_has_actionable_message() -> None:
    """A missing cached checkpoint explains how to prepare the machine."""
    runtime = FakeRuntime(
        cuda_available=False,
        fail_with=OSError("cache miss"),
    )

    with pytest.raises(
        GenerationBackendLoadError,
        match="local Hugging Face cache.*network access",
    ):
        load_generation_backend(
            GenerationConfig(local_files_only=True),
            runtime,
        )


def test_online_load_failure_hides_framework_details() -> None:
    """Expected Hub failures become concise stable project errors."""
    runtime = FakeRuntime(
        cuda_available=False,
        fail_with=ValueError("framework detail"),
    )

    with pytest.raises(
        GenerationBackendLoadError,
        match="Qwen/Qwen3-0.6B.*network access",
    ):
        load_generation_backend(GenerationConfig(), runtime)
