"""Tests for portable semantic backend loading."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from src.retrieval.semantic import (
    SemanticBackendLoadError,
    SemanticEncoderConfig,
    load_semantic_backend,
)


class FakeModel:
    """Record placement and evaluation-mode activation."""

    def __init__(self, hidden_size: int = 384) -> None:
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.devices: list[str] = []
        self.eval_called = False

    def to(self, device: str) -> None:
        self.devices.append(device)

    def eval(self) -> None:
        self.eval_called = True


@dataclass
class FakeRuntime:
    """Provide lightweight controllable model-loading behavior."""

    tokenizer: object = field(default_factory=object)
    model: FakeModel = field(default_factory=FakeModel)
    fail_with: Exception | None = None
    calls: list[tuple[str, str, bool]] = field(default_factory=list)

    def load_tokenizer(
        self,
        model_name: str,
        *,
        local_files_only: bool,
    ) -> object:
        self.calls.append(("tokenizer", model_name, local_files_only))
        if self.fail_with is not None:
            raise self.fail_with
        return self.tokenizer

    def load_model(
        self,
        model_name: str,
        *,
        local_files_only: bool,
    ) -> FakeModel:
        self.calls.append(("model", model_name, local_files_only))
        if self.fail_with is not None:
            raise self.fail_with
        return self.model


def test_loader_always_places_encoder_on_cpu() -> None:
    runtime = FakeRuntime()
    config = SemanticEncoderConfig(local_files_only=True)

    backend = load_semantic_backend(config, runtime)

    assert backend.dimension == 384
    assert runtime.model.devices == ["cpu"]
    assert runtime.model.eval_called
    assert runtime.calls == [
        ("tokenizer", config.model_name, True),
        ("model", config.model_name, True),
    ]


def test_offline_cache_failure_has_actionable_message() -> None:
    runtime = FakeRuntime(fail_with=OSError("cache miss"))

    with pytest.raises(
        SemanticBackendLoadError,
        match="local Hugging Face cache.*network access",
    ):
        load_semantic_backend(
            SemanticEncoderConfig(local_files_only=True),
            runtime,
        )


def test_loader_rejects_invalid_hidden_dimension() -> None:
    runtime = FakeRuntime(model=FakeModel(hidden_size=0))

    with pytest.raises(SemanticBackendLoadError, match="Unable to load"):
        load_semantic_backend(SemanticEncoderConfig(), runtime)
