"""Tests for MiniLM batching and attention-aware mean pooling."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.retrieval.semantic import (
    LoadedSemanticBackend,
    MiniLMEncoder,
    SemanticEncoderConfig,
    SemanticEncodingError,
)


class FakeTokenizer:
    """Return deterministic token IDs and masks for requested texts."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert padding and truncation and return_tensors == "pt"
        self.calls.append((texts, max_length))
        return {
            "input_ids": torch.tensor([[1, 2, 0]] * len(texts)),
            "attention_mask": torch.tensor([[1, 1, 0]] * len(texts)),
        }


@dataclass
class FakeModel:
    """Return fixed token states while recording inference mode."""

    hidden_state: torch.Tensor
    grad_enabled: list[bool]

    def __call__(self, **model_inputs: Any) -> SimpleNamespace:
        self.grad_enabled.append(torch.is_grad_enabled())
        count = model_inputs["input_ids"].shape[0]
        return SimpleNamespace(last_hidden_state=self.hidden_state[:count])


def _encoder(
    batch_size: int = 32,
) -> tuple[MiniLMEncoder, FakeTokenizer, FakeModel]:
    tokenizer = FakeTokenizer()
    hidden_state = torch.tensor(
        [
            [[3.0, 0.0], [0.0, 1.0], [100.0, 100.0]],
            [[0.0, 4.0], [2.0, 0.0], [100.0, 100.0]],
        ]
    )
    model = FakeModel(hidden_state, [])
    backend = LoadedSemanticBackend(tokenizer, model, dimension=2)
    config = SemanticEncoderConfig(batch_size=batch_size, max_length=17)
    return MiniLMEncoder(config, backend), tokenizer, model


def test_encoder_mean_pools_only_unmasked_tokens_and_normalizes() -> None:
    encoder, tokenizer, model = _encoder()

    embeddings = encoder.encode(["first", "second"])

    assert embeddings == pytest.approx(
        torch.tensor([[0.9486833, 0.3162278], [0.4472136, 0.8944272]])
    )
    assert tokenizer.calls == [(["first", "second"], 17)]
    assert model.grad_enabled == [False]
    assert embeddings.device.type == "cpu"


def test_encoder_splits_inputs_into_configured_batches() -> None:
    encoder, tokenizer, _ = _encoder(batch_size=1)

    embeddings = encoder.encode(["first", "second"])

    assert embeddings.shape == (2, 2)
    assert tokenizer.calls == [(["first"], 17), (["second"], 17)]


def test_encoder_returns_shaped_empty_matrix_without_backend_calls() -> None:
    encoder, tokenizer, model = _encoder()

    embeddings = encoder.encode([])

    assert embeddings.shape == (0, 2)
    assert tokenizer.calls == []
    assert model.grad_enabled == []


@pytest.mark.parametrize("text", ["", "   "])
def test_encoder_rejects_empty_text(text: str) -> None:
    encoder, _, _ = _encoder()

    with pytest.raises(ValueError, match="non-empty strings"):
        encoder.encode([text])


def test_encoder_rejects_missing_attention_mask() -> None:
    _, _, model = _encoder()
    backend = LoadedSemanticBackend(
        tokenizer=lambda *_args, **_kwargs: {},
        model=model,
        dimension=2,
    )
    encoder = MiniLMEncoder(backend=backend)

    with pytest.raises(SemanticEncodingError, match="attention mask"):
        encoder.encode(["text"])
