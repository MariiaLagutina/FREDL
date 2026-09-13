"""Encode text into normalized MiniLM vectors on CPU."""

from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from src.retrieval.semantic.config import SemanticEncoderConfig
from src.retrieval.semantic.runtime import (
    LoadedSemanticBackend,
    load_semantic_backend,
)


class SemanticEncodingError(RuntimeError):
    """Report malformed output from the configured semantic backend."""


class MiniLMEncoder:
    """Reuse one loaded MiniLM backend for deterministic CPU embeddings."""

    def __init__(
        self,
        config: SemanticEncoderConfig | None = None,
        backend: LoadedSemanticBackend | None = None,
    ) -> None:
        """Load or accept one backend without encoding any text yet."""
        self._config = config or SemanticEncoderConfig()
        self._backend = backend or load_semantic_backend(self._config)

    @property
    def dimension(self) -> int:
        """Return the encoder's fixed vector width."""
        return self._backend.dimension

    def encode(self, texts: Sequence[str]) -> Tensor:
        """Return one L2-normalized CPU vector per input text."""
        normalized_texts = tuple(texts)
        if any(
            not isinstance(text, str) or not text.strip()
            for text in normalized_texts
        ):
            raise ValueError("Semantic input texts must be non-empty strings")
        if not normalized_texts:
            return torch.empty((0, self.dimension), dtype=torch.float32)

        batches = [
            self._encode_batch(
                normalized_texts[start: start + self._config.batch_size]
            )
            for start in range(
                0,
                len(normalized_texts),
                self._config.batch_size,
            )
        ]
        return torch.cat(batches, dim=0)

    def _encode_batch(self, texts: Sequence[str]) -> Tensor:
        """Tokenize and mean-pool one bounded inference batch."""
        model_inputs: Any = self._backend.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self._config.max_length,
            return_tensors="pt",
        )
        attention_mask = model_inputs.get("attention_mask")
        if not isinstance(attention_mask, Tensor):
            raise SemanticEncodingError(
                "Semantic tokenizer did not return an attention mask"
            )
        with torch.inference_mode():
            output = self._backend.model(**model_inputs)
        hidden_state = getattr(output, "last_hidden_state", None)
        if not isinstance(hidden_state, Tensor):
            raise SemanticEncodingError(
                "Semantic model did not return last hidden state"
            )
        return _mean_pool(hidden_state, attention_mask, self.dimension)


def _mean_pool(
    hidden_state: Tensor,
    attention_mask: Tensor,
    expected_dimension: int,
) -> Tensor:
    """Average real token states and normalize each resulting vector."""
    if hidden_state.ndim != 3 or attention_mask.ndim != 2:
        raise SemanticEncodingError("Semantic backend returned invalid shapes")
    if hidden_state.shape[:2] != attention_mask.shape:
        raise SemanticEncodingError("Semantic token and mask shapes differ")
    if hidden_state.shape[2] != expected_dimension:
        raise SemanticEncodingError("Semantic embedding dimension changed")

    mask = attention_mask.to(dtype=hidden_state.dtype).unsqueeze(-1)
    token_counts = mask.sum(dim=1)
    if bool((token_counts == 0).any()):
        raise SemanticEncodingError("Semantic input has no unmasked tokens")
    pooled = (hidden_state * mask).sum(dim=1) / token_counts
    return torch.nn.functional.normalize(pooled, p=2, dim=1).to(
        device="cpu",
        dtype=torch.float32,
    )
