"""Tests for finite Hugging Face answer-mode scoring."""

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from src.generation import (
    AnswerMode,
    ChatMessage,
    ContextBuildResult,
    HuggingFaceAnswerModeScorer,
    LoadedGenerationBackend,
    TransformersSequenceScoringRuntime,
)


class RecordingRuntime:
    """Return fixed scores while preserving scorer boundary inputs."""

    def __init__(self, scores: Mapping[str, float]) -> None:
        self.scores = scores
        self.calls: list[
            tuple[tuple[ChatMessage, ...], tuple[str, ...]]
        ] = []

    def score_continuations(
        self,
        messages: Sequence[ChatMessage],
        backend: LoadedGenerationBackend,
        candidates: Sequence[str],
    ) -> Mapping[str, float]:
        del backend
        self.calls.append((tuple(messages), tuple(candidates)))
        return self.scores


class FakeTokenizer:
    """Encode a prompt and two mode labels with deterministic IDs."""

    def apply_chat_template(self, *_args: Any, **_kwargs: Any) -> object:
        return {"input_ids": torch.tensor([[7, 8]])}

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool,
    ) -> list[int]:
        del add_special_tokens
        return {
            "SUPPORTED": [1, 2],
            "INSUFFICIENT": [3],
        }[text]


class FakeCausalModel:
    """Prefer the supported label at every scored token position."""

    def __call__(self, **inputs: Any) -> object:
        input_ids = inputs["input_ids"]
        sequence_length = input_ids.shape[1]
        logits = torch.zeros((1, sequence_length, 9))
        logits[0, 1, 1] = 4.0
        logits[0, 2, 2] = 4.0
        logits[0, 1, 3] = 1.0
        return SimpleNamespace(logits=logits)


def _context() -> ContextBuildResult:
    return ContextBuildResult("evidence", (), 1, 0)


def test_hugging_face_scorer_maps_labels_back_to_modes() -> None:
    """The adapter keeps classifier text internal to typed mode scores."""
    backend = LoadedGenerationBackend(object(), object(), "cpu")
    runtime = RecordingRuntime(
        {"SUPPORTED": -0.2, "INSUFFICIENT": -0.8}
    )
    scorer = HuggingFaceAnswerModeScorer(backend, runtime)

    scores = scorer.score_modes(
        "What is configured?",
        _context(),
        (AnswerMode.SUPPORTED, AnswerMode.INSUFFICIENT),
    )

    assert scores == {
        AnswerMode.SUPPORTED: -0.2,
        AnswerMode.INSUFFICIENT: -0.8,
    }
    messages, candidates = runtime.calls[0]
    assert candidates == ("SUPPORTED", "INSUFFICIENT")
    assert "What is configured?" in messages[1].content


def test_hugging_face_scorer_rejects_incomplete_runtime_output() -> None:
    """Missing candidate evidence cannot reach the selector."""
    scorer = HuggingFaceAnswerModeScorer(
        LoadedGenerationBackend(object(), object(), "cpu"),
        RecordingRuntime({"SUPPORTED": -0.2}),
    )

    with pytest.raises(ValueError, match="score every candidate once"):
        scorer.score_modes(
            "What is configured?",
            _context(),
            (AnswerMode.SUPPORTED, AnswerMode.INSUFFICIENT),
        )


def test_transformers_runtime_scores_mean_candidate_log_probability() -> None:
    """Different token lengths remain comparable without free generation."""
    runtime = TransformersSequenceScoringRuntime()
    backend = LoadedGenerationBackend(
        FakeTokenizer(),
        FakeCausalModel(),
        "cpu",
    )

    scores = runtime.score_continuations(
        (ChatMessage("user", "Classify this."),),
        backend,
        ("SUPPORTED", "INSUFFICIENT"),
    )

    assert scores["SUPPORTED"] > scores["INSUFFICIENT"]
