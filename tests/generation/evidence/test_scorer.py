"""Tests for calibrated Hugging Face binary evidence scoring."""

from collections.abc import Mapping, Sequence

import pytest

from src.generation import ChatMessage, LoadedGenerationBackend
from src.generation.evidence import (
    EvidenceCandidate,
    EvidenceDecision,
    HuggingFaceBinaryEvidenceScorer,
)
from src.models import MinimalSource


class RecordingRuntime:
    """Return one configured option-score mapping per prompt order."""

    def __init__(
        self,
        score_batches: Sequence[Mapping[str, float]],
    ) -> None:
        self._score_batches = tuple(score_batches)
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
        return self._score_batches[len(self.calls) - 1]


def _candidate() -> EvidenceCandidate:
    text = "The timeout is 30 seconds."
    return EvidenceCandidate(
        source_number=1,
        source=MinimalSource(
            file_path="data/raw/config.md",
            first_character_index=0,
            last_character_index=len(text),
        ),
        text=text,
    )


def test_binary_scorer_averages_both_option_positions() -> None:
    """Each semantic decision receives both positional option labels."""
    runtime = RecordingRuntime(
        (
            {"A": -0.2, "B": -0.8},
            {"A": -0.7, "B": -0.3},
        )
    )
    scorer = HuggingFaceBinaryEvidenceScorer(
        LoadedGenerationBackend(object(), object(), "cpu"),
        runtime,
    )

    scores = scorer.score_evidence("What is the timeout?", _candidate())

    assert scores == {
        EvidenceDecision.ANSWERS: -0.25,
        EvidenceDecision.DOES_NOT_ANSWER: -0.75,
    }
    assert len(runtime.calls) == 2
    assert "- A: ANSWERS" in runtime.calls[0][0][1].content
    assert "- B: ANSWERS" in runtime.calls[1][0][1].content
    assert runtime.calls[0][1] == runtime.calls[1][1] == ("A", "B")


@pytest.mark.parametrize(
    "score_batches, message",
    [
        (({"A": -0.2},), "score both options"),
        (
            ({"A": float("nan"), "B": -0.2},),
            "non-finite",
        ),
    ],
)
def test_binary_scorer_rejects_invalid_runtime_output(
    score_batches: Sequence[Mapping[str, float]],
    message: str,
) -> None:
    """Invalid option evidence cannot reach the selector."""
    scorer = HuggingFaceBinaryEvidenceScorer(
        LoadedGenerationBackend(object(), object(), "cpu"),
        RecordingRuntime(score_batches),
    )

    with pytest.raises(ValueError, match=message):
        scorer.score_evidence("What is the timeout?", _candidate())
