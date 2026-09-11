"""Score binary source evidence with a loaded causal language model."""

from collections.abc import Mapping
import math

from src.generation.backend import LoadedGenerationBackend
from src.generation.evidence.models import (
    EvidenceCandidate,
    EvidenceDecision,
)
from src.generation.evidence.prompt import (
    build_evidence_messages,
    evidence_decision_options,
)
from src.generation.mode.scorer import (
    SequenceScoringRuntime,
    TransformersSequenceScoringRuntime,
)


class HuggingFaceBinaryEvidenceScorer:
    """Average binary evidence scores across both option positions."""

    def __init__(
        self,
        backend: LoadedGenerationBackend,
        runtime: SequenceScoringRuntime | None = None,
    ) -> None:
        self._backend = backend
        self._runtime = runtime or TransformersSequenceScoringRuntime()

    def score_evidence(
        self,
        question: str,
        candidate: EvidenceCandidate,
    ) -> Mapping[EvidenceDecision, float]:
        """Score both decisions without generating an unconstrained answer."""
        decisions = tuple(EvidenceDecision)
        accumulated: dict[EvidenceDecision, list[float]] = {
            decision: [] for decision in decisions
        }
        for ordered_decisions in (decisions, tuple(reversed(decisions))):
            options = evidence_decision_options(ordered_decisions)
            messages = build_evidence_messages(
                question,
                candidate,
                ordered_decisions,
            )
            option_labels = tuple(
                options[decision] for decision in ordered_decisions
            )
            scores = self._runtime.score_continuations(
                messages,
                self._backend,
                option_labels,
            )
            if set(scores) != set(option_labels):
                raise ValueError(
                    "Evidence scoring runtime must score both options"
                )
            if any(not math.isfinite(score) for score in scores.values()):
                raise ValueError(
                    "Evidence scoring runtime returned a non-finite score"
                )
            for decision in ordered_decisions:
                accumulated[decision].append(scores[options[decision]])

        return {
            decision: sum(accumulated[decision]) / len(accumulated[decision])
            for decision in decisions
        }
