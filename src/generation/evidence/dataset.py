"""Validate local training data for binary evidence classification."""

from enum import Enum
from pathlib import Path
from typing import AbstractSet

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.generation.evidence.models import EvidenceDecision


class EvidenceDatasetSplit(str, Enum):
    """Separate model fitting, tuning, and final evaluation examples."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class EvidenceExampleKind(str, Enum):
    """Describe the evidence distinction exercised by one example."""

    DIRECT_ANSWER = "direct_answer"
    UNRELATED = "unrelated"
    TOPICAL_NEAR_MISS = "topical_near_miss"
    PARTIAL_ANSWER = "partial_answer"
    QUESTION_INJECTION = "question_injection"
    SOURCE_INJECTION = "source_injection"


class EvidenceTrainingExample(BaseModel):
    """Represent one labelled question-source pair."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )

    example_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    source_text: str = Field(min_length=1)
    decision: EvidenceDecision
    split: EvidenceDatasetSplit
    kind: EvidenceExampleKind


class EvidenceTrainingDataset(BaseModel):
    """Contain a non-empty collection of labelled evidence examples."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    examples: tuple[EvidenceTrainingExample, ...] = Field(min_length=1)

    def examples_for(
        self,
        split: EvidenceDatasetSplit,
    ) -> tuple[EvidenceTrainingExample, ...]:
        """Return one split without changing source order."""
        return tuple(
            example for example in self.examples if example.split is split
        )


def load_evidence_training_dataset(
    path: Path,
    protected_test_question_ids: AbstractSet[str] = frozenset(),
) -> EvidenceTrainingDataset:
    """Load JSON and enforce identity, split, and label invariants."""
    try:
        dataset = EvidenceTrainingDataset.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as error:
        message = "Evidence training dataset JSON is invalid"
        raise ValueError(message) from error
    _validate_dataset(dataset, protected_test_question_ids)
    return dataset


def _validate_dataset(
    dataset: EvidenceTrainingDataset,
    protected_test_question_ids: AbstractSet[str],
) -> None:
    """Reject duplicate identities and cross-split question leakage."""
    example_ids: set[str] = set()
    question_splits: dict[str, EvidenceDatasetSplit] = {}
    decisions_by_learning_split: dict[
        EvidenceDatasetSplit,
        set[EvidenceDecision],
    ] = {
        EvidenceDatasetSplit.TRAIN: set(),
        EvidenceDatasetSplit.VALIDATION: set(),
    }

    for example in dataset.examples:
        if example.example_id in example_ids:
            raise ValueError("Evidence training example IDs must be unique")
        example_ids.add(example.example_id)

        previous_split = question_splits.setdefault(
            example.question_id,
            example.split,
        )
        if previous_split is not example.split:
            raise ValueError(
                "Evidence question IDs must not cross dataset splits"
            )
        if (
            example.question_id in protected_test_question_ids
            and example.split is not EvidenceDatasetSplit.TEST
        ):
            raise ValueError(
                "Protected test questions must not enter learning splits"
            )
        if example.split in decisions_by_learning_split:
            decisions_by_learning_split[example.split].add(
                example.decision
            )

    expected_decisions = set(EvidenceDecision)
    for split, decisions in decisions_by_learning_split.items():
        if decisions != expected_decisions:
            raise ValueError(
                f"Evidence {split.value} split must contain both decisions"
            )
