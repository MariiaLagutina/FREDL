"""Promote fully reviewed candidates into binary training data."""

from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.generation.evidence.dataset import (
    EvidenceExampleKind,
    EvidenceTrainingDataset,
    EvidenceTrainingExample,
    validate_evidence_training_dataset,
)
from src.generation.evidence.models import EvidenceDecision
from src.generation.evidence.review import EvidenceReviewQueue


Item = TypeVar("Item")


class EvidenceReviewAnnotation(BaseModel):
    """Record one explicit human decision for a pending candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str = Field(min_length=1)
    reviewed: bool = False
    include: bool = True
    decision: EvidenceDecision
    kind: EvidenceExampleKind
    note: str = ""


class EvidenceReviewAnnotations(BaseModel):
    """Contain a complete set of compact review decisions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: str = "1"
    annotations: tuple[EvidenceReviewAnnotation, ...] = Field(min_length=1)


def build_evidence_review_annotations(
    queue: EvidenceReviewQueue,
) -> EvidenceReviewAnnotations:
    """Create an unreviewed template with visible suggested labels."""
    return EvidenceReviewAnnotations(
        annotations=tuple(
            EvidenceReviewAnnotation(
                candidate_id=candidate.candidate_id,
                decision=candidate.suggested_decision,
                kind=candidate.suggested_kind,
            )
            for candidate in queue.candidates
        )
    )


def load_evidence_review_annotations(
    path: Path,
) -> EvidenceReviewAnnotations:
    """Load a strict UTF-8 JSON review annotation artifact."""
    try:
        return EvidenceReviewAnnotations.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as error:
        message = "Evidence review annotations JSON is invalid"
        raise ValueError(message) from error


def save_evidence_review_annotations(
    annotations: EvidenceReviewAnnotations,
    output_path: Path,
) -> None:
    """Write compact review annotations without source duplication."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        annotations.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def promote_reviewed_evidence(
    queue: EvidenceReviewQueue,
    reviews: EvidenceReviewAnnotations,
) -> EvidenceTrainingDataset:
    """Convert only a complete reviewed queue into training examples."""
    candidates_by_id = _unique_by_id(
        (
            (candidate.candidate_id, candidate)
            for candidate in queue.candidates
        ),
        "Evidence review candidate IDs must be unique",
    )
    reviews_by_id = _unique_by_id(
        ((review.candidate_id, review) for review in reviews.annotations),
        "Evidence review annotation IDs must be unique",
    )
    if set(candidates_by_id) != set(reviews_by_id):
        raise ValueError(
            "Evidence review candidates and annotations must contain the "
            "same IDs"
        )
    if any(not review.reviewed for review in reviews.annotations):
        raise ValueError(
            "Every evidence candidate must be explicitly reviewed"
        )

    examples = tuple(
        EvidenceTrainingExample(
            example_id=candidate.candidate_id,
            question_id=candidate.question_id,
            source_id=candidate.source_id,
            question=candidate.question,
            source_text=candidate.source_text,
            decision=review.decision,
            split=candidate.split,
            kind=review.kind,
        )
        for candidate in queue.candidates
        for review in (reviews_by_id[candidate.candidate_id],)
        if review.include
    )
    if not examples:
        raise ValueError("Evidence review excluded every training example")
    dataset = EvidenceTrainingDataset(examples=examples)
    validate_evidence_training_dataset(dataset)
    return dataset


def _unique_by_id(
    items: Iterable[tuple[str, Item]],
    message: str,
) -> dict[str, Item]:
    """Build an identity map without silently replacing duplicates."""
    identity_map: dict[str, Item] = {}
    for identity, item in items:
        if identity in identity_map:
            raise ValueError(message)
        identity_map[identity] = item
    return identity_map
