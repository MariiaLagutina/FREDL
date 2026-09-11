"""Tests for reviewed evidence promotion into training data."""

from pathlib import Path

import pytest

from src.generation.evidence import (
    EvidenceCandidateOrigin,
    EvidenceDatasetSplit,
    EvidenceDecision,
    EvidenceExampleKind,
    EvidenceReviewAnnotations,
    EvidenceReviewCandidate,
    EvidenceReviewQueue,
    build_evidence_review_annotations,
    load_evidence_review_annotations,
    promote_reviewed_evidence,
    save_evidence_review_annotations,
)


def _candidate(
    candidate_id: str,
    question_id: str,
    decision: EvidenceDecision,
    split: EvidenceDatasetSplit,
) -> EvidenceReviewCandidate:
    return EvidenceReviewCandidate(
        candidate_id=candidate_id,
        question_id=question_id,
        source_id=f"source-{candidate_id}:0:10",
        question=f"Question {question_id}?",
        reference_answer="Reference answer.",
        source_text=f"Source text {candidate_id}.",
        suggested_decision=decision,
        suggested_kind=(
            EvidenceExampleKind.DIRECT_ANSWER
            if decision is EvidenceDecision.ANSWERS
            else EvidenceExampleKind.TOPICAL_NEAR_MISS
        ),
        origin=EvidenceCandidateOrigin.GROUND_TRUTH,
        split=split,
    )


def _queue() -> EvidenceReviewQueue:
    return EvidenceReviewQueue(
        candidates=(
            _candidate(
                "train-a",
                "train-question-a",
                EvidenceDecision.ANSWERS,
                EvidenceDatasetSplit.TRAIN,
            ),
            _candidate(
                "train-b",
                "train-question-b",
                EvidenceDecision.DOES_NOT_ANSWER,
                EvidenceDatasetSplit.TRAIN,
            ),
            _candidate(
                "validation-a",
                "validation-question-a",
                EvidenceDecision.ANSWERS,
                EvidenceDatasetSplit.VALIDATION,
            ),
            _candidate(
                "validation-b",
                "validation-question-b",
                EvidenceDecision.DOES_NOT_ANSWER,
                EvidenceDatasetSplit.VALIDATION,
            ),
        )
    )


def _reviewed(queue: EvidenceReviewQueue) -> EvidenceReviewAnnotations:
    template = build_evidence_review_annotations(queue)
    return EvidenceReviewAnnotations(
        annotations=tuple(
            annotation.model_copy(update={"reviewed": True})
            for annotation in template.annotations
        )
    )


def test_template_keeps_suggestions_explicitly_unreviewed() -> None:
    """Generated labels cannot silently become human-approved truth."""
    queue = _queue()

    annotations = build_evidence_review_annotations(queue)

    assert len(annotations.annotations) == len(queue.candidates)
    assert all(not item.reviewed for item in annotations.annotations)
    assert annotations.annotations[0].decision is EvidenceDecision.ANSWERS


def test_promotion_preserves_order_and_applies_review_correction() -> None:
    """A reviewer may correct a suggestion before training promotion."""
    queue = _queue()
    reviews = list(_reviewed(queue).annotations)
    reviews[0] = reviews[0].model_copy(
        update={
            "decision": EvidenceDecision.DOES_NOT_ANSWER,
            "kind": EvidenceExampleKind.PARTIAL_ANSWER,
        }
    )
    reviews[1] = reviews[1].model_copy(
        update={"decision": EvidenceDecision.ANSWERS}
    )

    dataset = promote_reviewed_evidence(
        queue,
        EvidenceReviewAnnotations(annotations=tuple(reviews)),
    )

    assert [item.example_id for item in dataset.examples] == [
        candidate.candidate_id for candidate in queue.candidates
    ]
    assert dataset.examples[0].decision is EvidenceDecision.DOES_NOT_ANSWER
    assert dataset.examples[0].kind is EvidenceExampleKind.PARTIAL_ANSWER


def test_promotion_rejects_unreviewed_annotation() -> None:
    """One pending item blocks the complete queue from training."""
    queue = _queue()
    reviews = build_evidence_review_annotations(queue)

    with pytest.raises(ValueError, match="explicitly reviewed"):
        promote_reviewed_evidence(queue, reviews)


def test_promotion_rejects_missing_or_unrelated_ids() -> None:
    """Review annotations join only by the exact candidate identity."""
    queue = _queue()
    reviews = list(_reviewed(queue).annotations)
    reviews[0] = reviews[0].model_copy(update={"candidate_id": "other"})

    with pytest.raises(ValueError, match="same IDs"):
        promote_reviewed_evidence(
            queue,
            EvidenceReviewAnnotations(annotations=tuple(reviews)),
        )


def test_annotation_file_round_trip(tmp_path: Path) -> None:
    """Compact review decisions persist independently from source text."""
    annotations = build_evidence_review_annotations(_queue())
    path = tmp_path / "review/annotations.json"

    save_evidence_review_annotations(annotations, path)
    loaded = load_evidence_review_annotations(path)

    assert loaded == annotations
    assert path.read_text(encoding="utf-8").endswith("\n")
