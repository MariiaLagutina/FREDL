"""Tests for binary evidence training dataset boundaries."""

from pathlib import Path

import pytest

from src.generation.evidence import (
    EvidenceDatasetSplit,
    EvidenceDecision,
    EvidenceExampleKind,
    load_evidence_training_dataset,
)


def _example(
    example_id: str,
    question_id: str,
    decision: str,
    split: str,
) -> dict[str, str]:
    return {
        "example_id": example_id,
        "question_id": question_id,
        "source_id": f"source-{example_id}",
        "question": f"Question {question_id}?",
        "source_text": f"Evidence for {example_id}.",
        "decision": decision,
        "split": split,
        "kind": "direct_answer"
        if decision == "ANSWERS"
        else "topical_near_miss",
    }


def _valid_examples() -> list[dict[str, str]]:
    return [
        _example("train-a", "train-q-a", "ANSWERS", "train"),
        _example(
            "train-b",
            "train-q-b",
            "DOES_NOT_ANSWER",
            "train",
        ),
        _example(
            "validation-a",
            "validation-q-a",
            "ANSWERS",
            "validation",
        ),
        _example(
            "validation-b",
            "validation-q-b",
            "DOES_NOT_ANSWER",
            "validation",
        ),
        _example("test-a", "protected-q", "ANSWERS", "test"),
    ]


def _write_dataset(path: Path, examples: list[dict[str, str]]) -> None:
    import json

    path.write_text(
        json.dumps({"examples": examples}),
        encoding="utf-8",
    )


def test_loader_preserves_explicit_splits_and_typed_labels(
    tmp_path: Path,
) -> None:
    """Valid JSON becomes immutable typed training evidence."""
    path = tmp_path / "evidence.json"
    _write_dataset(path, _valid_examples())

    dataset = load_evidence_training_dataset(
        path,
        protected_test_question_ids={"protected-q"},
    )

    train = dataset.examples_for(EvidenceDatasetSplit.TRAIN)
    assert len(train) == 2
    assert train[0].decision is EvidenceDecision.ANSWERS
    assert train[1].kind is EvidenceExampleKind.TOPICAL_NEAR_MISS


def test_loader_rejects_duplicate_example_ids(tmp_path: Path) -> None:
    """Every training row keeps a stable unique identity."""
    examples = _valid_examples()
    examples[1]["example_id"] = examples[0]["example_id"]
    path = tmp_path / "duplicates.json"
    _write_dataset(path, examples)

    with pytest.raises(ValueError, match="example IDs must be unique"):
        load_evidence_training_dataset(path)


def test_loader_rejects_question_leakage_between_splits(
    tmp_path: Path,
) -> None:
    """Related source rows for one question stay in a single split."""
    examples = _valid_examples()
    examples[2]["question_id"] = examples[0]["question_id"]
    path = tmp_path / "leakage.json"
    _write_dataset(path, examples)

    with pytest.raises(ValueError, match="must not cross dataset splits"):
        load_evidence_training_dataset(path)


def test_loader_protects_held_out_question_ids(tmp_path: Path) -> None:
    """Known evaluation questions cannot silently enter model learning."""
    examples = _valid_examples()
    examples[0]["question_id"] = "protected-q"
    path = tmp_path / "protected.json"
    _write_dataset(path, examples)

    with pytest.raises(ValueError, match="must not enter learning splits"):
        load_evidence_training_dataset(
            path,
            protected_test_question_ids={"protected-q"},
        )


@pytest.mark.parametrize("split", ["train", "validation"])
def test_loader_requires_both_labels_in_learning_splits(
    tmp_path: Path,
    split: str,
) -> None:
    """A one-class split cannot measure or teach a binary distinction."""
    examples = [
        example
        for example in _valid_examples()
        if not (
            example["split"] == split
            and example["decision"] == "DOES_NOT_ANSWER"
        )
    ]
    path = tmp_path / "one-label.json"
    _write_dataset(path, examples)

    with pytest.raises(ValueError, match=f"{split} split"):
        load_evidence_training_dataset(path)


def test_loader_rejects_unknown_or_empty_fields(tmp_path: Path) -> None:
    """Malformed examples fail before they can reach fine-tuning."""
    examples = _valid_examples()
    examples[0]["question"] = " "
    examples[0]["unexpected"] = "value"
    path = tmp_path / "invalid.json"
    _write_dataset(path, examples)

    with pytest.raises(ValueError, match="dataset JSON is invalid"):
        load_evidence_training_dataset(path)
