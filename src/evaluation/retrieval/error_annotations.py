"""Load human-reviewed retrieval error classifications."""

import json
from pathlib import Path
from typing import Any

from src.evaluation.retrieval.error_models import (
    RetrievalErrorCategory,
    RetrievalMissAnnotation,
)


def load_error_annotations(
    path: Path,
) -> dict[str, RetrievalMissAnnotation]:
    """Load unique manual annotations from one local JSON artifact."""
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(
        payload.get("annotations"), list
    ):
        raise ValueError("Retrieval error annotations JSON is invalid")
    annotations: dict[str, RetrievalMissAnnotation] = {}
    for raw in payload["annotations"]:
        annotation = _parse_annotation(raw)
        if annotation.question_id in annotations:
            raise ValueError("Retrieval error annotation IDs must be unique")
        annotations[annotation.question_id] = annotation
    return annotations


def _parse_annotation(raw: Any) -> RetrievalMissAnnotation:
    """Validate one JSON annotation without accepting implicit coercion."""
    if not isinstance(raw, dict):
        raise ValueError("Retrieval error annotation must be an object")
    required = (
        "question_id",
        "category",
        "hypothesis",
        "proposed_fix",
        "next_test",
    )
    if set(raw) != set(required) or any(
        not isinstance(raw[field], str) for field in required
    ):
        raise ValueError("Retrieval error annotation fields are invalid")
    try:
        category = RetrievalErrorCategory(raw["category"])
    except ValueError as error:
        message = "Retrieval error annotation category is invalid"
        raise ValueError(message) from error
    return RetrievalMissAnnotation(
        question_id=raw["question_id"],
        category=category,
        hypothesis=raw["hypothesis"],
        proposed_fix=raw["proposed_fix"],
        next_test=raw["next_test"],
    )
