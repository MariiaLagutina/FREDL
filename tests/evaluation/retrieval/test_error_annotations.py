"""Tests for human-reviewed retrieval error annotations."""

from pathlib import Path

import pytest

from src.evaluation.retrieval.error_annotations import load_error_annotations
from src.evaluation.retrieval.error_models import RetrievalErrorCategory


def test_loads_complete_annotation(tmp_path: Path) -> None:
    path = tmp_path / "annotations.json"
    path.write_text(
        """{
  "annotations": [{
    "question_id": "docs-001",
    "category": "wrong_file",
    "hypothesis": "The implementation outranked the labelled guide.",
    "proposed_fix": "Add a dataset-kind path preference.",
    "next_test": "Compare the same miss with a Docs path preference."
  }]
}\n""",
        encoding="utf-8",
    )

    annotations = load_error_annotations(path)

    assert (
        annotations["docs-001"].category
        is RetrievalErrorCategory.WRONG_FILE
    )


def test_rejects_duplicate_annotation_ids(tmp_path: Path) -> None:
    annotation = """{
    "question_id": "docs-001",
    "category": "wrong_file",
    "hypothesis": "Evidence.",
    "proposed_fix": "Fix.",
    "next_test": "Test."
  }"""
    path = tmp_path / "annotations.json"
    path.write_text(
        '{"annotations": [' + annotation + "," + annotation + "]}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="IDs must be unique"):
        load_error_annotations(path)
