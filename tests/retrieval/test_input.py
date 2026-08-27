"""Tests for retrieval dataset loading."""

from pathlib import Path

import pytest

from src.retrieval import load_rag_dataset


def test_load_rag_dataset_reads_valid_utf8_json(tmp_path: Path) -> None:
    """A valid file becomes a typed dataset without changing identity."""
    input_path = tmp_path / "questions.json"
    input_path.write_text(
        """
        {
          "rag_questions": [
            {"question_id": "q-1", "question": "Где находится cache?"}
          ]
        }
        """,
        encoding="utf-8",
    )

    dataset = load_rag_dataset(input_path)

    assert dataset.rag_questions[0].question_id == "q-1"
    assert dataset.rag_questions[0].question == "Где находится cache?"


def test_load_rag_dataset_reports_invalid_schema(tmp_path: Path) -> None:
    """Malformed submission input fails at the filesystem boundary."""
    input_path = tmp_path / "questions.json"
    input_path.write_text('{"wrong_field": []}', encoding="utf-8")

    with pytest.raises(ValueError, match="Question dataset JSON is invalid"):
        load_rag_dataset(input_path)
