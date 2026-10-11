"""Tests for batch answer JSON boundaries."""

from pathlib import Path

import pytest

from src.generation import load_search_results, save_answers
from src.models import (
    MinimalAnswer,
    MinimalSource,
    SearchResults,
    SearchResultsAndAnswer,
)


def _source() -> MinimalSource:
    return MinimalSource(
        file_path="data/raw/guide.md",
        first_character_index=10,
        last_character_index=30,
    )


def test_loader_preserves_assignment_search_result_order(
    tmp_path: Path,
) -> None:
    """Valid retrieval JSON becomes the exact batch input model."""
    input_path = tmp_path / "search-results.json"
    input_path.write_text(
        """
        {
          "search_results": [
            {
              "question_id": "q-2",
              "question": "Second question?",
              "retrieved_sources": []
            },
            {
              "question_id": "q-1",
              "question": "First question?",
              "retrieved_sources": []
            }
          ],
          "k": 5
        }
        """,
        encoding="utf-8",
    )

    results = load_search_results(input_path)

    assert isinstance(results, SearchResults)
    assert [item.question_id for item in results.search_results] == [
        "q-2",
        "q-1",
    ]
    assert results.k == 5


def test_loader_reports_invalid_assignment_schema(tmp_path: Path) -> None:
    """Malformed retrieval input fails at the filesystem boundary."""
    input_path = tmp_path / "search-results.json"
    input_path.write_text('{"search_results": []}', encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Search results JSON is invalid",
    ):
        load_search_results(input_path)


def test_saver_writes_assignment_answers_atomically(tmp_path: Path) -> None:
    """The complete answer file is formatted, valid, and fully replaced."""
    results = SearchResultsAndAnswer(
        search_results=[
            MinimalAnswer(
                question_id="q-1",
                question="Where is the guide?",
                retrieved_sources=[_source()],
                answer="The guide is available. [Source 1]",
            )
        ],
        k=5,
    )
    output_path = tmp_path / "nested" / "answers.json"

    save_answers(results, output_path)

    restored = SearchResultsAndAnswer.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert restored == results
    assert output_path.read_text(encoding="utf-8").endswith("\n")
    assert not output_path.with_suffix(".json.tmp").exists()
