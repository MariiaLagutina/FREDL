"""Tests for retrieval result persistence."""

from pathlib import Path

from src.models import MinimalSearchResults, StudentSearchResults
from src.retrieval import save_search_results


def test_save_search_results_writes_valid_json_atomically(
    tmp_path: Path,
) -> None:
    """Submission output is formatted, validated, and fully replaced."""
    results = StudentSearchResults(
        search_results=[
            MinimalSearchResults(
                question_id="q-1",
                question="Where is the cache?",
                retrieved_sources=[],
            )
        ],
        k=5,
    )
    output_path = tmp_path / "nested" / "search-results.json"

    save_search_results(results, output_path)

    restored = StudentSearchResults.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert restored == results
    assert output_path.read_text(encoding="utf-8").endswith("\n")
    assert not output_path.with_suffix(".json.tmp").exists()
