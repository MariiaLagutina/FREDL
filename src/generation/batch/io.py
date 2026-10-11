"""Validate batch retrieval input and persist batch answer output."""

from pathlib import Path

from pydantic import ValidationError

from src.models import SearchResults, SearchResultsAndAnswer


def load_search_results(input_path: Path) -> SearchResults:
    """Read one UTF-8 retrieval result file through its batch schema."""
    try:
        return SearchResults.model_validate_json(
            input_path.read_text(encoding="utf-8")
        )
    except ValidationError as error:
        raise ValueError("Search results JSON is invalid") from error


def save_answers(
    results: SearchResultsAndAnswer,
    output_path: Path,
) -> None:
    """Atomically write formatted batch answers as UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        results.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
