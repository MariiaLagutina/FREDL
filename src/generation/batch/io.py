"""Validate batch retrieval input and persist assignment answer output."""

from pathlib import Path

from pydantic import ValidationError

from src.models import StudentSearchResults, StudentSearchResultsAndAnswer


def load_student_search_results(input_path: Path) -> StudentSearchResults:
    """Read one UTF-8 retrieval result file through its assignment schema."""
    try:
        return StudentSearchResults.model_validate_json(
            input_path.read_text(encoding="utf-8")
        )
    except ValidationError as error:
        raise ValueError("Student search results JSON is invalid") from error


def save_student_answers(
    results: StudentSearchResultsAndAnswer,
    output_path: Path,
) -> None:
    """Atomically write formatted assignment answers as UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        results.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
