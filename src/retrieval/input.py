"""Load validated retrieval input from JSON."""

from pathlib import Path

from pydantic import ValidationError

from src.models import RagDataset


def load_rag_dataset(input_path: Path) -> RagDataset:
    """Read one UTF-8 question dataset and validate its public schema."""
    try:
        return RagDataset.model_validate_json(
            input_path.read_text(encoding="utf-8")
        )
    except ValidationError as error:
        raise ValueError("Question dataset JSON is invalid") from error
