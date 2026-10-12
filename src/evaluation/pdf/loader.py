"""Load one private PDF golden dataset without weakening validation."""

from pathlib import Path

from pydantic import ValidationError

from src.evaluation.pdf.models import PdfGoldenDataset


def load_pdf_golden_dataset(path: Path) -> PdfGoldenDataset:
    """Read and strictly validate one local PDF golden JSON file."""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError("PDF golden dataset could not be read") from error

    try:
        return PdfGoldenDataset.model_validate_json(content)
    except ValidationError as error:
        raise ValueError("PDF golden dataset JSON is invalid") from error
