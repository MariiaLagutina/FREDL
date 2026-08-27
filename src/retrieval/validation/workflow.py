"""Validate one retrieval-results file against an exact local corpus."""

from pathlib import Path

from pydantic import ValidationError

from src.models import RetrievalResults
from src.retrieval.validation.corpus import load_source_texts
from src.retrieval.validation.invariants import MAX_SOURCE_LENGTH
from src.retrieval.validation.models import SourceValidationReport
from src.retrieval.validation.runner import validate_retrieval_results


def validate_retrieval_file(
    results_path: Path,
    project_root: Path,
    corpus_root: Path,
    *,
    max_source_length: int = MAX_SOURCE_LENGTH,
) -> SourceValidationReport:
    """Load one result file and audit all sources against the corpus."""
    try:
        results = RetrievalResults.model_validate_json(
            results_path.read_text(encoding="utf-8")
        )
    except ValidationError as error:
        raise ValueError("Retrieval results JSON is invalid") from error

    source_texts = load_source_texts(project_root, corpus_root)
    return validate_retrieval_results(
        results,
        source_texts,
        max_source_length=max_source_length,
    )
