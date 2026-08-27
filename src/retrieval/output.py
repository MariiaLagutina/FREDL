"""Persist validated retrieval results for submission."""

from pathlib import Path

from src.models import RetrievalResults


def save_search_results(
    results: RetrievalResults,
    output_path: Path,
) -> None:
    """Atomically write formatted retrieval results as UTF-8 JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        results.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
