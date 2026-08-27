"""Load exact corpus text once for retrieved-source validation."""

from pathlib import Path

from src.ingestion import discover_files, read_document


def load_source_texts(
    project_root: Path,
    corpus_root: Path,
) -> dict[str, str]:
    """Return exact text keyed by every discovered project-relative path."""
    return {
        document.file_path: document.text
        for document in (
            read_document(project_root, corpus_file)
            for corpus_file in discover_files(project_root, corpus_root)
        )
    }
