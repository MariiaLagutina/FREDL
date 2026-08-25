"""Load a fixed BM25 corpus and its validated query labels."""

from pathlib import Path

from src.evaluation.bm25.models import EvaluationSuite, QueryKind
from src.ingestion import (
    SourceDocument,
    chunk_document,
    discover_files,
    read_document,
)
from src.retrieval.bm25 import BM25Document, build_bm25_documents


def load_suite(suite_root: Path) -> EvaluationSuite:
    """Load and validate one versioned evaluation suite definition."""
    suite_path = suite_root / "suite.json"
    suite = EvaluationSuite.model_validate_json(
        suite_path.read_text(encoding="utf-8")
    )
    if suite.max_chunk_size <= 0:
        raise ValueError("Evaluation maximum chunk size must be positive")
    query_ids = [query.query_id for query in suite.queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Evaluation query IDs must be unique")
    if {query.kind for query in suite.queries} != set(QueryKind):
        raise ValueError("Evaluation suite must contain docs and code queries")
    return suite


def build_suite_documents(
    project_root: Path,
    suite_root: Path,
    suite: EvaluationSuite,
) -> list[BM25Document]:
    """Run normal ingestion over every fixed corpus file in the suite."""
    corpus_root = suite_root / "corpus"
    documents: list[BM25Document] = []
    sources: dict[str, SourceDocument] = {}
    for corpus_file in discover_files(project_root, corpus_root):
        source = read_document(project_root, corpus_file)
        sources[source.file_path] = source
        chunks = chunk_document(source, suite.max_chunk_size)
        documents.extend(build_bm25_documents(source, chunks))
    _validate_reference_sources(suite, sources)
    return documents


def _validate_reference_sources(
    suite: EvaluationSuite,
    documents: dict[str, SourceDocument],
) -> None:
    """Require every label to stay inside its fixed corpus document."""
    for query in suite.queries:
        for reference in query.sources:
            document = documents.get(reference.file_path)
            if document is None:
                raise ValueError("Evaluation reference file must exist")
            if reference.last_character_index > len(document.text):
                raise ValueError("Evaluation reference must stay inside file")
