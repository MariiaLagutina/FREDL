"""Tests for the complete retrieval file workflow."""

from pathlib import Path

from src.ingestion import Chunk
from src.models import RetrievalResults
from src.retrieval import run_retrieval, run_stored_retrieval
from src.retrieval.bm25 import BM25Document, BM25Index
from src.retrieval.index_store import IndexStore

FINGERPRINT = "a" * 64


def test_run_retrieval_connects_input_search_and_output(
    tmp_path: Path,
) -> None:
    """One loaded index serves a validated question file end to end."""
    index = BM25Index(
        [
            BM25Document(
                chunk=Chunk("src/cache.py", 10, 15, "cache"),
                content_terms=("cache",),
            )
        ]
    )
    input_path = tmp_path / "questions.json"
    input_path.write_text(
        '{"rag_questions": ['
        '{"question_id": "q-1", "question": "Where is cache?"}'
        "]}",
        encoding="utf-8",
    )
    output_path = tmp_path / "results" / "search-results.json"

    results = run_retrieval(index, input_path, output_path, k=1)

    restored = RetrievalResults.model_validate_json(
        output_path.read_text(encoding="utf-8")
    )
    assert restored == results
    assert restored.k == 1
    assert restored.search_results[0].retrieved_sources[0].file_path == (
        "src/cache.py"
    )


def test_run_stored_retrieval_loads_index_once_for_question_file(
    tmp_path: Path,
) -> None:
    """A compatible Step 13 snapshot feeds the complete workflow."""
    index_path = tmp_path / "bm25-index.json"
    IndexStore(index_path).save(
        BM25Index(
            [
                BM25Document(
                    chunk=Chunk("docs/cache.md", 0, 5, "cache"),
                    content_terms=("cache",),
                )
            ]
        ),
        FINGERPRINT,
    )
    input_path = tmp_path / "questions.json"
    input_path.write_text(
        '{"rag_questions": ['
        '{"question_id": "q-1", "question": "cache"}'
        "]}",
        encoding="utf-8",
    )
    output_path = tmp_path / "search-results.json"

    results = run_stored_retrieval(
        index_path,
        FINGERPRINT,
        input_path,
        output_path,
        k=1,
    )

    assert results.search_results[0].retrieved_sources[0].file_path == (
        "docs/cache.md"
    )
    assert output_path.exists()
