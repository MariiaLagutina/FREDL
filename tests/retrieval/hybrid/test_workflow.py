"""Tests for measured stored hybrid search orchestration."""

from pathlib import Path
from unittest.mock import patch

import json
import pytest
import torch

from src.ingestion import Chunk
from src.retrieval.bm25 import BM25Index
from src.retrieval.hybrid import (
    HybridHit,
    RRFParameters,
    run_stored_hybrid_retrieval,
    run_stored_hybrid_search,
)
from src.retrieval.semantic import SemanticEncoderConfig


def test_stored_hybrid_search_loads_compatible_resources_and_measures(
    tmp_path: Path,
) -> None:
    chunk = Chunk("src/cache.py", 10, 15, "cache")
    parameters = RRFParameters(semantic_weight=0.5)
    config = SemanticEncoderConfig(local_files_only=True)
    clock_values = iter(
        [1.0, 1.5, 2.0, 2.75, 3.0, 5.0, 6.0, 6.2, 7.0, 7.25]
    )

    with (
        patch("src.retrieval.hybrid.workflow.IndexStore") as lexical_store,
        patch(
            "src.retrieval.hybrid.workflow.SemanticIndexStore"
        ) as semantic_store,
        patch("src.retrieval.hybrid.workflow.MiniLMEncoder") as encoder_type,
        patch("src.retrieval.hybrid.workflow.HybridRetriever") as hybrid_type,
    ):
        lexical_store.return_value.load.return_value = BM25Index(())
        encoder_type.return_value.encode.return_value = torch.tensor(
            [[1.0, 0.0]]
        )
        hybrid_type.return_value.search_with_embedding.return_value = [
            HybridHit(chunk, 0.1, 1, 2)
        ]
        report = run_stored_hybrid_search(
            tmp_path / "bm25.json",
            tmp_path / "semantic",
            "Where is the cache?",
            1,
            10,
            config,
            corpus_fingerprint="a" * 64,
            pipeline_fingerprint="b" * 64,
            rrf_parameters=parameters,
            clock=lambda: next(clock_values),
        )

    lexical_store.return_value.load.assert_called_once_with(
        "a" * 64,
        "b" * 64,
    )
    semantic_store.return_value.load.assert_called_once_with(
        expected_corpus_fingerprint="a" * 64,
        expected_pipeline_fingerprint="b" * 64,
        expected_model_name=config.model_name,
        expected_model_revision=config.model_revision,
    )
    encoder_type.assert_called_once_with(config)
    encoder_type.return_value.encode.assert_called_once_with(
        ("Where is the cache?",)
    )
    search_call = hybrid_type.return_value.search_with_embedding.call_args
    assert search_call.args[0] == "Where is the cache?"
    assert torch.equal(
        search_call.args[1],
        encoder_type.return_value.encode.return_value[0],
    )
    assert search_call.kwargs == {"top_k": 1, "candidate_k": 10}
    assert report.sources[0].file_path == "src/cache.py"
    assert report.lexical_index_load_seconds == 0.5
    assert report.semantic_index_load_seconds == 0.75
    assert report.model_load_seconds == 2.0
    assert report.query_encoding_seconds == pytest.approx(0.2)
    assert report.search_seconds == 0.25


def test_stored_hybrid_search_validates_before_loading_indexes(
    tmp_path: Path,
) -> None:
    with (
        patch("src.retrieval.hybrid.workflow.IndexStore") as lexical_store,
        patch(
            "src.retrieval.hybrid.workflow.SemanticIndexStore"
        ) as semantic_store,
    ):
        try:
            run_stored_hybrid_search(
                tmp_path / "bm25.json",
                tmp_path / "semantic",
                "!!!",
                5,
                20,
                SemanticEncoderConfig(),
                corpus_fingerprint="a" * 64,
                pipeline_fingerprint="b" * 64,
            )
        except ValueError as error:
            assert "searchable text" in str(error)
        else:
            raise AssertionError("Expected an unsearchable query error")

    lexical_store.assert_not_called()
    semantic_store.assert_not_called()


def test_stored_hybrid_batch_preserves_questions_and_encodes_once(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "questions.json"
    output_path = tmp_path / "results.json"
    input_path.write_text(
        json.dumps(
            {
                "rag_questions": [
                    {"question_id": "q-1", "question": "First question"},
                    {"question_id": "q-2", "question": "Second question"},
                ]
            }
        ),
        encoding="utf-8",
    )
    config = SemanticEncoderConfig(local_files_only=True)
    first = Chunk("first.py", 0, 5, "first")
    second = Chunk("second.py", 0, 6, "second")
    clock_values = iter(
        [0.0, 0.5, 1.0, 1.75, 2.0, 4.0, 5.0, 5.4, 6.0, 6.2]
    )

    with (
        patch("src.retrieval.hybrid.workflow.IndexStore") as lexical_store,
        patch(
            "src.retrieval.hybrid.workflow.SemanticIndexStore"
        ),
        patch("src.retrieval.hybrid.workflow.MiniLMEncoder") as encoder_type,
        patch("src.retrieval.hybrid.workflow.HybridRetriever") as hybrid_type,
    ):
        lexical_store.return_value.load.return_value = BM25Index(())
        encoder_type.return_value.encode.return_value = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]]
        )
        hybrid_type.return_value.search_with_embedding.side_effect = [
            [HybridHit(first, 0.1, 1, 2)],
            [HybridHit(second, 0.1, 2, 1)],
        ]
        report = run_stored_hybrid_retrieval(
            tmp_path / "bm25.json",
            tmp_path / "semantic",
            input_path,
            output_path,
            1,
            10,
            config,
            corpus_fingerprint="a" * 64,
            pipeline_fingerprint="b" * 64,
            clock=lambda: next(clock_values),
        )

    encoder_type.return_value.encode.assert_called_once_with(
        ("First question", "Second question")
    )
    assert [item.question_id for item in report.results.search_results] == [
        "q-1",
        "q-2",
    ]
    assert json.loads(output_path.read_text())["k"] == 1
    assert report.query_count == 2
    assert report.average_query_seconds == pytest.approx(0.3)
