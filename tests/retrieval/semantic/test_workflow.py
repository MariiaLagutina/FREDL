"""Tests for measured semantic index construction."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.ingestion import Chunk
from src.retrieval.bm25 import BM25Index
from src.retrieval.semantic import (
    SemanticDocument,
    SemanticEncoderConfig,
    SemanticIndex,
    build_and_store_semantic_index,
    run_stored_semantic_search,
)


def test_build_workflow_measures_model_encoding_and_save_separately(
    tmp_path: Path,
) -> None:
    lexical_index = BM25Index(())
    semantic_index = SemanticIndex((), torch.empty((0, 384)))
    clock_values = iter([1.0, 2.5, 3.0, 7.0, 8.0, 8.25])
    config = SemanticEncoderConfig(local_files_only=True)

    with (
        patch("src.retrieval.semantic.workflow.load_semantic_backend") as load,
        patch(
            "src.retrieval.semantic.workflow.MiniLMEncoder"
        ) as encoder_type,
        patch(
            "src.retrieval.semantic.workflow.build_semantic_index",
            return_value=semantic_index,
        ) as build,
        patch(
            "src.retrieval.semantic.workflow.SemanticIndexStore"
        ) as store_type,
    ):
        backend = load.return_value
        encoder = encoder_type.return_value
        report = build_and_store_semantic_index(
            lexical_index,
            tmp_path / "semantic-index",
            config,
            corpus_fingerprint="a" * 64,
            pipeline_fingerprint="b" * 64,
            clock=lambda: next(clock_values),
        )

    encoder_type.assert_called_once_with(config, backend)
    build.assert_called_once_with(lexical_index, encoder)
    store_type.return_value.save.assert_called_once_with(
        semantic_index,
        corpus_fingerprint="a" * 64,
        pipeline_fingerprint="b" * 64,
        model_name=config.model_name,
        model_revision=config.model_revision,
    )
    assert report.document_count == 0
    assert report.dimension == 384
    assert report.model_load_seconds == 1.5
    assert report.encoding_seconds == 4.0
    assert report.save_seconds == 0.25


def test_search_workflow_loads_encodes_and_reports_exact_source(
    tmp_path: Path,
) -> None:
    document = SemanticDocument(
        Chunk(
            file_path="guide.md",
            start=10,
            end=14,
            text="text",
            section_path=(),
        )
    )
    index = SemanticIndex((document,), torch.tensor([[1.0, 0.0]]))
    config = SemanticEncoderConfig(local_files_only=True)
    clock_values = iter([1.0, 1.5, 2.0, 5.0, 6.0, 6.2, 7.0, 7.01])

    with (
        patch(
            "src.retrieval.semantic.workflow.SemanticIndexStore"
        ) as store_type,
        patch(
            "src.retrieval.semantic.workflow.MiniLMEncoder"
        ) as encoder_type,
    ):
        store_type.return_value.load.return_value = index
        encoder_type.return_value.encode.return_value = torch.tensor(
            [[1.0, 0.0]]
        )
        report = run_stored_semantic_search(
            tmp_path / "semantic-index",
            "Where is the guide?",
            1,
            config,
            corpus_fingerprint="a" * 64,
            pipeline_fingerprint="b" * 64,
            clock=lambda: next(clock_values),
        )

    store_type.return_value.load.assert_called_once_with(
        expected_corpus_fingerprint="a" * 64,
        expected_pipeline_fingerprint="b" * 64,
        expected_model_name=config.model_name,
        expected_model_revision=config.model_revision,
    )
    encoder_type.assert_called_once_with(config)
    encoder_type.return_value.encode.assert_called_once_with(
        ("Where is the guide?",)
    )
    assert report.sources[0].file_path == "guide.md"
    assert report.sources[0].first_character_index == 10
    assert report.sources[0].last_character_index == 14
    assert report.index_load_seconds == 0.5
    assert report.model_load_seconds == 3.0
    assert report.query_encoding_seconds == pytest.approx(0.2)
    assert report.search_seconds == pytest.approx(0.01)
