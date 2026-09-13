"""Tests for measured semantic index construction."""

from pathlib import Path
from unittest.mock import patch

import torch

from src.retrieval.bm25 import BM25Index
from src.retrieval.semantic import (
    SemanticEncoderConfig,
    SemanticIndex,
    build_and_store_semantic_index,
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
