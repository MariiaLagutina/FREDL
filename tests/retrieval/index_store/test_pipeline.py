"""Tests for deterministic retrieval pipeline compatibility identity."""

from dataclasses import replace

import pytest

from src.retrieval.bm25 import BM25Parameters
from src.retrieval.index_store import PipelineConfig, fingerprint_pipeline


def fingerprint(
    config: PipelineConfig,
    schema_version: int = 2,
) -> str:
    """Fingerprint one representative persisted-index configuration."""
    return fingerprint_pipeline(
        config,
        index_schema_version=schema_version,
    )


def test_pipeline_fingerprint_is_stable_sha256() -> None:
    """Equal declared build inputs produce one canonical identity."""
    first = fingerprint(PipelineConfig())
    second = fingerprint(PipelineConfig())

    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


@pytest.mark.parametrize(
    "changed",
    [
        PipelineConfig(max_chunk_size=1000),
        PipelineConfig(chunker_version=2),
        PipelineConfig(tokenizer_version=2),
        PipelineConfig(parameters=BM25Parameters(k1=1.2)),
        PipelineConfig(parameters=BM25Parameters(b=0.5)),
        PipelineConfig(parameters=BM25Parameters(metadata_weight=1.5)),
    ],
)
def test_pipeline_fingerprint_changes_with_build_input(
    changed: PipelineConfig,
) -> None:
    """Every declared chunking, tokenization, and scoring input is bound."""
    assert fingerprint(changed) != fingerprint(PipelineConfig())


def test_pipeline_fingerprint_changes_with_index_schema() -> None:
    """A persisted representation change requires a different identity."""
    assert fingerprint(PipelineConfig(), 2) != fingerprint(
        PipelineConfig(),
        3,
    )


@pytest.mark.parametrize(
    "config",
    [
        PipelineConfig(max_chunk_size=1),
        PipelineConfig(max_chunk_size=2000),
    ],
)
def test_pipeline_config_accepts_supported_chunk_limits(
    config: PipelineConfig,
) -> None:
    """Boundary chunk sizes supported by ingestion remain valid."""
    assert replace(config) == config


@pytest.mark.parametrize("max_chunk_size", [0, -1, 2001])
def test_pipeline_config_rejects_unsupported_chunk_limit(
    max_chunk_size: int,
) -> None:
    """A build config cannot request an unsupported chunk size."""
    with pytest.raises(ValueError, match="Maximum chunk size"):
        PipelineConfig(max_chunk_size=max_chunk_size)
