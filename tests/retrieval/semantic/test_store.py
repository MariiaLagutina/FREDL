"""Tests for checksum-linked semantic index persistence."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.ingestion import Chunk
from src.retrieval.semantic import (
    IncompatibleSemanticIndexError,
    SemanticDocument,
    SemanticIndex,
    SemanticIndexStore,
)

CORPUS_FINGERPRINT = "a" * 64
PIPELINE_FINGERPRINT = "b" * 64
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c" * 40


def _index() -> SemanticIndex:
    documents = (
        SemanticDocument(
            Chunk(
                file_path="guide.md",
                start=5,
                end=10,
                text="first",
                section_path=("Guide",),
            )
        ),
        SemanticDocument(
            Chunk(
                file_path="source.py",
                start=20,
                end=26,
                text="second",
                section_path=(),
            )
        ),
    )
    return SemanticIndex(
        documents,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )


def _save(store: SemanticIndexStore) -> None:
    store.save(
        _index(),
        corpus_fingerprint=CORPUS_FINGERPRINT,
        pipeline_fingerprint=PIPELINE_FINGERPRINT,
        model_name=MODEL_NAME,
        model_revision=MODEL_REVISION,
    )


def _load(store: SemanticIndexStore) -> SemanticIndex:
    return store.load(
        expected_corpus_fingerprint=CORPUS_FINGERPRINT,
        expected_pipeline_fingerprint=PIPELINE_FINGERPRINT,
        expected_model_name=MODEL_NAME,
        expected_model_revision=MODEL_REVISION,
    )


def test_store_round_trip_preserves_documents_and_embeddings(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "semantic-index"
    store = SemanticIndexStore(directory)

    _save(store)
    loaded = _load(store)

    assert loaded.documents == _index().documents
    assert torch.equal(loaded.embeddings, _index().embeddings)
    assert sorted(path.name for path in directory.iterdir()) == [
        "embeddings.pt",
        "metadata.json",
    ]
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["model_name"] == MODEL_NAME
    assert metadata["model_revision"] == MODEL_REVISION
    assert metadata["dimension"] == 2
    assert len(metadata["embeddings_sha256"]) == 64


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("expected_corpus_fingerprint", "d" * 64, "corpus differs"),
        ("expected_pipeline_fingerprint", "d" * 64, "pipeline differs"),
        ("expected_model_name", "different/model", "model differs"),
        ("expected_model_revision", "d" * 40, "model revision differs"),
    ],
)
def test_load_rejects_incompatible_identity_before_loading_vectors(
    tmp_path: Path,
    argument: str,
    value: str,
    message: str,
) -> None:
    store = SemanticIndexStore(tmp_path / "semantic-index")
    _save(store)
    arguments = {
        "expected_corpus_fingerprint": CORPUS_FINGERPRINT,
        "expected_pipeline_fingerprint": PIPELINE_FINGERPRINT,
        "expected_model_name": MODEL_NAME,
        "expected_model_revision": MODEL_REVISION,
    }
    arguments[argument] = value

    with patch("src.retrieval.semantic.store.torch.load") as load_tensor:
        with pytest.raises(IncompatibleSemanticIndexError, match=message):
            store.load(**arguments)

    load_tensor.assert_not_called()


def test_load_rejects_changed_embeddings_before_deserialization(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "semantic-index"
    store = SemanticIndexStore(directory)
    _save(store)
    with (directory / "embeddings.pt").open("ab") as stream:
        stream.write(b"changed")

    with patch("src.retrieval.semantic.store.torch.load") as load_tensor:
        with pytest.raises(
            IncompatibleSemanticIndexError,
            match="checksum differs",
        ):
            _load(store)

    load_tensor.assert_not_called()


def test_load_rejects_metadata_schema_mismatch(tmp_path: Path) -> None:
    directory = tmp_path / "semantic-index"
    store = SemanticIndexStore(directory)
    _save(store)
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["schema_version"] = 999
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(IncompatibleSemanticIndexError, match="schema differs"):
        _load(store)


def test_load_rejects_tensor_dimension_mismatch(tmp_path: Path) -> None:
    directory = tmp_path / "semantic-index"
    store = SemanticIndexStore(directory)
    _save(store)
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["dimension"] = 3
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(
        IncompatibleSemanticIndexError,
        match="dimension differs",
    ):
        _load(store)


def test_save_replaces_existing_snapshot_without_temporary_files(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "semantic-index"
    store = SemanticIndexStore(directory)

    _save(store)
    _save(store)

    assert _load(store).documents == _index().documents
    assert not list(directory.glob("*.tmp"))
