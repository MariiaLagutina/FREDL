"""Safely persist semantic metadata and its aligned embedding matrix."""

from hashlib import sha256
from pathlib import Path
from pickle import UnpicklingError
import re

from pydantic import ValidationError
import torch
from torch import Tensor

from src.ingestion import Chunk
from src.retrieval.semantic.index import SemanticIndex
from src.retrieval.semantic.models import SemanticDocument
from src.retrieval.semantic.store_models import (
    StoredSemanticChunk,
    StoredSemanticIndex,
)

SEMANTIC_SCHEMA_VERSION = 1
_METADATA_NAME = "metadata.json"
_EMBEDDINGS_NAME = "embeddings.pt"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class IncompatibleSemanticIndexError(ValueError):
    """Report that a semantic snapshot must be rebuilt."""


class SemanticIndexStore:
    """Persist one validated semantic index inside an explicit directory."""

    def __init__(self, directory: Path) -> None:
        """Store paths without touching the filesystem."""
        self._directory = directory
        self._metadata_path = directory / _METADATA_NAME
        self._embeddings_path = directory / _EMBEDDINGS_NAME

    def save(
        self,
        index: SemanticIndex,
        *,
        corpus_fingerprint: str,
        pipeline_fingerprint: str,
        model_name: str,
        model_revision: str,
    ) -> None:
        """Atomically publish vectors before their checksum metadata."""
        _validate_identity(
            corpus_fingerprint,
            pipeline_fingerprint,
            model_name,
            model_revision,
        )
        self._directory.mkdir(parents=True, exist_ok=True)
        embeddings_temporary = self._embeddings_path.with_suffix(".pt.tmp")
        metadata_temporary = self._metadata_path.with_suffix(".json.tmp")
        try:
            torch.save(index.embeddings, embeddings_temporary)
            embeddings_hash = _fingerprint_file(embeddings_temporary)
            snapshot = _snapshot_from_index(
                index,
                corpus_fingerprint=corpus_fingerprint,
                pipeline_fingerprint=pipeline_fingerprint,
                model_name=model_name,
                model_revision=model_revision,
                embeddings_sha256=embeddings_hash,
            )
            metadata_temporary.write_text(
                snapshot.model_dump_json(indent=2) + "\n",
                encoding="utf-8",
            )
            embeddings_temporary.replace(self._embeddings_path)
            metadata_temporary.replace(self._metadata_path)
        finally:
            embeddings_temporary.unlink(missing_ok=True)
            metadata_temporary.unlink(missing_ok=True)

    def load(
        self,
        *,
        expected_corpus_fingerprint: str,
        expected_pipeline_fingerprint: str,
        expected_model_name: str,
        expected_model_revision: str,
    ) -> SemanticIndex:
        """Load vectors only after validating metadata and checksum."""
        _validate_identity(
            expected_corpus_fingerprint,
            expected_pipeline_fingerprint,
            expected_model_name,
            expected_model_revision,
        )
        snapshot = _load_metadata(self._metadata_path)
        _require_compatible(
            snapshot,
            expected_corpus_fingerprint=expected_corpus_fingerprint,
            expected_pipeline_fingerprint=expected_pipeline_fingerprint,
            expected_model_name=expected_model_name,
            expected_model_revision=expected_model_revision,
        )
        embeddings_hash = _fingerprint_file(self._embeddings_path)
        if embeddings_hash != snapshot.embeddings_sha256:
            raise IncompatibleSemanticIndexError(
                "Stored semantic embeddings checksum differs; reindex required"
            )
        embeddings = _load_embeddings(self._embeddings_path)
        if embeddings.ndim != 2 or embeddings.shape[1] != snapshot.dimension:
            raise IncompatibleSemanticIndexError(
                "Stored semantic embedding dimension differs; reindex required"
            )
        return SemanticIndex(
            tuple(
                SemanticDocument(
                    Chunk(
                        file_path=document.file_path,
                        start=document.start,
                        end=document.end,
                        text=document.text,
                        section_path=document.section_path,
                    )
                )
                for document in snapshot.documents
            ),
            embeddings,
        )


def _snapshot_from_index(
    index: SemanticIndex,
    *,
    corpus_fingerprint: str,
    pipeline_fingerprint: str,
    model_name: str,
    model_revision: str,
    embeddings_sha256: str,
) -> StoredSemanticIndex:
    """Convert one runtime index into checksum-linked metadata."""
    return StoredSemanticIndex(
        schema_version=SEMANTIC_SCHEMA_VERSION,
        corpus_fingerprint=corpus_fingerprint,
        pipeline_fingerprint=pipeline_fingerprint,
        model_name=model_name,
        model_revision=model_revision,
        dimension=index.dimension,
        embeddings_sha256=embeddings_sha256,
        documents=tuple(
            StoredSemanticChunk(
                file_path=document.chunk.file_path,
                start=document.chunk.start,
                end=document.chunk.end,
                text=document.chunk.text,
                section_path=document.chunk.section_path,
            )
            for document in index.documents
        ),
    )


def _load_metadata(path: Path) -> StoredSemanticIndex:
    """Read strict semantic metadata with a stable validation error."""
    try:
        return StoredSemanticIndex.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except ValidationError as error:
        raise ValueError(
            "Stored semantic index metadata is invalid"
        ) from error


def _load_embeddings(path: Path) -> Tensor:
    """Load only tensor weights, never arbitrary pickled objects."""
    try:
        embeddings = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, UnpicklingError) as error:
        raise ValueError("Stored semantic embeddings are invalid") from error
    if not isinstance(embeddings, Tensor):
        raise ValueError("Stored semantic embeddings must contain one tensor")
    return embeddings


def _require_compatible(
    snapshot: StoredSemanticIndex,
    *,
    expected_corpus_fingerprint: str,
    expected_pipeline_fingerprint: str,
    expected_model_name: str,
    expected_model_revision: str,
) -> None:
    """Reject every known identity mismatch before tensor loading."""
    comparisons = (
        (snapshot.schema_version, SEMANTIC_SCHEMA_VERSION, "schema"),
        (snapshot.corpus_fingerprint, expected_corpus_fingerprint, "corpus"),
        (
            snapshot.pipeline_fingerprint,
            expected_pipeline_fingerprint,
            "pipeline",
        ),
        (snapshot.model_name, expected_model_name, "model"),
        (snapshot.model_revision, expected_model_revision, "model revision"),
    )
    for actual, expected, label in comparisons:
        if actual != expected:
            raise IncompatibleSemanticIndexError(
                f"Stored semantic index {label} differs; reindex required"
            )


def _validate_identity(
    corpus_fingerprint: str,
    pipeline_fingerprint: str,
    model_name: str,
    model_revision: str,
) -> None:
    """Require canonical fingerprints and explicit model identity."""
    for fingerprint in (corpus_fingerprint, pipeline_fingerprint):
        if _SHA256_PATTERN.fullmatch(fingerprint) is None:
            raise ValueError(
                "Semantic fingerprint must be lowercase SHA-256 hex"
            )
    if not model_name.strip():
        raise ValueError("Semantic model name must not be empty")
    if not model_revision.strip():
        raise ValueError("Semantic model revision must not be empty")


def _fingerprint_file(path: Path) -> str:
    """Hash one binary artifact without loading it into memory."""
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
