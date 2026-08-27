"""Save and load validated BM25 snapshots without corpus parsing."""

from pathlib import Path
import re

from pydantic import ValidationError

from src.ingestion import Chunk
from src.retrieval.bm25 import BM25Document, BM25Index, BM25Parameters
from src.retrieval.index_store.models import (
    StoredBM25Index,
    StoredChunk,
    StoredDocument,
    StoredParameters,
)

SCHEMA_VERSION = 2
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class IncompatibleIndexError(ValueError):
    """Report that a persisted index must be rebuilt."""


class IndexStore:
    """Persist one BM25 index as a deterministic, versioned JSON file."""

    def __init__(self, path: Path) -> None:
        """Store the explicit snapshot path without touching the filesystem."""
        self._path = path

    def save(
        self,
        index: BM25Index,
        corpus_fingerprint: str,
        pipeline_fingerprint: str,
    ) -> None:
        """Atomically save exact chunks, lexical fields, and parameters."""
        _validate_fingerprint(corpus_fingerprint)
        _validate_fingerprint(pipeline_fingerprint)
        snapshot = _snapshot_from_index(
            index,
            corpus_fingerprint,
            pipeline_fingerprint,
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary_path.write_text(
            snapshot.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self._path)

    def load(
        self,
        expected_corpus_fingerprint: str,
        expected_pipeline_fingerprint: str,
    ) -> BM25Index:
        """Load a compatible snapshot without reading source documents."""
        _validate_fingerprint(expected_corpus_fingerprint)
        _validate_fingerprint(expected_pipeline_fingerprint)
        try:
            snapshot = StoredBM25Index.model_validate_json(
                self._path.read_text(encoding="utf-8")
            )
        except ValidationError as error:
            raise ValueError("Stored BM25 index is invalid") from error

        if snapshot.schema_version != SCHEMA_VERSION:
            raise IncompatibleIndexError(
                "Stored BM25 index schema is incompatible; reindex required"
            )
        if snapshot.corpus_fingerprint != expected_corpus_fingerprint:
            raise IncompatibleIndexError(
                "Stored BM25 index corpus differs; reindex required"
            )
        if snapshot.pipeline_fingerprint != expected_pipeline_fingerprint:
            raise IncompatibleIndexError(
                "Stored BM25 index pipeline differs; reindex required"
            )
        return _index_from_snapshot(snapshot)


def _snapshot_from_index(
    index: BM25Index,
    corpus_fingerprint: str,
    pipeline_fingerprint: str,
) -> StoredBM25Index:
    """Convert runtime objects into the validated persistence schema."""
    parameters = index.parameters
    return StoredBM25Index(
        schema_version=SCHEMA_VERSION,
        corpus_fingerprint=corpus_fingerprint,
        pipeline_fingerprint=pipeline_fingerprint,
        parameters=StoredParameters(
            k1=parameters.k1,
            b=parameters.b,
            metadata_weight=parameters.metadata_weight,
        ),
        documents=tuple(
            StoredDocument(
                chunk=StoredChunk(
                    file_path=document.chunk.file_path,
                    start=document.chunk.start,
                    end=document.chunk.end,
                    text=document.chunk.text,
                    section_path=document.chunk.section_path,
                ),
                content_terms=document.content_terms,
                metadata_terms=document.metadata_terms,
            )
            for document in index.documents
        ),
    )


def _index_from_snapshot(snapshot: StoredBM25Index) -> BM25Index:
    """Rebuild runtime scoring structures from stored lexical fields."""
    parameters = snapshot.parameters
    return BM25Index(
        [
            BM25Document(
                chunk=Chunk(
                    file_path=document.chunk.file_path,
                    start=document.chunk.start,
                    end=document.chunk.end,
                    text=document.chunk.text,
                    section_path=document.chunk.section_path,
                ),
                content_terms=document.content_terms,
                metadata_terms=document.metadata_terms,
            )
            for document in snapshot.documents
        ],
        BM25Parameters(
            k1=parameters.k1,
            b=parameters.b,
            metadata_weight=parameters.metadata_weight,
        ),
    )


def _validate_fingerprint(corpus_fingerprint: str) -> None:
    """Require the canonical lowercase SHA-256 representation."""
    if _SHA256_PATTERN.fullmatch(corpus_fingerprint) is None:
        raise ValueError("Corpus fingerprint must be a lowercase SHA-256 hex")
