"""Save and load validated BM25 snapshots without corpus parsing."""

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re

from pydantic import ValidationError

from src.ingestion import Chunk
from src.retrieval.bm25 import BM25Document, BM25Index, BM25Parameters
from src.retrieval.index_store.models import (
    StoredBM25Index,
    StoredChunk,
    StoredDocument,
    StoredFileFingerprint,
    StoredParameters,
)

SCHEMA_VERSION = 3
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class IncompatibleIndexError(ValueError):
    """Report that a persisted index must be rebuilt."""


@dataclass(frozen=True, slots=True)
class ReusableIndexSnapshot:
    """Expose one prior snapshot's documents grouped for incremental reuse."""

    documents_by_file: Mapping[str, tuple[BM25Document, ...]]
    file_fingerprints: Mapping[str, str]


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
        file_fingerprints: Mapping[str, str] | None = None,
    ) -> None:
        """Atomically save exact chunks, lexical fields, and parameters."""
        _validate_fingerprint(corpus_fingerprint)
        _validate_fingerprint(pipeline_fingerprint)
        for content_fingerprint in (file_fingerprints or {}).values():
            _validate_fingerprint(content_fingerprint)
        snapshot = _snapshot_from_index(
            index,
            corpus_fingerprint,
            pipeline_fingerprint,
            file_fingerprints or {},
        )
        snapshot = snapshot.model_copy(
            update={"snapshot_checksum": _snapshot_checksum(snapshot)}
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
        if not _checksum_matches(snapshot):
            raise ValueError(
                "Stored BM25 index checksum differs; reindex required"
            )
        return _index_from_snapshot(snapshot)

    def load_for_reuse(
        self,
        expected_pipeline_fingerprint: str,
    ) -> ReusableIndexSnapshot | None:
        """Return a prior snapshot's documents grouped by file, if reusable.

        Returns ``None`` instead of raising whenever the prior snapshot
        cannot be trusted for incremental reuse: it is missing, unreadable,
        built by an incompatible schema, or built by a different pipeline.
        Any of these are a normal, expected outcome (for example the very
        first `index` run) and simply mean every file must be rebuilt.
        """
        _validate_fingerprint(expected_pipeline_fingerprint)
        try:
            raw_snapshot = self._path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            snapshot = StoredBM25Index.model_validate_json(raw_snapshot)
        except ValidationError:
            return None
        if snapshot.schema_version != SCHEMA_VERSION:
            return None
        if snapshot.pipeline_fingerprint != expected_pipeline_fingerprint:
            return None
        if not snapshot.file_fingerprints:
            return None
        if snapshot.snapshot_checksum is None or not _checksum_matches(
            snapshot
        ):
            return None

        documents_by_file: dict[str, list[BM25Document]] = {}
        for stored_document in snapshot.documents:
            documents_by_file.setdefault(
                stored_document.chunk.file_path, []
            ).append(_document_from_stored(stored_document))
        return ReusableIndexSnapshot(
            documents_by_file={
                file_path: tuple(documents)
                for file_path, documents in documents_by_file.items()
            },
            file_fingerprints={
                entry.file_path: entry.content_fingerprint
                for entry in snapshot.file_fingerprints
            },
        )


def _snapshot_from_index(
    index: BM25Index,
    corpus_fingerprint: str,
    pipeline_fingerprint: str,
    file_fingerprints: Mapping[str, str],
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
            identifier_weight=parameters.identifier_weight,
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
                identifier_terms=document.identifier_terms,
            )
            for document in index.documents
        ),
        file_fingerprints=tuple(
            StoredFileFingerprint(
                file_path=file_path,
                content_fingerprint=content_fingerprint,
            )
            for file_path, content_fingerprint in sorted(
                file_fingerprints.items()
            )
        ),
    )


def _document_from_stored(document: StoredDocument) -> BM25Document:
    """Rebuild one runtime document from its stored lexical fields."""
    return BM25Document(
        chunk=Chunk(
            file_path=document.chunk.file_path,
            start=document.chunk.start,
            end=document.chunk.end,
            text=document.chunk.text,
            section_path=document.chunk.section_path,
        ),
        content_terms=document.content_terms,
        metadata_terms=document.metadata_terms,
        identifier_terms=document.identifier_terms,
    )


def _index_from_snapshot(snapshot: StoredBM25Index) -> BM25Index:
    """Rebuild runtime scoring structures from stored lexical fields."""
    parameters = snapshot.parameters
    return BM25Index(
        [_document_from_stored(document) for document in snapshot.documents],
        BM25Parameters(
            k1=parameters.k1,
            b=parameters.b,
            metadata_weight=parameters.metadata_weight,
            identifier_weight=parameters.identifier_weight,
        ),
    )


def _validate_fingerprint(corpus_fingerprint: str) -> None:
    """Require the canonical lowercase SHA-256 representation."""
    if _SHA256_PATTERN.fullmatch(corpus_fingerprint) is None:
        raise ValueError("Corpus fingerprint must be a lowercase SHA-256 hex")


def _snapshot_checksum(snapshot: StoredBM25Index) -> str:
    """Hash canonical persisted fields excluding the checksum itself."""
    payload = snapshot.model_dump_json(exclude={"snapshot_checksum"})
    return sha256(payload.encode("utf-8")).hexdigest()


def _checksum_matches(snapshot: StoredBM25Index) -> bool:
    """Accept legacy snapshots or verify a present content checksum."""
    checksum = snapshot.snapshot_checksum
    return checksum is None or (
        _SHA256_PATTERN.fullmatch(checksum) is not None
        and checksum == _snapshot_checksum(snapshot)
    )
