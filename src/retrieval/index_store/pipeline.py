"""Identify every declared input that shapes a persisted BM25 index."""

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json

from src.ingestion.chunking.orchestrator import MAX_CHUNK_SIZE
from src.retrieval.bm25 import BM25Parameters


CHUNKER_VERSION = 1
TOKENIZER_VERSION = 1


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Declare configurable and versioned BM25 index build inputs."""

    max_chunk_size: int = MAX_CHUNK_SIZE
    parameters: BM25Parameters = field(default_factory=BM25Parameters)
    chunker_version: int = CHUNKER_VERSION
    tokenizer_version: int = TOKENIZER_VERSION

    def __post_init__(self) -> None:
        """Reject configuration values unsupported by the current pipeline."""
        if not 0 < self.max_chunk_size <= MAX_CHUNK_SIZE:
            raise ValueError(
                f"Maximum chunk size must be between 1 and {MAX_CHUNK_SIZE}"
            )
        if self.chunker_version <= 0:
            raise ValueError("Chunker version must be positive")
        if self.tokenizer_version <= 0:
            raise ValueError("Tokenizer version must be positive")


def fingerprint_pipeline(
    config: PipelineConfig,
    *,
    index_schema_version: int,
) -> str:
    """Hash the canonical declared inputs for one index build."""
    if index_schema_version <= 0:
        raise ValueError("Index schema version must be positive")
    payload = {
        "chunker_version": config.chunker_version,
        "index_schema_version": index_schema_version,
        "max_chunk_size": config.max_chunk_size,
        "parameters": asdict(config.parameters),
        "tokenizer_version": config.tokenizer_version,
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(canonical_json.encode("utf-8")).hexdigest()
