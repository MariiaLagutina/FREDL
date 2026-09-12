"""Read and atomically update a local validated-answer cache."""

from hashlib import sha256
from pathlib import Path

from pydantic import ValidationError

from src.generation.cache.models import (
    AnswerCacheKey,
    StoredAnswerCache,
    StoredValidatedAnswer,
)
from src.generation.query.models import QueryAnswerResult

CACHE_SCHEMA_VERSION = 1


class ValidatedAnswerCache:
    """Store validated answers and require exact compatibility on lookup."""

    def __init__(self, path: Path) -> None:
        """Keep the explicit cache path without touching the filesystem."""
        self._path = path

    def get(self, key: AnswerCacheKey) -> QueryAnswerResult | None:
        """Return an exact compatible hit or report a cache miss."""
        entries = self._load_entries()
        entry = entries.get(_key_digest(key))
        if entry is None or entry.key != key:
            return None
        return QueryAnswerResult(
            answer=entry.answer,
            retrieved_sources=entry.retrieved_sources,
            context_sources=entry.context_sources,
            used_context_tokens=entry.used_context_tokens,
            skipped_source_count=entry.skipped_source_count,
            prompt_version=entry.key.prompt_version,
        )

    def put(self, key: AnswerCacheKey, result: QueryAnswerResult) -> None:
        """Atomically replace the entry for one exact compatibility key."""
        if result.prompt_version != key.prompt_version:
            raise ValueError("Answer prompt version does not match cache key")
        entries = self._load_entries()
        entries[_key_digest(key)] = StoredValidatedAnswer(
            key=key,
            answer=result.answer,
            retrieved_sources=result.retrieved_sources,
            context_sources=result.context_sources,
            used_context_tokens=result.used_context_tokens,
            skipped_source_count=result.skipped_source_count,
        )
        snapshot = StoredAnswerCache(
            schema_version=CACHE_SCHEMA_VERSION,
            entries=tuple(entries[digest] for digest in sorted(entries)),
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary_path.write_text(
            snapshot.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self._path)

    def _load_entries(self) -> dict[str, StoredValidatedAnswer]:
        """Load and validate entries, treating an absent file as empty."""
        if not self._path.exists():
            return {}
        try:
            snapshot = StoredAnswerCache.model_validate_json(
                self._path.read_text(encoding="utf-8")
            )
        except (OSError, ValidationError) as error:
            raise ValueError("Stored answer cache is invalid") from error
        if snapshot.schema_version != CACHE_SCHEMA_VERSION:
            raise ValueError("Stored answer cache schema is incompatible")
        entries = {_key_digest(entry.key): entry for entry in snapshot.entries}
        if len(entries) != len(snapshot.entries):
            raise ValueError("Stored answer cache contains duplicate keys")
        return entries


def normalize_question(question: str) -> str:
    """Canonicalize insignificant whitespace and letter case."""
    normalized = " ".join(question.split()).casefold()
    if not normalized:
        raise ValueError("Question must not be empty")
    return normalized


def _key_digest(key: AnswerCacheKey) -> str:
    """Build a stable lookup identifier from canonical key JSON."""
    payload = key.model_dump_json()
    return sha256(payload.encode("utf-8")).hexdigest()
