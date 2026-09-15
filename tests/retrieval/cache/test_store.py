"""Tests for exact search-result cache compatibility."""

import json
from pathlib import Path

import pytest

from src.retrieval.cache import SearchCacheKey, SearchResultCache
from src.retrieval.cache.store import normalize_question
from src.models import MinimalSource


def _key(**changes: object) -> SearchCacheKey:
    values: dict[str, object] = {
        "question": normalize_question("  How does the CACHE work? "),
        "corpus_fingerprint": "a" * 64,
        "pipeline_fingerprint": "b" * 64,
        "k": 5,
        "identifier_match_weight": 0.0,
        "identifier_candidate_depth": 0,
        "auxiliary_path_penalty": 0.5,
        "path_candidate_depth": 20,
    }
    values.update(changes)
    return SearchCacheKey.model_validate(values)


def _sources() -> list[MinimalSource]:
    return [
        MinimalSource(
            file_path="docs/cache.md",
            first_character_index=0,
            last_character_index=20,
        )
    ]


def test_absent_cache_is_a_miss(tmp_path: Path) -> None:
    cache = SearchResultCache(tmp_path / "search.json")

    assert cache.get(_key()) is None


def test_search_result_round_trips(tmp_path: Path) -> None:
    cache = SearchResultCache(tmp_path / "search.json")
    expected = _sources()

    cache.put(_key(), expected)

    assert cache.get(_key()) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("question", "another question"),
        ("corpus_fingerprint", "c" * 64),
        ("pipeline_fingerprint", "d" * 64),
        ("k", 10),
        ("identifier_match_weight", 0.3),
        ("identifier_candidate_depth", 10),
        ("auxiliary_path_penalty", 0.6),
        ("path_candidate_depth", 10),
    ],
)
def test_changed_search_input_is_a_miss(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    cache = SearchResultCache(tmp_path / "search.json")
    cache.put(_key(), _sources())

    assert cache.get(_key(**{field: value})) is None


def test_question_normalization_ignores_case_and_whitespace() -> None:
    assert normalize_question(" HOW   does Cache work?\n") == (
        "how does cache work?"
    )


def test_rejects_invalid_storage(tmp_path: Path) -> None:
    path = tmp_path / "search.json"
    cache = SearchResultCache(path)
    path.write_text("not JSON", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid"):
        cache.get(_key())


def test_rejects_duplicate_storage_keys(tmp_path: Path) -> None:
    path = tmp_path / "search.json"
    cache = SearchResultCache(path)
    cache.put(_key(), _sources())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"].append(payload["entries"][0])
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate keys"):
        cache.get(_key())


def test_rejects_incompatible_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "search.json"
    cache = SearchResultCache(path)
    cache.put(_key(), _sources())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible"):
        cache.get(_key())


def test_stored_result_is_unaffected_by_later_mutation(
    tmp_path: Path,
) -> None:
    cache = SearchResultCache(tmp_path / "search.json")
    sources = _sources()
    cache.put(_key(), sources)

    sources.append(
        MinimalSource(
            file_path="docs/other.md",
            first_character_index=0,
            last_character_index=10,
        )
    )

    assert cache.get(_key()) == _sources()
