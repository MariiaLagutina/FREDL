"""Tests for exact validated-answer cache compatibility."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.generation.cache import AnswerCacheKey, ValidatedAnswerCache
from src.generation.cache.store import normalize_question
from src.generation.query import QueryAnswerResult
from src.models import MinimalSource


def _key(**changes: object) -> AnswerCacheKey:
    values: dict[str, object] = {
        "question": normalize_question("  How does the CACHE work? "),
        "corpus_fingerprint": "a" * 64,
        "pipeline_fingerprint": "b" * 64,
        "k": 5,
        "context_token_budget": 2048,
        "auxiliary_path_penalty": 0.5,
        "path_candidate_depth": 20,
        "prompt_version": "v1",
        "model_name": "Qwen/Qwen3-0.6B",
        "device": "cpu",
        "max_new_tokens": 256,
        "enable_thinking": False,
        "do_sample": False,
    }
    values.update(changes)
    return AnswerCacheKey.model_validate(values)


def _result() -> QueryAnswerResult:
    source = MinimalSource(
        file_path="docs/cache.md",
        first_character_index=0,
        last_character_index=20,
    )
    return QueryAnswerResult(
        answer="It stores validated answers. [Source 1]",
        retrieved_sources=(source,),
        context_sources=(source,),
        used_context_tokens=12,
        skipped_source_count=0,
        prompt_version="v1",
    )


def test_absent_cache_is_a_miss(tmp_path: Path) -> None:
    cache = ValidatedAnswerCache(tmp_path / "answers.json")

    assert cache.get(_key()) is None


def test_validated_answer_round_trips(tmp_path: Path) -> None:
    cache = ValidatedAnswerCache(tmp_path / "answers.json")
    expected = _result()

    cache.put(_key(), expected)

    assert cache.get(_key()) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("question", "another question"),
        ("corpus_fingerprint", "c" * 64),
        ("pipeline_fingerprint", "d" * 64),
        ("k", 10),
        ("context_token_budget", 1024),
        ("auxiliary_path_penalty", 0.6),
        ("path_candidate_depth", 10),
        ("prompt_version", "v2"),
        ("model_name", "another/model"),
        ("device", "cuda"),
        ("max_new_tokens", 128),
        ("enable_thinking", True),
        ("do_sample", True),
    ],
)
def test_changed_answer_input_is_a_miss(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    cache = ValidatedAnswerCache(tmp_path / "answers.json")
    cache.put(_key(), _result())

    assert cache.get(_key(**{field: value})) is None


def test_question_normalization_ignores_case_and_whitespace() -> None:
    assert normalize_question(" HOW   does Cache work?\n") == (
        "how does cache work?"
    )


def test_rejects_prompt_version_mismatch(tmp_path: Path) -> None:
    cache = ValidatedAnswerCache(tmp_path / "answers.json")

    with pytest.raises(ValueError, match="prompt version"):
        cache.put(_key(prompt_version="v2"), _result())


def test_rejects_invalid_storage(tmp_path: Path) -> None:
    path = tmp_path / "answers.json"
    cache = ValidatedAnswerCache(path)
    path.write_text("not JSON", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid"):
        cache.get(_key())


def test_rejects_duplicate_storage_keys(tmp_path: Path) -> None:
    path = tmp_path / "answers.json"
    cache = ValidatedAnswerCache(path)
    cache.put(_key(), _result())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["entries"].append(payload["entries"][0])
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate keys"):
        cache.get(_key())


def test_result_is_immutable_after_storage(tmp_path: Path) -> None:
    cache = ValidatedAnswerCache(tmp_path / "answers.json")
    result = _result()
    cache.put(_key(), result)

    assert replace(result, answer="changed") != cache.get(_key())
