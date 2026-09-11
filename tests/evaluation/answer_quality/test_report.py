"""Tests for persisted answer-quality diagnostic evidence."""

import json
from pathlib import Path

from src.evaluation.answer_quality import (
    AnswerQualityCaseEvidence,
    AnswerQualityDiagnosticReport,
    GenerationAttemptEvidence,
    write_diagnostic_report,
)
from src.models import MinimalSource


def test_write_diagnostic_report_preserves_review_evidence(
    tmp_path: Path,
) -> None:
    """The JSON report keeps raw attempts, outcomes, and run settings."""
    source = MinimalSource(
        file_path="data/raw/cache.md",
        first_character_index=0,
        last_character_index=20,
    )
    report = AnswerQualityDiagnosticReport(
        model_name="Qwen/test",
        device="cuda",
        prompt_version="grounded-v1",
        max_new_tokens=128,
        context_token_budget=1000,
        search_k=1,
        cases=(
            AnswerQualityCaseEvidence(
                question_id="q-1",
                question="Which cache policy is used?",
                retrieved_sources=(source,),
                context_sources=(source,),
                used_context_tokens=12,
                skipped_source_count=0,
                attempts=(
                    GenerationAttemptEvidence(1, "It uses LRU."),
                    GenerationAttemptEvidence(
                        2,
                        "It uses LRU. [Source 1]",
                    ),
                ),
                accepted_answer="It uses LRU. [Source 1]",
                error_type=None,
                error=None,
            ),
        ),
    )
    output_path = tmp_path / "reports" / "quality.json"

    write_diagnostic_report(report, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["question_count"] == 1
    assert payload["passed_count"] == 1
    assert payload["failed_count"] == 0
    assert payload["model"] == "Qwen/test"
    assert payload["device"] == "cuda"
    assert payload["cases"][0]["passed_validation"] is True
    assert payload["cases"][0]["attempts"] == [
        {"answer": "It uses LRU.", "attempt": 1},
        {"answer": "It uses LRU. [Source 1]", "attempt": 2},
    ]
    assert payload["cases"][0]["context_sources"] == [
        source.model_dump()
    ]
    assert not output_path.with_suffix(".json.tmp").exists()
