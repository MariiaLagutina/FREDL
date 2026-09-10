"""Persist complete answer-quality diagnostic evidence as JSON."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from src.evaluation.answer_quality.models import (
    AnswerQualityCaseEvidence,
    AnswerQualityDiagnosticReport,
    GenerationAttemptEvidence,
)


SCHEMA_VERSION = 1


def write_diagnostic_report(
    report: AnswerQualityDiagnosticReport,
    output_path: Path,
) -> None:
    """Atomically write one reviewable diagnostic report as UTF-8 JSON."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": report.model_name,
        "device": report.device,
        "prompt_version": report.prompt_version,
        "max_new_tokens": report.max_new_tokens,
        "context_token_budget": report.context_token_budget,
        "search_k": report.search_k,
        "question_count": len(report.cases),
        "passed_count": report.passed_count,
        "failed_count": report.failed_count,
        "cases": [_serialize_case(case) for case in report.cases],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def _serialize_case(case: AnswerQualityCaseEvidence) -> dict[str, Any]:
    """Convert one case and its raw attempts to JSON-ready values."""
    return {
        "question_id": case.question_id,
        "question": case.question,
        "passed_validation": case.passed_validation,
        "retrieved_sources": [
            source.model_dump() for source in case.retrieved_sources
        ],
        "context_sources": [
            source.model_dump() for source in case.context_sources
        ],
        "used_context_tokens": case.used_context_tokens,
        "skipped_source_count": case.skipped_source_count,
        "attempts": [
            _serialize_attempt(attempt) for attempt in case.attempts
        ],
        "accepted_answer": case.accepted_answer,
        "error_type": case.error_type,
        "error": case.error,
    }


def _serialize_attempt(
    attempt: GenerationAttemptEvidence,
) -> dict[str, object]:
    """Convert one unvalidated model response to JSON-ready values."""
    return {
        "attempt": attempt.attempt,
        "answer": attempt.answer,
    }
