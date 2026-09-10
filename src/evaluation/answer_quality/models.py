"""Immutable evidence models for manual answer-quality review."""

from dataclasses import dataclass

from src.models import MinimalSource


@dataclass(frozen=True, slots=True)
class GenerationAttemptEvidence:
    """Preserve one raw model response before deterministic validation."""

    attempt: int
    answer: str


@dataclass(frozen=True, slots=True)
class AnswerQualityCaseEvidence:
    """Keep one question's complete generation and failure evidence."""

    question_id: str
    question: str
    retrieved_sources: tuple[MinimalSource, ...]
    context_sources: tuple[MinimalSource, ...]
    used_context_tokens: int
    skipped_source_count: int
    attempts: tuple[GenerationAttemptEvidence, ...]
    accepted_answer: str | None
    error_type: str | None
    error: str | None

    @property
    def passed_validation(self) -> bool:
        """Return whether deterministic validation accepted an answer."""
        return self.accepted_answer is not None


@dataclass(frozen=True, slots=True)
class AnswerQualityDiagnosticReport:
    """Describe one reproducible dataset generation diagnostic run."""

    model_name: str
    device: str
    prompt_version: str
    max_new_tokens: int
    context_token_budget: int
    search_k: int
    cases: tuple[AnswerQualityCaseEvidence, ...]

    @property
    def passed_count(self) -> int:
        """Count cases accepted by deterministic validation."""
        return sum(case.passed_validation for case in self.cases)

    @property
    def failed_count(self) -> int:
        """Count cases retained as controlled diagnostic failures."""
        return len(self.cases) - self.passed_count
