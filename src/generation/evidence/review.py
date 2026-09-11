"""Build a human-review queue for binary evidence training examples."""

from collections.abc import Mapping, Sequence
from enum import Enum
import hashlib
import math
from pathlib import Path
from typing import AbstractSet

from pydantic import BaseModel, ConfigDict

from src.evaluation.retrieval import RetrievalEvaluationCase, sources_match
from src.generation.context import load_ranked_source_texts
from src.generation.evidence.dataset import (
    EvidenceDatasetSplit,
    EvidenceExampleKind,
)
from src.generation.evidence.models import EvidenceDecision
from src.models import MinimalSource


class EvidenceCandidateOrigin(str, Enum):
    """Identify why a source became a proposed training example."""

    GROUND_TRUTH = "ground_truth"
    RETRIEVAL_MATCH = "retrieval_match"
    RETRIEVAL_NON_MATCH = "retrieval_non_match"


class EvidenceReviewCandidate(BaseModel):
    """Represent one unapproved label proposal for human review."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    question_id: str
    source_id: str
    question: str
    reference_answer: str
    source_text: str
    suggested_decision: EvidenceDecision
    suggested_kind: EvidenceExampleKind
    origin: EvidenceCandidateOrigin
    split: EvidenceDatasetSplit


class EvidenceReviewQueue(BaseModel):
    """Contain deterministic pending candidates outside training data."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format_version: str = "1"
    candidates: tuple[EvidenceReviewCandidate, ...]


def build_evidence_review_queue(
    cases: Sequence[RetrievalEvaluationCase],
    reference_answers: Mapping[str, str],
    project_root: Path,
    corpus_root: Path,
    *,
    protected_question_ids: AbstractSet[str] = frozenset(),
    validation_fraction: float = 0.2,
    split_seed: str = "evidence-v1",
) -> EvidenceReviewQueue:
    """Propose reference positives and retrieval hard negatives."""
    if (
        not math.isfinite(validation_fraction)
        or validation_fraction <= 0
        or validation_fraction >= 1
    ):
        raise ValueError("Validation fraction must be between zero and one")
    if not split_seed:
        raise ValueError("Evidence split seed must not be empty")
    case_ids = [case.question_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Evidence review question IDs must be unique")
    if set(reference_answers) != set(case_ids):
        raise ValueError(
            "Evidence review cases and answers must contain the same IDs"
        )
    if any(not answer.strip() for answer in reference_answers.values()):
        raise ValueError("Evidence review answers must not be empty")

    candidates: list[EvidenceReviewCandidate] = []
    for case in cases:
        if case.question_id in protected_question_ids:
            continue
        split = _split_for_question(
            case.question_id,
            validation_fraction,
            split_seed,
        )
        reference_identities = {
            _source_identity(source) for source in case.references
        }
        ordered_sources = _unique_sources(
            (*case.references, *case.retrieved)
        )
        documents = load_ranked_source_texts(
            project_root,
            corpus_root,
            ordered_sources,
        )
        for source in ordered_sources:
            identity = _source_identity(source)
            is_reference = identity in reference_identities
            is_retrieval_match = (
                not is_reference
                and any(
                    sources_match(source, reference)
                    for reference in case.references
                )
            )
            suggested_answers = is_reference or is_retrieval_match
            source_id = _format_source_id(source)
            candidates.append(
                EvidenceReviewCandidate(
                    candidate_id=_candidate_id(
                        case.question_id,
                        source_id,
                    ),
                    question_id=case.question_id,
                    source_id=source_id,
                    question=case.question,
                    reference_answer=reference_answers[case.question_id],
                    source_text=_exact_source_text(
                        source,
                        documents[source.file_path],
                    ),
                    suggested_decision=(
                        EvidenceDecision.ANSWERS
                        if suggested_answers
                        else EvidenceDecision.DOES_NOT_ANSWER
                    ),
                    suggested_kind=(
                        EvidenceExampleKind.DIRECT_ANSWER
                        if suggested_answers
                        else EvidenceExampleKind.TOPICAL_NEAR_MISS
                    ),
                    origin=(
                        EvidenceCandidateOrigin.GROUND_TRUTH
                        if is_reference
                        else (
                            EvidenceCandidateOrigin.RETRIEVAL_MATCH
                            if is_retrieval_match
                            else EvidenceCandidateOrigin.RETRIEVAL_NON_MATCH
                        )
                    ),
                    split=split,
                )
            )
    return EvidenceReviewQueue(candidates=tuple(candidates))


def save_evidence_review_queue(
    queue: EvidenceReviewQueue,
    output_path: Path,
) -> None:
    """Write one deterministic UTF-8 JSON review artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        queue.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )


def _split_for_question(
    question_id: str,
    validation_fraction: float,
    split_seed: str,
) -> EvidenceDatasetSplit:
    """Assign every source for one question to the same stable split."""
    digest = hashlib.sha256(
        f"{split_seed}:{question_id}".encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    if bucket < validation_fraction:
        return EvidenceDatasetSplit.VALIDATION
    return EvidenceDatasetSplit.TRAIN


def _unique_sources(
    sources: Sequence[MinimalSource],
) -> tuple[MinimalSource, ...]:
    """Preserve first occurrence of each exact source location."""
    unique: list[MinimalSource] = []
    seen: set[tuple[str, int, int]] = set()
    for source in sources:
        identity = _source_identity(source)
        if identity in seen:
            continue
        seen.add(identity)
        unique.append(source)
    return tuple(unique)


def _source_identity(source: MinimalSource) -> tuple[str, int, int]:
    return (
        source.file_path,
        source.first_character_index,
        source.last_character_index,
    )


def _format_source_id(source: MinimalSource) -> str:
    return (
        f"{source.file_path}:{source.first_character_index}:"
        f"{source.last_character_index}"
    )


def _candidate_id(question_id: str, source_id: str) -> str:
    digest = hashlib.sha256(
        f"{question_id}:{source_id}".encode("utf-8")
    ).hexdigest()
    return digest[:20]


def _exact_source_text(source: MinimalSource, document: str) -> str:
    start = source.first_character_index
    end = source.last_character_index
    if start < 0 or end <= start or end > len(document):
        raise ValueError("Evidence review source range is invalid")
    return document[start:end]
