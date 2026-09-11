"""Tests for deterministic binary evidence review candidates."""

from pathlib import Path

import pytest

from src.evaluation.retrieval import RetrievalEvaluationCase
from src.generation.evidence import (
    EvidenceCandidateOrigin,
    EvidenceDatasetSplit,
    EvidenceDecision,
    EvidenceExampleKind,
    build_evidence_review_queue,
    save_evidence_review_queue,
)
from src.models import MinimalSource


def _source(path: str, start: int, end: int) -> MinimalSource:
    return MinimalSource(
        file_path=path,
        first_character_index=start,
        last_character_index=end,
    )


def _case(question_id: str) -> RetrievalEvaluationCase:
    reference = _source("data/raw/docs.md", 0, 6)
    overlapping = _source("data/raw/docs.md", 0, 5)
    negative = _source("data/raw/docs.md", 7, 16)
    return RetrievalEvaluationCase(
        question_id=question_id,
        question="Which policy is used?",
        references=(reference,),
        retrieved=(negative, overlapping, reference, negative),
    )


def _roots(tmp_path: Path) -> tuple[Path, Path]:
    project_root = tmp_path / "project"
    corpus_root = project_root / "data/raw"
    corpus_root.mkdir(parents=True)
    (corpus_root / "docs.md").write_text(
        "answer near-miss",
        encoding="utf-8",
    )
    return project_root, corpus_root


def _answers(*question_ids: str) -> dict[str, str]:
    return {
        question_id: "The expected answer."
        for question_id in question_ids
    }


def test_builder_proposes_reviewable_labels_without_duplicates(
    tmp_path: Path,
) -> None:
    """References lead and repeated retrieval locations appear once."""
    project_root, corpus_root = _roots(tmp_path)

    queue = build_evidence_review_queue(
        (_case("question-1"),),
        _answers("question-1"),
        project_root,
        corpus_root,
    )

    assert len(queue.candidates) == 3
    positive, negative, overlapping = queue.candidates
    assert positive.source_text == "answer"
    assert positive.reference_answer == "The expected answer."
    assert positive.suggested_decision is EvidenceDecision.ANSWERS
    assert positive.suggested_kind is EvidenceExampleKind.DIRECT_ANSWER
    assert positive.origin is EvidenceCandidateOrigin.GROUND_TRUTH
    assert overlapping.source_text == "answe"
    assert overlapping.suggested_decision is EvidenceDecision.ANSWERS
    assert overlapping.origin is EvidenceCandidateOrigin.RETRIEVAL_MATCH
    assert negative.source_text == "near-miss"
    assert negative.suggested_decision is EvidenceDecision.DOES_NOT_ANSWER
    assert negative.origin is EvidenceCandidateOrigin.RETRIEVAL_NON_MATCH


def test_builder_keeps_question_groups_in_one_stable_split(
    tmp_path: Path,
) -> None:
    """Source rows cannot leak one question across learning splits."""
    project_root, corpus_root = _roots(tmp_path)

    first = build_evidence_review_queue(
        (_case("question-1"), _case("question-2")),
        _answers("question-1", "question-2"),
        project_root,
        corpus_root,
        split_seed="fixed",
    )
    second = build_evidence_review_queue(
        (_case("question-1"), _case("question-2")),
        _answers("question-1", "question-2"),
        project_root,
        corpus_root,
        split_seed="fixed",
    )

    assert first == second
    split_by_question: dict[str, set[EvidenceDatasetSplit]] = {}
    for candidate in first.candidates:
        split_by_question.setdefault(candidate.question_id, set()).add(
            candidate.split
        )
    assert all(len(splits) == 1 for splits in split_by_question.values())


def test_builder_excludes_protected_evaluation_questions(
    tmp_path: Path,
) -> None:
    """Held-out cases never become pending learning candidates."""
    project_root, corpus_root = _roots(tmp_path)

    queue = build_evidence_review_queue(
        (_case("learn"), _case("held-out")),
        _answers("learn", "held-out"),
        project_root,
        corpus_root,
        protected_question_ids={"held-out"},
    )

    assert {item.question_id for item in queue.candidates} == {"learn"}


def test_builder_rejects_invalid_source_range(tmp_path: Path) -> None:
    """A stale ground-truth span cannot silently enter review."""
    project_root, corpus_root = _roots(tmp_path)
    invalid = RetrievalEvaluationCase(
        question_id="invalid",
        question="Question?",
        references=(_source("data/raw/docs.md", 0, 100),),
        retrieved=(),
    )

    with pytest.raises(ValueError, match="source range is invalid"):
        build_evidence_review_queue(
            (invalid,),
            _answers("invalid"),
            project_root,
            corpus_root,
        )


@pytest.mark.parametrize("fraction", [-0.1, 0, 1, 1.1, float("nan")])
def test_builder_rejects_invalid_validation_fraction(
    tmp_path: Path,
    fraction: float,
) -> None:
    """A usable deterministic split needs a real interior fraction."""
    project_root, corpus_root = _roots(tmp_path)

    with pytest.raises(ValueError, match="between zero and one"):
        build_evidence_review_queue(
            (),
            {},
            project_root,
            corpus_root,
            validation_fraction=fraction,
        )


def test_writer_serializes_deterministic_review_json(tmp_path: Path) -> None:
    """Generated review data remains an explicit external artifact."""
    project_root, corpus_root = _roots(tmp_path)
    queue = build_evidence_review_queue(
        (_case("question-1"),),
        _answers("question-1"),
        project_root,
        corpus_root,
    )
    output_path = tmp_path / "output/review.json"

    save_evidence_review_queue(queue, output_path)

    content = output_path.read_text(encoding="utf-8")
    assert content.endswith("\n")
    assert '"format_version": "1"' in content
    assert '"suggested_decision": "ANSWERS"' in content


def test_builder_rejects_answer_identity_mismatch(tmp_path: Path) -> None:
    """Review evidence cannot silently join to another reference answer."""
    project_root, corpus_root = _roots(tmp_path)

    with pytest.raises(ValueError, match="must contain the same IDs"):
        build_evidence_review_queue(
            (_case("question-1"),),
            {"other-question": "Answer."},
            project_root,
            corpus_root,
        )
