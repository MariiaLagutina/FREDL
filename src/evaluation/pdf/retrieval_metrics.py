"""Score ranked PDF retrieval results against human relevance judgments."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from src.evaluation.pdf.judgments import (
    PdfJudgmentSet,
    PdfRecordRelevance,
)
from src.evaluation.pdf.models import (
    PdfGoldenAnswerability,
    PdfGoldenDataset,
)
from src.ingestion.pdf.index_records import PdfIndexRecord


@dataclass(frozen=True)
class PdfRetrievalResult:
    """Store one retriever's ranked records and abstention decision."""

    case_id: str
    ranked_record_ids: tuple[str, ...]
    context_record_ids: tuple[str, ...] = ()
    abstained: bool = False


@dataclass(frozen=True)
class PdfRetrievalCaseScore:
    """Describe retrieval quality for one golden question."""

    case_id: str
    answerability: PdfGoldenAnswerability
    required_total: int
    required_retrieved_at_k: int
    required_recall_at_k: float | None
    all_required_at_k: bool | None
    required_evidence_total: int
    required_evidence_covered_at_k: int
    required_evidence_coverage_at_k: float | None
    all_required_evidence_at_k: bool | None
    first_required_rank: int | None
    reciprocal_rank: float | None
    judged_irrelevant_at_k: int
    unjudged_at_k: int
    abstention_correct: bool


@dataclass(frozen=True)
class PdfRetrievalEvaluation:
    """Aggregate interpretable retrieval metrics across golden questions."""

    top_k: int
    case_scores: tuple[PdfRetrievalCaseScore, ...]
    mean_required_recall_at_k: float
    all_required_at_k_rate: float
    mean_required_evidence_coverage_at_k: float
    all_required_evidence_at_k_rate: float
    mean_reciprocal_rank: float
    negative_abstention_accuracy: float


def evaluate_pdf_retrieval(
    dataset: PdfGoldenDataset,
    judgment_set: PdfJudgmentSet,
    records: Sequence[PdfIndexRecord],
    results: Sequence[PdfRetrievalResult],
    *,
    top_k: int = 5,
) -> PdfRetrievalEvaluation:
    """Score every case without guessing the relevance of unjudged records."""
    if top_k <= 0:
        raise ValueError("PDF retrieval evaluation top_k must be positive")

    cases = {case.id: case for case in dataset.cases}
    results_by_case = _index_results(results)
    if set(results_by_case) != set(cases):
        raise ValueError("PDF retrieval results must cover every golden case")

    record_ids = {record.record_id for record in records}
    judgments_by_case = _index_judgments(judgment_set)
    scores = tuple(
        _score_case(
            case_id,
            cases[case_id].answerability,
            results_by_case[case_id],
            judgments_by_case.get(case_id, {}),
            record_ids,
            top_k,
        )
        for case_id in cases
    )

    answerable = tuple(
        score
        for score in scores
        if score.answerability is not PdfGoldenAnswerability.NOT_FOUND
    )
    negatives = tuple(
        score
        for score in scores
        if score.answerability is PdfGoldenAnswerability.NOT_FOUND
    )
    return PdfRetrievalEvaluation(
        top_k=top_k,
        case_scores=scores,
        mean_required_recall_at_k=_mean(
            score.required_recall_at_k for score in answerable
        ),
        all_required_at_k_rate=_mean(
            float(score.all_required_at_k is True)
            for score in answerable
        ),
        mean_required_evidence_coverage_at_k=_mean(
            score.required_evidence_coverage_at_k
            for score in answerable
        ),
        all_required_evidence_at_k_rate=_mean(
            float(score.all_required_evidence_at_k is True)
            for score in answerable
        ),
        mean_reciprocal_rank=_mean(
            score.reciprocal_rank for score in answerable
        ),
        negative_abstention_accuracy=_mean(
            float(score.abstention_correct) for score in negatives
        ),
    )


def _index_results(
    results: Sequence[PdfRetrievalResult],
) -> dict[str, PdfRetrievalResult]:
    indexed = {result.case_id: result for result in results}
    if len(indexed) != len(results):
        raise ValueError("PDF retrieval results must have unique case IDs")
    return indexed


def _index_judgments(
    judgment_set: PdfJudgmentSet,
) -> dict[str, dict[str, tuple[PdfRecordRelevance, frozenset[int]]]]:
    priority = {
        PdfRecordRelevance.IRRELEVANT: 0,
        PdfRecordRelevance.SUPPORTING: 1,
        PdfRecordRelevance.REQUIRED: 2,
    }
    indexed: dict[
        str,
        dict[str, tuple[PdfRecordRelevance, frozenset[int]]],
    ] = {}
    for judgment in judgment_set.judgments:
        records = indexed.setdefault(judgment.case_id, {})
        current = records.get(judgment.record_id)
        evidence_indexes = frozenset({judgment.evidence_index})
        if current is None:
            records[judgment.record_id] = (
                judgment.relevance,
                evidence_indexes,
            )
            continue
        relevance, current_evidence_indexes = current
        records[judgment.record_id] = (
            max(
                (relevance, judgment.relevance),
                key=priority.__getitem__,
            ),
            current_evidence_indexes.union(evidence_indexes),
        )
    return indexed


def _score_case(
    case_id: str,
    answerability: PdfGoldenAnswerability,
    result: PdfRetrievalResult,
    judgments: dict[
        str,
        tuple[PdfRecordRelevance, frozenset[int]],
    ],
    known_record_ids: set[str],
    top_k: int,
) -> PdfRetrievalCaseScore:
    if len(result.ranked_record_ids) != len(set(result.ranked_record_ids)):
        raise ValueError("Ranked PDF retrieval records must be unique")
    if not set(result.ranked_record_ids).issubset(known_record_ids):
        raise ValueError("PDF retrieval result references an unknown record")
    if len(result.context_record_ids) != len(set(result.context_record_ids)):
        raise ValueError("PDF retrieval context records must be unique")
    if not set(result.context_record_ids).issubset(known_record_ids):
        raise ValueError("PDF retrieval context references an unknown record")

    ranked = result.ranked_record_ids[:top_k]
    evidence_records = set(ranked).union(result.context_record_ids)
    required = {
        record_id
        for record_id, (relevance, _) in judgments.items()
        if relevance is PdfRecordRelevance.REQUIRED
    }
    retrieved_required = required.intersection(ranked)
    required_evidence = {
        evidence_index
        for relevance, evidence_indexes in judgments.values()
        if relevance is PdfRecordRelevance.REQUIRED
        for evidence_index in evidence_indexes
    }
    covered_required_evidence = {
        evidence_index
        for record_id in evidence_records
        for relevance, evidence_indexes in (
            judgments.get(record_id, (None, frozenset())),
        )
        if relevance is PdfRecordRelevance.REQUIRED
        for evidence_index in evidence_indexes
    }
    first_required_rank = next(
        (
            rank
            for rank, record_id in enumerate(result.ranked_record_ids, start=1)
            if record_id in required
        ),
        None,
    )
    is_negative = answerability is PdfGoldenAnswerability.NOT_FOUND
    required_recall = (
        None if is_negative else len(retrieved_required) / len(required)
    )
    return PdfRetrievalCaseScore(
        case_id=case_id,
        answerability=answerability,
        required_total=len(required),
        required_retrieved_at_k=len(retrieved_required),
        required_recall_at_k=required_recall,
        all_required_at_k=(
            None if is_negative else required == retrieved_required
        ),
        required_evidence_total=len(required_evidence),
        required_evidence_covered_at_k=len(covered_required_evidence),
        required_evidence_coverage_at_k=(
            None
            if is_negative
            else len(covered_required_evidence) / len(required_evidence)
        ),
        all_required_evidence_at_k=(
            None
            if is_negative
            else required_evidence == covered_required_evidence
        ),
        first_required_rank=first_required_rank,
        reciprocal_rank=(
            None
            if is_negative
            else (
                0.0
                if first_required_rank is None
                else 1 / first_required_rank
            )
        ),
        judged_irrelevant_at_k=sum(
            judgments.get(record_id, (None, frozenset()))[0]
            is PdfRecordRelevance.IRRELEVANT
            for record_id in ranked
        ),
        unjudged_at_k=sum(
            record_id not in judgments for record_id in ranked
        ),
        abstention_correct=(
            result.abstained if is_negative else not result.abstained
        ),
    )


def _mean(values: Iterable[float | None]) -> float:
    present = tuple(value for value in values if value is not None)
    return sum(present) / len(present) if present else 0.0
