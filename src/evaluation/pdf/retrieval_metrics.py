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
    """Describe retrieval quality for one golden question.

    Record recall uses each record's strongest label in the case. Evidence
    coverage counts an evidence unit only when a record is required for
    that unit: ranked coverage uses the top-k ranking alone, context
    coverage also includes the neighbour context records.
    """

    case_id: str
    answerability: PdfGoldenAnswerability
    required_total: int
    required_retrieved_at_k: int
    required_recall_at_k: float | None
    all_required_at_k: bool | None
    required_evidence_total: int
    ranked_required_evidence_covered_at_k: int
    ranked_required_evidence_coverage_at_k: float | None
    all_ranked_required_evidence_at_k: bool | None
    context_required_evidence_covered_at_k: int
    context_required_evidence_coverage_at_k: float | None
    all_context_required_evidence_at_k: bool | None
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
    mean_ranked_required_evidence_coverage_at_k: float
    all_ranked_required_evidence_at_k_rate: float
    mean_context_required_evidence_coverage_at_k: float
    all_context_required_evidence_at_k_rate: float
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
        mean_ranked_required_evidence_coverage_at_k=_mean(
            score.ranked_required_evidence_coverage_at_k
            for score in answerable
        ),
        all_ranked_required_evidence_at_k_rate=_mean(
            float(score.all_ranked_required_evidence_at_k is True)
            for score in answerable
        ),
        mean_context_required_evidence_coverage_at_k=_mean(
            score.context_required_evidence_coverage_at_k
            for score in answerable
        ),
        all_context_required_evidence_at_k_rate=_mean(
            float(score.all_context_required_evidence_at_k is True)
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


_EvidenceLabels = dict[str, dict[int, PdfRecordRelevance]]

_PRIORITY = {
    PdfRecordRelevance.IRRELEVANT: 0,
    PdfRecordRelevance.SUPPORTING: 1,
    PdfRecordRelevance.REQUIRED: 2,
}


def _index_judgments(
    judgment_set: PdfJudgmentSet,
) -> dict[str, _EvidenceLabels]:
    """Keep each label at the evidence unit it was given for."""
    indexed: dict[str, _EvidenceLabels] = {}
    for judgment in judgment_set.judgments:
        records = indexed.setdefault(judgment.case_id, {})
        records.setdefault(judgment.record_id, {})[
            judgment.evidence_index
        ] = judgment.relevance
    return indexed


def _record_relevance(
    labels: dict[int, PdfRecordRelevance],
) -> PdfRecordRelevance:
    return max(labels.values(), key=_PRIORITY.__getitem__)


def _required_evidence(
    judgments: _EvidenceLabels,
    record_ids: Iterable[str],
) -> set[int]:
    return {
        evidence_index
        for record_id in record_ids
        for evidence_index, relevance in judgments.get(record_id, {}).items()
        if relevance is PdfRecordRelevance.REQUIRED
    }


def _coverage(
    covered: set[int],
    required: set[int],
    is_negative: bool,
) -> tuple[float | None, bool | None]:
    if is_negative:
        return None, None
    return len(covered) / len(required), covered == required


def _score_case(
    case_id: str,
    answerability: PdfGoldenAnswerability,
    result: PdfRetrievalResult,
    judgments: _EvidenceLabels,
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
    relevance = {
        record_id: _record_relevance(labels)
        for record_id, labels in judgments.items()
    }
    required = {
        record_id
        for record_id, label in relevance.items()
        if label is PdfRecordRelevance.REQUIRED
    }
    retrieved_required = required.intersection(ranked)
    required_evidence = _required_evidence(judgments, judgments)
    ranked_evidence = _required_evidence(judgments, ranked)
    context_evidence = _required_evidence(
        judgments, (*ranked, *result.context_record_ids),
    )
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
    ranked_coverage, all_ranked = _coverage(
        ranked_evidence, required_evidence, is_negative,
    )
    context_coverage, all_context = _coverage(
        context_evidence, required_evidence, is_negative,
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
        ranked_required_evidence_covered_at_k=len(ranked_evidence),
        ranked_required_evidence_coverage_at_k=ranked_coverage,
        all_ranked_required_evidence_at_k=all_ranked,
        context_required_evidence_covered_at_k=len(context_evidence),
        context_required_evidence_coverage_at_k=context_coverage,
        all_context_required_evidence_at_k=all_context,
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
            relevance.get(record_id) is PdfRecordRelevance.IRRELEVANT
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
