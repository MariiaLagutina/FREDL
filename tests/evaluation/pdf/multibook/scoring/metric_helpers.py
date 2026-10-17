"""Synthetic label sheets and validated metric invocations."""

import csv
from hashlib import sha256
import io
import json
from collections.abc import Callable

from src.evaluation.pdf.multibook.scoring.collector import (
    collect_pdf_multibook_judgments,
)
from src.evaluation.pdf.multibook.scoring.judgment_models import (
    PdfCollectedJudgments, PdfWorksheetSubmission,
)
from src.evaluation.pdf.multibook.scoring.metrics import (
    score_pdf_multibook_retrieval,
)
from src.evaluation.pdf.multibook.scoring.metric_models import (
    PdfQueryRetrievalMetrics,
)
from src.evaluation.pdf.multibook.pooling.models import (
    PdfRetrievalJudgmentPool,
)
from src.evaluation.pdf.multibook.pooling.worksheet import (
    POOL_WORKSHEET_COLUMNS, render_retrieval_pool_worksheet,
)
from tests.evaluation.pdf.multibook.retrieval.contract_helpers import HASH
from tests.evaluation.pdf.multibook.pooling.pool_helpers import PoolInputs


Decision = Callable[[dict[str, str]], tuple[str, str]]


def worksheet(inputs: PoolInputs, pool: PdfRetrievalJudgmentPool) -> str:
    golden, corpus, _, _ = inputs
    return render_retrieval_pool_worksheet(golden, corpus, pool)[1]


def fill(sheet: str, decide: Decision) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, list(POOL_WORKSHEET_COLUMNS),
                            lineterminator='\n')
    writer.writeheader()
    for row in csv.DictReader(io.StringIO(sheet)):
        row['relevance'], row['supplies_evidence'] = decide(row)
        writer.writerow(row)
    return output.getvalue()


def submission(sheet: str) -> PdfWorksheetSubmission:
    content = sheet.encode('utf-8')
    return PdfWorksheetSubmission(
        content=content, sha256=sha256(content).hexdigest(),
        reviewer='Synthetic reviewer',
    )


def collect(
    inputs: PoolInputs, pool: PdfRetrievalJudgmentPool, sheet: str,
) -> PdfCollectedJudgments:
    golden, corpus, runs, pages = inputs
    return collect_pdf_multibook_judgments(
        golden=golden, golden_sha256=HASH, corpus=corpus,
        runs=runs, reading_pages=pages, pool=pool,
        submissions=(submission(sheet),),
    )


def score(
    inputs: PoolInputs, pool: PdfRetrievalJudgmentPool,
    judgments: PdfCollectedJudgments,
) -> tuple[PdfQueryRetrievalMetrics, ...]:
    golden, corpus, runs, pages = inputs
    return score_pdf_multibook_retrieval(
        golden=golden, golden_sha256=HASH, corpus=corpus,
        runs=runs, reading_pages=pages, pool=pool, judgments=judgments,
    )


def metric_row(
    case_id: str, interpretation_id: str | None = None, *, category: str,
    outcome: str = 'complete', ber: tuple[float, float] | None = None,
    retriever: str = 'minilm',
) -> PdfQueryRetrievalMetrics:
    """Build one validated per-query row with chosen BER@5/BER@10 values.

    A row without ber has no declared sufficient set, so every metric is
    null. Localization equals BER and MRR equals BER@5, for brevity.
    """
    track = ('clarified' if interpretation_id is not None
             else 'original_ambiguous' if outcome == 'ambiguous'
             else 'original')
    values = (None, None) if ber is None else ber

    def coverage(cutoff: int, value: float | None) -> dict[str, object]:
        return {'cutoff': cutoff, 'ber': value,
                'sufficient_set_success': None if value is None
                else value == 1,
                'localization_coverage': value,
                'supplied_units': ['u'] if value else [],
                'localized_units': ['u'] if value else []}

    tracks = [coverage(cutoff, value)
              for cutoff, value in zip((5, 10), values)]
    zero = None if ber is None else 0.0
    return PdfQueryRetrievalMetrics.model_validate_json(json.dumps({
        'retriever': retriever,
        'query': {'case_id': case_id, 'interpretation_id': interpretation_id},
        'track': track, 'category': category,
        'expected_behavior': 'answer' if track == 'original'
        else 'clarification_required',
        'outcome': outcome,
        'eligible_sufficient_sets': [] if ber is None else [['u']],
        'exclusion': 'no_declared_sufficient_set' if ber is None else None,
        'ranked_only': tracks, 'with_context': tracks,
        'context_only_gain': [{
            'cutoff': cutoff, 'ber': zero, 'sufficient_set_success': zero,
            'localization_coverage': zero,
        } for cutoff in (5, 10)],
        'mrr_at_10_localization': values[0],
        'mrr_at_10_supply': values[0],
    }))
