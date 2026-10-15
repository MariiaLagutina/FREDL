"""Synthetic label sheets and validated metric invocations."""

import csv
from hashlib import sha256
import io
from collections.abc import Callable

from src.evaluation.pdf.multibook_collector import (
    collect_pdf_multibook_judgments,
)
from src.evaluation.pdf.multibook_judgment_models import (
    PdfCollectedJudgments, PdfWorksheetSubmission,
)
from src.evaluation.pdf.multibook_metrics import score_pdf_multibook_retrieval
from src.evaluation.pdf.multibook_metric_models import PdfQueryRetrievalMetrics
from src.evaluation.pdf.multibook_pool_models import PdfRetrievalJudgmentPool
from src.evaluation.pdf.multibook_pool_worksheet import (
    POOL_WORKSHEET_COLUMNS, render_retrieval_pool_worksheet,
)
from tests.evaluation.pdf.multibook_contract_helpers import HASH
from tests.evaluation.pdf.multibook_pool_helpers import PoolInputs


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
