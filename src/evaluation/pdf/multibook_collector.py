"""Strict CSV collection for exact Slice 3 interpretation-aware bindings."""

from collections.abc import Mapping, Sequence
import csv
from hashlib import sha256
import io
import json

from src.evaluation.pdf.judgments import PdfRecordRelevance
from src.evaluation.pdf.library import PdfLibraryCorpus
from src.evaluation.pdf.multibook_golden import PdfMultibookGolden
from src.evaluation.pdf.multibook_judgment_models import (
    PdfCollectedJudgment, PdfCollectedJudgments, PdfWorksheetSubmission,
    PdfWorksheetSource,
)
from src.evaluation.pdf.multibook_pool_models import (
    PdfPoolingRun, PdfRetrievalJudgmentPool,
)
from src.evaluation.pdf.multibook_pooling import (
    require_pdf_retrieval_pool_matches,
)
from src.evaluation.pdf.multibook_pool_worksheet import (
    POOL_WORKSHEET_COLUMNS, render_retrieval_pool_worksheet,
)
from src.ingestion.pdf.reading_text import PdfReadingPage


Binding = tuple[str, str | None, str, str]


class PdfMissingJudgments(ValueError):
    """Expose all incomplete evaluated bindings without substituting zeros."""

    def __init__(self, bindings: Sequence[Binding]) -> None:
        self.bindings = tuple(bindings)
        super().__init__('Incomplete judgments: ' + json.dumps(self.bindings))


def collect_pdf_multibook_judgments(
    *, golden: PdfMultibookGolden, golden_sha256: str,
    corpus: PdfLibraryCorpus, pool: PdfRetrievalJudgmentPool,
    runs: Sequence[PdfPoolingRun],
    reading_pages: Mapping[str, Sequence[PdfReadingPage]],
    submissions: Sequence[PdfWorksheetSubmission],
) -> PdfCollectedJudgments:
    """Validate all rows and hashes; blank labels are valid draft inputs.

    Files may partition the worksheet by rows, but their union must contain
    exactly every pool task once. Item numbers must match the blind template;
    row order cannot transfer a label to another interpretation.
    """
    pool = PdfRetrievalJudgmentPool.model_validate(pool.model_dump())
    require_pdf_retrieval_pool_matches(
        pool, golden=golden, golden_sha256=golden_sha256, corpus=corpus,
        runs=runs, reading_pages=reading_pages,
    )
    _, template = render_retrieval_pool_worksheet(golden, corpus, pool)
    rows = list(csv.DictReader(io.StringIO(template)))
    tasks = {task.key: task for task in pool.tasks}
    expected = {row['item']: (
        row['case_id'], row['interpretation_id'] or None,
        row['unit_id'], row['record_id'],
    ) for row in rows}
    labels: dict[Binding, PdfCollectedJudgment] = {}
    sources = []
    for supplied in submissions:
        supplied = PdfWorksheetSubmission.model_validate(supplied.model_dump())
        if sha256(supplied.content).hexdigest() != supplied.sha256:
            raise ValueError('Worksheet file SHA256 differs')
        if not supplied.reviewer.strip():
            raise ValueError('Reviewer must not be blank')
        sources.append(PdfWorksheetSource(sha256=supplied.sha256,
                                          reviewer=supplied.reviewer))
        reader = csv.DictReader(io.StringIO(supplied.content.decode('utf-8'),
                                            newline=''), strict=True)
        try:
            if tuple(reader.fieldnames or ()) != POOL_WORKSHEET_COLUMNS:
                raise ValueError('Worksheet columns differ')
            supplied_rows = list(reader)
        except csv.Error:
            raise ValueError('Malformed worksheet CSV') from None
        for row in supplied_rows:
            if None in row or any(value is None for value in row.values()):
                raise ValueError('Malformed worksheet row')
            key = (row['case_id'], row['interpretation_id'] or None,
                   row['unit_id'], row['record_id'])
            if expected.get(row['item']) != key or key not in tasks:
                raise ValueError('Worksheet row binding differs')
            if key in labels:
                raise ValueError('Duplicate/conflicting worksheet binding')
            supply = row['supplies_evidence']
            if supply not in ('', 'yes', 'no'):
                raise ValueError('Supply must be yes, no or blank')
            labels[key] = PdfCollectedJudgment(
                task=tasks[key],
                relevance=PdfRecordRelevance(row['relevance'])
                if row['relevance'] else None,
                supplies_evidence=None if not supply else supply == 'yes',
                reviewer=supplied.reviewer, worksheet_sha256=supplied.sha256,
                note=row['note'],
            )
    missing = [task.key for task in pool.tasks if task.key not in labels]
    if missing:
        raise PdfMissingJudgments(missing)
    return PdfCollectedJudgments(
        golden_sha256=golden_sha256,
        golden_content_fingerprint=golden.content_fingerprint,
        library_fingerprint=corpus.manifest.fingerprint,
        pool_fingerprint=pool.fingerprint,
        sources=tuple(sorted(sources, key=lambda source: (
            source.sha256, source.reviewer,
        ))),
        judgments=tuple(labels[task.key] for task in pool.tasks),
    )


def require_collected_judgments_match(
    judgments: PdfCollectedJudgments, pool: PdfRetrievalJudgmentPool,
) -> PdfCollectedJudgments:
    """Revalidate copied models, exact pool scope and input identities."""
    judgments = PdfCollectedJudgments.model_validate(judgments.model_dump())
    if (
        judgments.pool_fingerprint != pool.fingerprint
        or judgments.golden_sha256 != pool.golden_sha256
        or judgments.golden_content_fingerprint
        != pool.golden_content_fingerprint
        or judgments.library_fingerprint != pool.library_fingerprint
    ):
        raise ValueError('Judgments belong to different inputs')
    if tuple(item.task for item in judgments.judgments) != pool.tasks:
        raise ValueError('Judgment bindings differ from pool scope/order')
    return judgments
