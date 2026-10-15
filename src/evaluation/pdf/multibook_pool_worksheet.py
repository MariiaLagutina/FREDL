"""Render blind tasks; provenance stays in the separate pool artifact."""

import csv
import io
import re

from src.evaluation.pdf.library import PdfLibraryCorpus
from src.evaluation.pdf.multibook_golden import PdfMultibookGolden
from src.evaluation.pdf.multibook_pool_models import PdfRetrievalJudgmentPool
from src.evaluation.pdf.pooling import LABELING_GUIDE
from src.ingestion.pdf.fingerprints import fingerprint_canonical_json
from src.ingestion.pdf.index_records import PdfIndexRecord


POOL_WORKSHEET_COLUMNS = (
    "item", "case_id", "interpretation_id", "unit_id", "record_id",
    "relevance", "supplies_evidence", "note",
)


def render_retrieval_pool_worksheet(
    golden: PdfMultibookGolden, corpus: PdfLibraryCorpus,
    pool: PdfRetrievalJudgmentPool,
) -> tuple[str, str]:
    """Render a previously compatibility-checked pool with blank human labels.

    Query/unit groups retain contract order. Records use an identity hash
    independent of ranks, origins, pool fingerprint and source reading order.
    No scores, retrievers, anchor flags, expected outcomes or reference
    answers are shown. Unit content and original source text remain visible.
    """
    golden = PdfMultibookGolden.model_validate(golden.model_dump())
    pool = PdfRetrievalJudgmentPool.model_validate(pool.model_dump())
    if golden.content_fingerprint != pool.golden_content_fingerprint or (
        corpus.manifest.fingerprint != pool.library_fingerprint
    ):
        raise ValueError("Worksheet inputs differ from pool")
    cases = {case.case_id: case for case in golden.cases}
    queries = {query.identity.key: query for query in pool.query_plan.queries}
    records = {record.record_id: record for record in corpus.source_records}
    lines = ["# Relevance judgment worksheet", "", "Private review material.",
             "", LABELING_GUIDE]
    sheet = io.StringIO()
    writer = csv.writer(sheet, lineterminator="\n")
    writer.writerow(POOL_WORKSHEET_COLUMNS)
    group_order = {key: index for index, key in enumerate(dict.fromkeys(
        (*task.query.key, task.unit_id) for task in pool.tasks
    ))}
    review_tasks = sorted(pool.tasks, key=lambda task: (
        group_order[(*task.query.key, task.unit_id)],
        fingerprint_canonical_json({
            "schema": "fredl.pdf.blind-review-order-v1",
            "query": task.query.model_dump(mode="json"),
            "unit_id": task.unit_id, "record_id": task.record_id,
            "book_path": records[task.record_id].book_path,
        }), task.record_id,
    ))
    current = None
    for number, task in enumerate(review_tasks, 1):
        item = f"W{number:04d}"
        key = (*task.query.key, task.unit_id)
        if key != current:
            query = queries[task.query.key]
            case = cases[task.query.case_id]
            unit = golden.units[task.unit_id]
            lines.extend([
                f"## {task.query.case_id}", "", f"Question: {query.text}", "",
                "Interpretation: " + (task.query.interpretation_id
                                      or "original question"), "",
                f"Evidence unit: {unit.unit_id} ({unit.modality.value})", "",
                f"Required content: {unit.required_content}", "",
            ])
            if query.track == "original_ambiguous":
                meanings = [meaning.label for meaning in case.interpretations
                            if task.unit_id in meaning.unit_ids]
                lines.extend([
                    "Conditional meaning for this unit: " + "; ".join(
                        meanings,
                    ), "Judge this unit separately; do not combine meanings.",
                    "",
                ])
            current = key
        lines.extend(_review_record(item, records[task.record_id]))
        writer.writerow((item, task.query.case_id,
                         task.query.interpretation_id or "", task.unit_id,
                         task.record_id, "", "", ""))
    no_units = [
        query for query in pool.query_plan.queries
        if not any(task.query == query.identity for task in pool.tasks)
    ]
    for query in no_units:
        lines.extend([
            f"## {query.identity.case_id}", "", f"Question: {query.text}", "",
            "Interpretation: " + (query.identity.interpretation_id
                                  or "original question"), "",
            "No evidence-unit tasks. Candidates remain in the inventory; "
            "no relevance label is inferred.", "",
        ])
    return "\n".join(lines) + "\n", sheet.getvalue()


def _review_record(item: str, record: PdfIndexRecord) -> list[str]:
    """Keep source characters inside a fence longer than its backticks."""
    fence = '`' * max(3, 1 + max(
        (len(run) for run in re.findall(r'`+', record.text)), default=0,
    ))
    pages = ', '.join(str(page) for page in sorted({
        span.page_number for span in record.page_spans
    }))
    section = ' > '.join(record.section_path) or '(no verified section)'
    # A separator before the closing fence is markup, not a source-text edit.
    separator = '' if record.text.endswith('\n') else '\n'
    return [f'#### {item}', '', f'Record: {record.record_id}', '',
            f'{record.book_path}, physical pages {pages}; section: {section}',
            '', f'{fence}text\n{record.text}{separator}{fence}', '']
