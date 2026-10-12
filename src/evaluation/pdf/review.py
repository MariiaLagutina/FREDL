"""Render private PDF evidence candidates for human relevance review."""

from collections.abc import Sequence
from pathlib import Path

from src.evaluation.pdf.evidence import find_pdf_evidence_candidates
from src.evaluation.pdf.models import (
    PdfGoldenAnswerability,
    PdfGoldenDataset,
)
from src.ingestion.pdf.index_records import PdfIndexRecord


def render_pdf_evidence_review(
    dataset: PdfGoldenDataset,
    records: Sequence[PdfIndexRecord],
) -> str:
    """Render candidate records without treating page overlap as relevance."""
    candidates = find_pdf_evidence_candidates(dataset, records)
    records_by_id = {record.record_id: record for record in records}
    lines = [
        f"# PDF evidence review: {dataset.dataset_id}",
        "",
        "> Page overlap creates candidates, not relevance labels. Mark each "
        "record as required, supporting, or irrelevant.",
        "",
    ]

    by_case: dict[str, list[int]] = {}
    for index, candidate in enumerate(candidates):
        by_case.setdefault(candidate.case_id, []).append(index)

    for case in dataset.cases:
        lines.extend([
            f"## {case.id}: {case.question}",
            "",
            f"**Answerability:** `{case.answerability.value}`",
            "",
            f"**Expected answer:** {case.expected_answer}",
            "",
        ])
        if case.answerability is PdfGoldenAnswerability.NOT_FOUND:
            lines.extend([
                "No relevant record is expected for this question.",
                "",
            ])
            continue

        for candidate_index in by_case.get(case.id, []):
            candidate = candidates[candidate_index]
            pages = ", ".join(map(str, candidate.physical_pages))
            lines.extend([
                f"### Evidence {candidate.evidence_index + 1}",
                "",
                f"- Label: `{candidate.role.value}`",
                f"- Modality: `{candidate.modality.value}`",
                f"- Physical pages: {pages}",
            ])
            if candidate.requires_visual_processing:
                lines.append(
                    "- Visual review required: text records cannot verify "
                    "the figure itself."
                )
            if candidate.missing_pages:
                missing = ", ".join(map(str, candidate.missing_pages))
                lines.append(f"- Missing from index records: {missing}")
            lines.append("")

            for record_id in candidate.page_record_ids:
                record = records_by_id[record_id]
                lines.extend(_render_record(record))

    return "\n".join(lines).rstrip() + "\n"


def write_pdf_evidence_review(
    path: Path,
    dataset: PdfGoldenDataset,
    records: Sequence[PdfIndexRecord],
) -> None:
    """Write one private human-review report beside local evaluation data."""
    content = render_pdf_evidence_review(dataset, records)
    try:
        path.write_text(content, encoding="utf-8")
    except OSError as error:
        raise ValueError("PDF evidence review could not be written") from error


def _render_record(record: PdfIndexRecord) -> list[str]:
    pages = tuple(dict.fromkeys(
        span.page_number for span in record.page_spans
    ))
    page_label = ", ".join(map(str, pages))
    section = " > ".join(record.section_path) or "Front matter"
    text_lines = record.text.splitlines() or [record.text]
    return [
        f"#### {record.record_id}",
        "",
        "- Decision: [ ] required  [ ] supporting  [ ] irrelevant",
        f"- Pages: {page_label}",
        f"- Section: {section}",
        f"- Flow range: {record.flow_index}:{record.flow_start}-"
        f"{record.flow_end}",
        "",
        *(f"    {line}" for line in text_lines),
        "",
    ]
