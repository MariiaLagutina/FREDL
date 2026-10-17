"""Synthetic PDFs, review packages and decisions for multi-book tests."""

import csv
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.evaluation.pdf.multibook.curation.golden import (
    PdfExpectedOutcome,
    PdfLimitationReason,
)
from src.evaluation.pdf.multibook.curation.importer import (
    PdfImportDecision,
    PdfImportDecisions,
)


BOOKS = {"ALP": "books/alpha.pdf", "BET": "books/beta.pdf"}
PAGES: dict[str, tuple[str, ...]] = {
    "ALP": ("Queues hold work until a worker is free.",
            "Workers take one job at a time from the queue."),
    "BET": ("Caches keep recent answers close to the reader.",
            "A clipped listing ends before its return statement."),
}


def _anchor(alias: str, page: int, anchor: str, text: str) -> object:
    start = text.index(anchor)
    return {
        "book": alias, "book_path": BOOKS[alias], "physical_page": page,
        "locator": {"exact_anchor": anchor, "reading_text_start": start,
                    "reading_text_end": start + len(anchor)},
        "page_text_sha256": sha256(text.encode("utf-8")).hexdigest(),
        "verified_anchor_section_path": [],
        "reliability_findings": [],
    }


def _unit(unit_id: str, modality: str, references: list[object]) -> object:
    return {
        "id": unit_id, "required_content": f"Content of {unit_id}.",
        "modality": modality, "references": references,
        "acceptable_alternatives": [], "human_review_status": "pending",
    }


def _candidate(
    case_id: str,
    category: str,
    proposal: str,
    source: str,
    sets: list[list[str]],
    approved: str | None,
    others: list[str] | None = None,
    **review: Any,
) -> dict[str, Any]:
    return {
        "id": case_id, "category": category,
        "question": f"Proposed question {case_id}?",
        "candidate_expected_answerability": proposal,
        "candidate_source_support": source,
        "candidate_expected_answer": "Reference notes.",
        "minimal_sufficient_evidence_sets": sets,
        "supporting_or_distractor_units": others or [],
        "human_review": {
            "decision": "accept", "approved_answerability": approved,
            "approved_evidence_sets": sets, "reviewer": "Reviewer",
            "notes": "", **review,
        },
    }


def _package_files(texts: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    alpha, beta = texts["ALP"], texts["BET"]
    candidates = [
        _candidate("q-queue", "single_book", "complete", "complete",
                   [["alp-queue"]], "complete",
                   approved_question="Approved question about queues?"),
        _candidate("q-both", "complementary_multi_book", "complete",
                   "complete", [["alp-queue", "bet-cache"]], "complete"),
        _candidate("q-absent", "hard_negative", "unsupported", "not_found",
                   [], "unsupported", ["alp-worker"],
                   source_support="unsupported"),
        _candidate("q-figure", "reliability_limited", "extraction_limited",
                   "complete_in_visual_source", [["alp-worker"]],
                   "extraction_limited"),
        _candidate("q-clipped", "reliability_limited", "extraction_limited",
                   "incomplete_in_pdf", [], None, ["bet-clipped"],
                   source_support="incomplete",
                   text_context_support="incomplete",
                   proposed_limitation_reason="source_incomplete"),
    ]
    return {
        "candidates.json": {
            "book_aliases": {"ALP": "alpha", "BET": "beta"},
            "candidates": candidates,
            "evidence_units": {
                "alp-queue": _unit("alp-queue", "text", [
                    _anchor("ALP", 1, "Queues hold work", alpha[0])]),
                "alp-worker": _unit("alp-worker", "figure", [
                    _anchor("ALP", 2, "Workers take one job", alpha[1])]),
                "bet-cache": _unit("bet-cache", "text", [
                    _anchor("BET", 1, "Caches keep recent answers",
                            beta[0])]),
                "bet-clipped": _unit("bet-clipped", "structured_text", [
                    _anchor("BET", 2, "A clipped listing", beta[1])]),
            },
        },
        "source-manifest.json": {"books": {
            alias: {"book_path": path, "pdf_sha256": "a" * 64,
                    "record_fingerprint": "b" * 64}
            for alias, path in BOOKS.items()
        }},
    }


def _write_package(path: Path, files: dict[str, Any]) -> Path:
    path.mkdir(parents=True)
    for name, content in files.items():
        (path / name).write_text(json.dumps(content), encoding="utf-8")
    with (path / "review-decisions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "decision"])
        for candidate in files["candidates.json"]["candidates"]:
            writer.writerow([candidate["id"], "accept"])
    _refresh_checksums(path)
    return path


def _refresh_checksums(path: Path) -> None:
    listed = {
        item.relative_to(path).as_posix():
            sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.rglob("*"))
        if item.is_file() and item.name != "package-checksums.json"
    }
    (path / "package-checksums.json").write_text(json.dumps(listed))


def _decisions() -> PdfImportDecisions:
    return PdfImportDecisions(decisions=(
        PdfImportDecision(
            case_id="q-clipped",
            expected_outcome=PdfExpectedOutcome.SOURCE_INCOMPLETE,
            provenance="Reviewer decision D3",
        ),
        PdfImportDecision(
            case_id="q-figure",
            limitation_reason=(
                PdfLimitationReason.VISUAL_EVIDENCE_UNAVAILABLE
            ),
            provenance="Reviewer checkpoint note",
        ),
    ))


def _write_pdf(path: Path, texts: tuple[str, ...]) -> None:
    writer = PdfWriter()
    font = writer._add_object(DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    }))
    for text in texts:
        page = writer.add_blank_page(width=400, height=400)
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
        })
        content = DecodedStreamObject()
        content.set_data(
            f"BT /F1 11 Tf 20 300 Td ({text}) Tj ET".encode("ascii")
        )
        page[NameObject("/Contents")] = writer._add_object(content)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        writer.write(stream)


def _with_ambiguous(texts: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    files = _package_files(texts)
    files["candidates.json"]["candidates"].append(_candidate(
        "q-which", "ambiguous_terminology", "ambiguous_or_conflicting",
        "conditional", [], "ambiguous_or_conflicting",
        ["alp-queue", "bet-cache"],
    ))
    return files


def _interpretations() -> PdfImportDecision:
    from src.evaluation.pdf.multibook.curation.golden import PdfInterpretation

    def meaning(name: str, unit: str, outcome: str) -> PdfInterpretation:
        support = "complete" if outcome == "complete" else "partial"
        return PdfInterpretation.model_validate({
            "interpretation_id": name, "label": f"Meaning {name}",
            "clarified_question": f"Clarified {name}?",
            "expected_outcome": outcome, "source_support": support,
            "text_context_support": support,
            "sufficient_sets": [[unit]],
        })

    return PdfImportDecision(
        case_id="q-which",
        interpretations=(meaning("queue", "alp-queue", "complete"),
                         meaning("cache", "bet-cache", "partial")),
        provenance="Reviewer interpretation decision",
    )


def _as_v1(golden: Any) -> Any:
    from src.evaluation.pdf.multibook.curation.golden_v1 import (
        PdfMultibookGoldenV1,
    )

    data = golden.model_dump(mode="json", by_alias=True)
    data["version"] = 1
    data.pop("migrated_from")
    for case in data["cases"]:
        del case["expected_behavior"], case["interpretations"]
    return PdfMultibookGoldenV1.model_validate(data)
