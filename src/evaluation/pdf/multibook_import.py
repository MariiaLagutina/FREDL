"""Import a reviewed candidate package as a draft multi-book golden set.

The import is deterministic and never freezes a dataset. It verifies the
package checksums and that every candidate was accepted by a human, uses
only human-approved wording and evidence sets, and stops when a case has
no approved outcome unless an explicit, recorded decision supplies one.
Every normalization is reported; the package itself is never modified.
"""

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.evaluation.pdf.library import (
    PdfLibraryBookIdentity,
    PdfLibraryCorpus,
    PdfLibraryManifest,
    require_library_matches_manifest,
)
from src.evaluation.pdf.models import PdfGoldenModality
from src.evaluation.pdf.multibook_golden import (
    PdfEvidenceLocator,
    PdfEvidenceUnit,
    PdfExpectedOutcome,
    PdfGoldenFreezeRecord,
    PdfGoldenProvenance,
    PdfLimitationReason,
    PdfMultibookCase,
    PdfMultibookCategory,
    PdfMultibookGolden,
    PdfSourceSupport,
    PdfTextSupport,
)
from src.ingestion.pdf.reading_text import PdfReadingPage


PDF_MULTIBOOK_DATASET_ID = "fredl-multibook-golden-v1"
CHECKSUM_FILE = "package-checksums.json"
CANDIDATES_FILE = "candidates.json"
DECISIONS_FILE = "review-decisions.csv"
SOURCE_MANIFEST_FILE = "source-manifest.json"

_APPROVED_OUTCOMES = {
    "complete": PdfExpectedOutcome.COMPLETE,
    "partial": PdfExpectedOutcome.PARTIAL,
    "ambiguous_or_conflicting": PdfExpectedOutcome.AMBIGUOUS,
    "unsupported": PdfExpectedOutcome.UNSUPPORTED,
    "extraction_limited": PdfExpectedOutcome.EXTRACTION_LIMITED,
    "source_incomplete": PdfExpectedOutcome.SOURCE_INCOMPLETE,
}
# Review-package vocabulary for support, mapped to the schema. A value
# outside these tables stops the import instead of being guessed.
_SOURCE_SUPPORT = {
    "complete": PdfSourceSupport.COMPLETE,
    "complete_in_visual_source": PdfSourceSupport.COMPLETE,
    "complete_in_structured_source": PdfSourceSupport.COMPLETE,
    "partial": PdfSourceSupport.PARTIAL,
    "conditional": PdfSourceSupport.CONDITIONAL,
    "not_found": PdfSourceSupport.NOT_FOUND,
    "unsupported": PdfSourceSupport.NOT_FOUND,
    "incomplete": PdfSourceSupport.INCOMPLETE,
    "incomplete_in_pdf": PdfSourceSupport.INCOMPLETE,
}
_TEXT_SUPPORT = {
    "complete": PdfTextSupport.COMPLETE,
    "partial": PdfTextSupport.PARTIAL,
    "conditional": PdfTextSupport.CONDITIONAL,
    "not_found": PdfTextSupport.NOT_FOUND,
    "unsupported": PdfTextSupport.NOT_FOUND,
    "extraction_limited": PdfTextSupport.EXTRACTION_LIMITED,
    "incomplete": PdfTextSupport.INCOMPLETE,
}
_OUTCOME_SUPPORT = {
    PdfExpectedOutcome.COMPLETE: (
        PdfSourceSupport.COMPLETE, PdfTextSupport.COMPLETE,
    ),
    PdfExpectedOutcome.PARTIAL: (
        PdfSourceSupport.PARTIAL, PdfTextSupport.PARTIAL,
    ),
    PdfExpectedOutcome.AMBIGUOUS: (
        PdfSourceSupport.CONDITIONAL, PdfTextSupport.CONDITIONAL,
    ),
    PdfExpectedOutcome.UNSUPPORTED: (
        PdfSourceSupport.NOT_FOUND, PdfTextSupport.NOT_FOUND,
    ),
    PdfExpectedOutcome.EXTRACTION_LIMITED: (
        PdfSourceSupport.COMPLETE, PdfTextSupport.EXTRACTION_LIMITED,
    ),
    PdfExpectedOutcome.SOURCE_INCOMPLETE: (
        PdfSourceSupport.INCOMPLETE, PdfTextSupport.INCOMPLETE,
    ),
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PdfImportDecision(_StrictModel):
    """Supply a human decision the review package does not encode.

    A decision may fill a missing outcome or limitation reason; it can
    never replace a value the package already approved.
    """

    case_id: str = Field(min_length=1)
    expected_outcome: PdfExpectedOutcome | None = None
    limitation_reason: PdfLimitationReason | None = None
    provenance: str = Field(min_length=1)


class PdfImportDecisions(_StrictModel):
    """Hold every explicit decision used by one import."""

    schema_name: Literal["fredl.pdf.multibook-import-decisions"] = Field(
        default="fredl.pdf.multibook-import-decisions", alias="schema",
    )
    version: Literal[1] = 1
    decisions: tuple[PdfImportDecision, ...] = ()

    model_config = ConfigDict(
        frozen=True, extra="forbid", populate_by_name=True,
    )


@dataclass(frozen=True, slots=True)
class PdfImportFinding:
    """Describe one normalization, decision, or difference worth review."""

    subject: str
    kind: str
    detail: str


@dataclass(frozen=True, slots=True)
class PdfReviewPackage:
    """Hold the verified files of one candidate review package."""

    package_id: str
    candidates: Mapping[str, Any]
    decision_rows: tuple[Mapping[str, str], ...]
    source_manifest: Mapping[str, Any]
    file_sha256: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class PdfMultibookImport:
    """Return a draft dataset with every finding that needs review."""

    golden: PdfMultibookGolden
    findings: tuple[PdfImportFinding, ...]


def load_pdf_review_package(package_dir: Path) -> PdfReviewPackage:
    """Read a package after checking every file against its checksums."""
    try:
        listed = json.loads(
            (package_dir / CHECKSUM_FILE).read_text(encoding="utf-8"),
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Review package checksums could not be read") from (
            error
        )
    present = {
        path.relative_to(package_dir).as_posix()
        for path in package_dir.rglob("*")
        if path.is_file() and path.name != CHECKSUM_FILE
    }
    if present != set(listed):
        extra = sorted(present - set(listed))
        missing = sorted(set(listed) - present)
        raise ValueError(
            "Review package files differ from its checksums: "
            f"unlisted {extra}, missing {missing}"
        )
    hashes = {}
    for name, expected in sorted(listed.items()):
        actual = sha256((package_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Review package file changed: {name}")
        hashes[name] = actual
    with (package_dir / DECISIONS_FILE).open(encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    return PdfReviewPackage(
        package_id=package_dir.name,
        candidates=json.loads(
            (package_dir / CANDIDATES_FILE).read_text(encoding="utf-8"),
        ),
        decision_rows=rows,
        source_manifest=json.loads(
            (package_dir / SOURCE_MANIFEST_FILE).read_text(encoding="utf-8"),
        ),
        file_sha256=hashes,
    )


def load_pdf_import_decisions(path: Path) -> PdfImportDecisions:
    """Read the explicit decisions file for an import."""
    try:
        return PdfImportDecisions.model_validate_json(
            path.read_text(encoding="utf-8"),
        )
    except OSError as error:
        raise ValueError("Import decisions could not be read") from error
    except ValidationError as error:
        raise ValueError("Import decisions are invalid") from error


def import_pdf_multibook_candidates(
    package: PdfReviewPackage,
    decisions: PdfImportDecisions,
    dataset_id: str = PDF_MULTIBOOK_DATASET_ID,
    language: str = "en",
) -> PdfMultibookImport:
    """Build a draft golden set from human-approved candidate fields only."""
    findings: list[PdfImportFinding] = []
    candidates = package.candidates["candidates"]
    by_decision = _decisions_by_case(decisions)
    _require_all_accepted(candidates, package.decision_rows)
    aliases = package.candidates["book_aliases"]
    library = _library_manifest(package.source_manifest, aliases)
    units = tuple(
        _evidence_unit(unit, findings)
        for unit in package.candidates["evidence_units"].values()
    )
    cases = tuple(
        _case(candidate, by_decision.get(candidate["id"]), findings)
        for candidate in candidates
    )
    unused = sorted(set(by_decision) - {case.case_id for case in cases})
    if unused:
        raise ValueError(f"Import decisions name unknown cases: {unused}")
    try:
        golden = PdfMultibookGolden(
            dataset_id=dataset_id,
            status="draft",
            language=language,
            library=library,
            provenance=PdfGoldenProvenance(
                package_id=package.package_id,
                candidates_sha256=package.file_sha256[CANDIDATES_FILE],
                review_decisions_sha256=package.file_sha256[DECISIONS_FILE],
                source_manifest_sha256=package.file_sha256[
                    SOURCE_MANIFEST_FILE
                ],
                decisions=tuple(
                    f"{item.case_id}: {item.provenance}"
                    for item in decisions.decisions
                ),
            ),
            evidence_units=units,
            cases=cases,
        )
    except ValidationError as error:
        raise ValueError(
            f"Imported candidates are invalid: {error.errors()[0]['msg']}",
        ) from error
    return PdfMultibookImport(golden=golden, findings=tuple(findings))


def verify_pdf_multibook_golden(
    golden: PdfMultibookGolden,
    library: PdfLibraryCorpus,
    reading_pages: Mapping[str, Sequence[PdfReadingPage]],
) -> tuple[PdfImportFinding, ...]:
    """Recheck every locator against the current extracted library.

    Files and record sets must match the pinned library. Each locator's
    page text must hash as recorded, hold its anchor at the recorded
    offsets, and overlap at least one record. The result reports how
    many records each unit's anchors overlap; it assigns no relevance.
    """
    require_library_matches_manifest(library, golden.library)
    texts = {
        (book_path, page.page_number): page.reading_text
        for book_path, pages in reading_pages.items()
        for page in pages
    }
    records = library.source_records
    findings = []
    for unit in golden.evidence_units:
        overlapping: set[str] = set()
        for locator in (*unit.locators, *unit.supporting_locators):
            text = texts.get((locator.book_path, locator.physical_page))
            if text is None:
                raise ValueError(
                    f"{unit.unit_id}: page {locator.physical_page} is missing"
                )
            if sha256(text.encode("utf-8")).hexdigest() != (
                locator.page_text_sha256
            ):
                raise ValueError(f"{unit.unit_id}: page text changed")
            if text[locator.page_text_start:locator.page_text_end] != (
                locator.exact_anchor
            ):
                raise ValueError(f"{unit.unit_id}: anchor moved")
            hits = anchor_records(locator, records)
            if not hits:
                raise ValueError(f"{unit.unit_id}: anchor overlaps no record")
            if locator in unit.locators:
                overlapping.update(hits)
        findings.append(PdfImportFinding(
            unit.unit_id, "locators_verified",
            f"{len(unit.locators)} required and "
            f"{len(unit.supporting_locators)} supporting locators match; "
            f"required anchors overlap {len(overlapping)} record(s)",
        ))
    return tuple(findings)


def anchor_records(
    locator: PdfEvidenceLocator,
    records: Sequence[Any],
) -> tuple[str, ...]:
    """Return records whose page spans overlap one locator's anchor."""
    return tuple(
        record.record_id
        for record in records
        if record.book_path == locator.book_path
        and any(
            span.page_number == locator.physical_page
            and span.page_source_start < locator.page_text_end
            and locator.page_text_start < span.page_source_end
            for span in record.page_spans
        )
    )


def freeze_pdf_multibook_golden(
    draft: PdfMultibookGolden,
    *,
    approved_by: str,
    approval_note: str,
) -> PdfMultibookGolden:
    """Return the frozen copy of a reviewed draft, content unchanged."""
    if draft.status != "draft":
        raise ValueError("Only a draft multi-book golden set can be frozen")
    frozen = draft.model_copy(update={
        "status": "frozen",
        "freeze_record": PdfGoldenFreezeRecord(
            approved_by=approved_by, approval_note=approval_note,
        ),
    })
    frozen = PdfMultibookGolden.model_validate(
        frozen.model_dump(mode="json", by_alias=True),
    )
    require_freeze_preserves_draft(draft, frozen)
    return frozen


def require_freeze_preserves_draft(
    draft: PdfMultibookGolden,
    frozen: PdfMultibookGolden,
) -> None:
    """Prove that freezing changed nothing except the freeze state."""
    if frozen.status != "frozen":
        raise ValueError("Multi-book golden set is not frozen")
    if frozen.content_fingerprint != draft.content_fingerprint:
        raise ValueError("Frozen multi-book golden set differs from draft")


def write_pdf_multibook_golden(path: Path, golden: PdfMultibookGolden) -> None:
    """Write a dataset once; an existing file must already be identical."""
    content = golden.model_dump_json(by_alias=True, indent=1) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content)
    except FileExistsError as error:
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Refusing to overwrite {path}") from error


def render_pdf_import_report(
    imported: PdfMultibookImport,
    verification: Sequence[PdfImportFinding] = (),
) -> str:
    """Render a private, human-readable report of the import."""
    golden = imported.golden
    lines = [
        "# Multi-book golden import report",
        "",
        "Private review material. This draft is not frozen and holds no "
        "record relevance judgments.",
        "",
        f"- Dataset: `{golden.dataset_id}` (status `{golden.status}`)",
        f"- Content fingerprint: `{golden.content_fingerprint}`",
        f"- Library fingerprint: `{golden.library.fingerprint}`",
        f"- Cases: {len(golden.cases)}; evidence units: "
        f"{len(golden.evidence_units)}",
        "",
        "## Expected outcomes",
        "",
        "| Case | Category | Outcome | Source | Extracted text | Limitation |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in golden.cases:
        reason = case.limitation_reason
        lines.append(
            f"| {case.case_id} | {case.category.value} | "
            f"{case.expected_outcome.value} | {case.source_support.value} | "
            f"{case.text_context_support.value} | "
            f"{reason.value if reason else '-'} |"
        )
    for title, items in (
        ("Import findings", imported.findings),
        ("Locator verification", verification),
    ):
        lines.extend(["", f"## {title}", ""])
        if not items:
            lines.append("- None.")
        for item in items:
            lines.append(f"- `{item.subject}` {item.kind}: {item.detail}")
    return "\n".join(lines) + "\n"


def _decisions_by_case(
    decisions: PdfImportDecisions,
) -> dict[str, PdfImportDecision]:
    result: dict[str, PdfImportDecision] = {}
    for decision in decisions.decisions:
        if decision.case_id in result:
            raise ValueError(f"Duplicate import decision: {decision.case_id}")
        result[decision.case_id] = decision
    return result


def _require_all_accepted(
    candidates: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, str]],
) -> None:
    csv_decisions = {row["id"]: row["decision"] for row in rows}
    json_ids = [candidate["id"] for candidate in candidates]
    if len(json_ids) != len(set(json_ids)) or set(json_ids) != set(
        csv_decisions
    ):
        raise ValueError("Candidate IDs differ between JSON and CSV")
    for candidate in candidates:
        decision = candidate["human_review"]["decision"]
        if decision != "accept" or csv_decisions[candidate["id"]] != "accept":
            raise ValueError(
                f"{candidate['id']} is not accepted by human review"
            )


def _library_manifest(
    source_manifest: Mapping[str, Any],
    aliases: Mapping[str, str],
) -> PdfLibraryManifest:
    books = []
    for alias, book in source_manifest["books"].items():
        if f"books/{aliases[alias]}.pdf" != book["book_path"]:
            raise ValueError(f"Book alias {alias} disagrees with its path")
        books.append(PdfLibraryBookIdentity(
            book_path=book["book_path"],
            pdf_sha256=book["pdf_sha256"],
            record_set_fingerprint=book["record_fingerprint"],
        ))
    return PdfLibraryManifest(
        books=tuple(sorted(books, key=lambda item: item.book_path)),
    )


def _evidence_unit(
    unit: Mapping[str, Any],
    findings: list[PdfImportFinding],
) -> PdfEvidenceUnit:
    locators = tuple(
        PdfEvidenceLocator(
            book_path=reference["book_path"],
            physical_page=reference["physical_page"],
            exact_anchor=reference["locator"]["exact_anchor"],
            page_text_start=reference["locator"]["reading_text_start"],
            page_text_end=reference["locator"]["reading_text_end"],
            page_text_sha256=reference["page_text_sha256"],
            verified_section_path=tuple(
                reference["verified_anchor_section_path"],
            ),
            reliability_findings=_findings(reference),
        )
        for reference in unit["references"]
    )
    supporting = tuple(
        PdfEvidenceLocator(
            book_path=alternative["book_path"],
            physical_page=alternative["physical_page"],
            exact_anchor=locator["exact_anchor"],
            page_text_start=locator["reading_text_start"],
            page_text_end=locator["reading_text_end"],
            page_text_sha256=alternative["page_text_sha256"],
            verified_section_path=tuple(alternative["verified_section_path"]),
        )
        for alternative in unit.get("supporting_evidence_alternatives", ())
        for locator in alternative["locators"]
    )
    if supporting:
        findings.append(PdfImportFinding(
            unit["id"], "supporting_locators",
            f"{len(supporting)} verified supporting locator(s) kept as "
            "non-mandatory corroboration",
        ))
    notes = tuple(unit.get("acceptable_alternatives", ()))
    return PdfEvidenceUnit(
        unit_id=unit["id"],
        required_content=unit["required_content"],
        modality=PdfGoldenModality(unit["modality"]),
        locators=locators,
        supporting_locators=supporting,
        notes=notes,
    )


def _findings(reference: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted({
        f"{item['kind']}:{item['support']}"
        for item in reference.get("reliability_findings", ())
    }))


def _case(
    candidate: Mapping[str, Any],
    decision: PdfImportDecision | None,
    findings: list[PdfImportFinding],
) -> PdfMultibookCase:
    case_id = candidate["id"]
    review = candidate["human_review"]
    outcome = _outcome(case_id, review, decision, findings)
    proposal = candidate["candidate_expected_answerability"]
    if _APPROVED_OUTCOMES.get(proposal) is not outcome:
        findings.append(PdfImportFinding(
            case_id, "proposal_differs",
            f"original proposal {proposal}; imported outcome "
            f"{outcome.value}. The package keeps the proposal unchanged",
        ))
    reason = _limitation(case_id, review, decision, findings)
    source, text = _OUTCOME_SUPPORT[outcome]
    _compare_support(case_id, candidate, review, source, text, findings)

    question = review.get("approved_question") or candidate["question"]
    if question != candidate["question"]:
        findings.append(PdfImportFinding(
            case_id, "approved_question",
            "human-approved wording replaces the proposal; the original "
            "stays in the review package",
        ))
    sets = tuple(tuple(item) for item in review["approved_evidence_sets"])
    proposed = tuple(
        tuple(item) for item in candidate["minimal_sufficient_evidence_sets"]
    )
    if sets != proposed:
        findings.append(PdfImportFinding(
            case_id, "approved_evidence_sets",
            f"approved {_sets(sets)} replaces proposed {_sets(proposed)}",
        ))
    others = tuple(candidate["supporting_or_distractor_units"])
    in_sets = {unit for item in sets for unit in item}
    dropped = tuple(unit for unit in others if unit in in_sets)
    if dropped:
        findings.append(PdfImportFinding(
            case_id, "related_units",
            f"{', '.join(dropped)} now required by an approved set; "
            "removed from related units",
        ))
    others = tuple(unit for unit in others if unit not in in_sets)
    ambiguous = outcome is PdfExpectedOutcome.AMBIGUOUS
    return PdfMultibookCase(
        case_id=case_id,
        question=question,
        category=PdfMultibookCategory(candidate["category"]),
        expected_outcome=outcome,
        source_support=source,
        text_context_support=text,
        limitation_reason=reason,
        sufficient_sets=sets,
        conditional_units=others if ambiguous else (),
        related_units=() if ambiguous else others,
        reference_notes=candidate.get("candidate_expected_answer", ""),
    )


def _outcome(
    case_id: str,
    review: Mapping[str, Any],
    decision: PdfImportDecision | None,
    findings: list[PdfImportFinding],
) -> PdfExpectedOutcome:
    approved = review.get("approved_answerability")
    decided = decision.expected_outcome if decision else None
    if approved is None:
        if decided is None:
            raise ValueError(f"{case_id} has no approved expected outcome")
        assert decision is not None
        findings.append(PdfImportFinding(
            case_id, "decision_applied",
            f"expected outcome {decided.value} from an explicit decision "
            f"({decision.provenance}); the package recorded none",
        ))
        return decided
    if approved not in _APPROVED_OUTCOMES:
        raise ValueError(f"{case_id} has unknown outcome {approved}")
    outcome = _APPROVED_OUTCOMES[approved]
    if decided is not None and decided is not outcome:
        raise ValueError(
            f"{case_id}: decision {decided.value} contradicts approved "
            f"{outcome.value}"
        )
    if approved != outcome.value:
        findings.append(PdfImportFinding(
            case_id, "normalized_outcome", f"{approved} -> {outcome.value}",
        ))
    return outcome


def _limitation(
    case_id: str,
    review: Mapping[str, Any],
    decision: PdfImportDecision | None,
    findings: list[PdfImportFinding],
) -> PdfLimitationReason | None:
    recorded = review.get("limitation_reason") or review.get(
        "proposed_limitation_reason",
    )
    package = PdfLimitationReason(recorded) if recorded else None
    decided = decision.limitation_reason if decision else None
    if package is not None and decided is not None and package is not decided:
        raise ValueError(
            f"{case_id}: decided limitation {decided.value} contradicts "
            f"recorded {package.value}"
        )
    if package is None and decided is not None:
        assert decision is not None
        findings.append(PdfImportFinding(
            case_id, "decision_applied",
            f"limitation reason {decided.value} from an explicit decision "
            f"({decision.provenance}); the package recorded it only as text",
        ))
    return package or decided


def _compare_support(
    case_id: str,
    candidate: Mapping[str, Any],
    review: Mapping[str, Any],
    source: PdfSourceSupport,
    text: PdfTextSupport,
    findings: list[PdfImportFinding],
) -> None:
    for label, value, table, expected in (
        ("proposed source support", candidate["candidate_source_support"],
         _SOURCE_SUPPORT, source),
        ("reviewed source support", review.get("source_support"),
         _SOURCE_SUPPORT, source),
        ("reviewed text support", review.get("text_context_support"),
         _TEXT_SUPPORT, text),
    ):
        if value is None:
            continue
        mapped = table.get(value)
        if mapped is None:
            raise ValueError(f"{case_id}: unknown {label} {value}")
        if mapped is not expected:
            raise ValueError(
                f"{case_id}: {label} {value} conflicts with the expected "
                f"outcome ({expected.value})"
            )
        if value != mapped.value:
            findings.append(PdfImportFinding(
                case_id, "normalized_support",
                f"{label} {value} -> {mapped.value}",
            ))


def _sets(sets: Sequence[Sequence[str]]) -> str:
    return " OR ".join("(" + " AND ".join(item) + ")" for item in sets) or "-"
