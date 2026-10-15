"""Build candidate pools and a blind worksheet for human judgments.

A pool lists records a reviewer must judge per evidence unit. Records
come from verified locators (the anchor's own record and the other
records on its page) and, once retrieval runs are approved, from the top
results of each retriever. The worksheet never shows why a record was
offered, so no retriever can be favoured while labelling.
"""

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import io
import re
from typing import cast, Literal

from src.evaluation.pdf.judgments import PdfRecordRelevance
from src.evaluation.pdf.multibook.library import PdfLibraryCorpus
from src.evaluation.pdf.multibook.curation.golden import (
    PdfEvidenceLocator,
    PdfMultibookCase,
    PdfMultibookGolden,
)
from src.evaluation.pdf.multibook.curation.importer import anchor_records
from src.evaluation.pdf.multibook.curation.pilot_judgments import (
    PdfJudgmentPool,
    PdfMultibookJudgment,
    PdfMultibookJudgmentSet,
    PdfPoolEntry,
    PdfPoolSource,
    pool_units,
    require_pool_matches,
)
from src.ingestion.pdf.index_records import PdfIndexRecord


WORKSHEET_COLUMNS = (
    "item", "case_id", "unit_id", "record_id", "relevance",
    "supplies_evidence", "note",
)
RETRIEVAL_SOURCES = (
    PdfPoolSource.BM25, PdfPoolSource.MINILM, PdfPoolSource.HYBRID,
)
_SOURCE_ORDER = {source: index for index, source in enumerate(PdfPoolSource)}
_WORDS = re.compile(r"\S+")
_YES = {"yes", "y", "true"}
_NO = {"no", "n", "false"}

LABELING_GUIDE = """\
## How to label

Judge each record only against the evidence unit named above it, and only
from the record text shown. Do not use outside knowledge, other records,
or the original PDF to fill gaps; note them instead.

- **required**: the record contains this unit's required content, or the
  part of it that this record is expected to carry.
- **supporting**: the record helps with the unit (context, a related
  definition, corroboration) but is not where its required content is.
- **irrelevant**: the record does not help with this unit, even if it
  shares terms with it.

For required and supporting records, also fill `supplies_evidence`:

- **yes**: the record text itself provides the content.
- **no**: the record only points at the content (a figure caption, a
  flattened formula, a clipped listing) that its text does not carry.

Leave `supplies_evidence` empty for irrelevant records. For an ambiguous
question, judge each conditional unit only for its own reading. Leave a
row empty if you cannot decide; an unjudged record is never counted as
irrelevant. Use `note` for anything unclear.
"""


@dataclass(frozen=True, slots=True)
class PdfPoolSummary:
    """Describe the review effort a pool requires."""

    entries: int
    unique_records: int
    words: int
    by_case: tuple[tuple[str, int, int], ...]


def anchor_pool_entries(
    golden: PdfMultibookGolden,
    library: PdfLibraryCorpus,
    case_ids: Sequence[str],
) -> tuple[PdfPoolEntry, ...]:
    """Offer every record on a verified locator page of each judged unit.

    A record holding a required anchor is marked anchor_overlap; other
    records on those pages, or on supporting-locator pages, are
    locator_page. Neither source assigns relevance.
    """
    cases = _selected_cases(golden, case_ids)
    records = library.source_records
    entries: list[PdfPoolEntry] = []
    for case in cases:
        for unit_id in pool_units(case):
            unit = golden.units[unit_id]
            sources: dict[str, set[PdfPoolSource]] = {}
            for locator in unit.locators:
                for record_id in anchor_records(locator, records):
                    sources.setdefault(record_id, set()).add(
                        PdfPoolSource.ANCHOR_OVERLAP,
                    )
            for locator in (*unit.locators, *unit.supporting_locators):
                for record_id in _page_records(locator, records):
                    sources.setdefault(record_id, set()).add(
                        PdfPoolSource.LOCATOR_PAGE,
                    )
            entries.extend(
                PdfPoolEntry(
                    case_id=case.case_id, unit_id=unit_id,
                    record_id=record_id, sources=_ordered(found),
                )
                for record_id, found in sources.items()
            )
    return tuple(entries)


def retrieval_pool_entries(
    golden: PdfMultibookGolden,
    rankings: Mapping[PdfPoolSource, Mapping[str, Sequence[str]]],
    depth: int,
    case_ids: Sequence[str],
) -> tuple[PdfPoolEntry, ...]:
    """Offer each retriever's top records for every judged unit of a case.

    rankings maps a retriever to its ranked record IDs per case. Every
    selected case must have a ranking from each of the three retrievers.
    """
    if depth <= 0:
        raise ValueError("Pool depth must be positive")
    if set(rankings) != set(RETRIEVAL_SOURCES):
        raise ValueError("Retrieval pooling needs BM25, MiniLM, and hybrid")
    entries: list[PdfPoolEntry] = []
    for case in _selected_cases(golden, case_ids):
        found: dict[str, set[PdfPoolSource]] = {}
        for source, by_case in rankings.items():
            if case.case_id not in by_case:
                raise ValueError(
                    f"{source.value} has no ranking for {case.case_id}"
                )
            for record_id in by_case[case.case_id][:depth]:
                found.setdefault(record_id, set()).add(source)
        entries.extend(
            PdfPoolEntry(
                case_id=case.case_id, unit_id=unit_id, record_id=record_id,
                sources=_ordered(sources),
            )
            for unit_id in pool_units(case)
            for record_id, sources in found.items()
        )
    return tuple(entries)


def build_judgment_pool(
    golden: PdfMultibookGolden,
    library: PdfLibraryCorpus,
    entries: Iterable[PdfPoolEntry],
    retrieval_depth: int | None = None,
) -> PdfJudgmentPool:
    """Merge entries by record and unit, in a deterministic order.

    Order follows the golden cases, each case's judged units, and the
    records' canonical reading order; it never reflects a ranking.
    """
    merged: dict[tuple[str, str, str], set[PdfPoolSource]] = {}
    for entry in entries:
        merged.setdefault(entry.key, set()).update(entry.sources)
    case_order = {case.case_id: index
                  for index, case in enumerate(golden.cases)}
    unit_order = {
        (case.case_id, unit_id): index
        for case in golden.cases
        for index, unit_id in enumerate(pool_units(case))
    }
    record_order = {record.record_id: index
                    for index, record in enumerate(library.source_records)}
    for key in merged:
        if key[2] not in record_order:
            raise ValueError(f"Pool names unknown record {key[2]}")
        if key[:2] not in unit_order:
            raise ValueError(f"Pool names no judged unit: {key[0]}/{key[1]}")
    keys = sorted(merged, key=lambda key: (
        case_order[key[0]], unit_order[key[:2]], record_order[key[2]],
    ))
    pool = PdfJudgmentPool(
        golden_content_fingerprint=golden.content_fingerprint,
        library_fingerprint=library.manifest.fingerprint,
        retrieval_depth=retrieval_depth,
        entries=tuple(
            PdfPoolEntry(case_id=key[0], unit_id=key[1], record_id=key[2],
                         sources=_ordered(merged[key]))
            for key in keys
        ),
    )
    require_pool_matches(pool, golden, library)
    return pool


def summarize_pool(
    pool: PdfJudgmentPool,
    library: PdfLibraryCorpus,
) -> PdfPoolSummary:
    """Count entries, distinct records, and words a reviewer must read."""
    texts = {record.record_id: record.text
             for record in library.source_records}
    unique = {entry.record_id for entry in pool.entries}
    by_case: dict[str, tuple[int, set[str]]] = {}
    for entry in pool.entries:
        count, records = by_case.get(entry.case_id, (0, set()))
        records.add(entry.record_id)
        by_case[entry.case_id] = (count + 1, records)
    return PdfPoolSummary(
        entries=len(pool.entries),
        unique_records=len(unique),
        words=sum(len(_WORDS.findall(texts[record])) for record in unique),
        by_case=tuple(
            (case_id, count, len(records))
            for case_id, (count, records) in by_case.items()
        ),
    )


def render_judgment_worksheet(
    golden: PdfMultibookGolden,
    library: PdfLibraryCorpus,
    pool: PdfJudgmentPool,
) -> tuple[str, str]:
    """Return a private Markdown worksheet and its blank CSV answer sheet.

    The worksheet shows each question, its evidence units with verified
    locators, and the full text of each offered record. Pool sources and
    rankings are deliberately omitted.
    """
    require_pool_matches(pool, golden, library)
    cases = {case.case_id: case for case in golden.cases}
    records = {record.record_id: record for record in library.source_records}
    lines = [
        "# Relevance judgment worksheet",
        "",
        "Private review material: questions and book text. Fill the CSV "
        "sheet; this file is for reading.",
        "",
        LABELING_GUIDE,
    ]
    rows = io.StringIO()
    writer = csv.writer(rows, lineterminator="\n")
    writer.writerow(WORKSHEET_COLUMNS)
    current: tuple[str, str] | None = None
    for number, entry in enumerate(pool.entries, start=1):
        item = f"W{number:04d}"
        if current is None or current[0] != entry.case_id:
            case = cases[entry.case_id]
            lines.extend([
                f"## {case.case_id}", "",
                f"**Question:** {case.question}", "",
                f"**Expected outcome:** {case.expected_outcome.value}; "
                f"source {case.source_support.value}, extracted text "
                f"{case.text_context_support.value}.",
                "",
            ])
        if current != (entry.case_id, entry.unit_id):
            lines.extend(_unit_lines(
                golden, entry.unit_id, cases[entry.case_id],
            ))
            current = (entry.case_id, entry.unit_id)
        record = records[entry.record_id]
        lines.extend(_record_lines(item, record))
        writer.writerow((item, entry.case_id, entry.unit_id,
                         entry.record_id, "", "", ""))
    return "\n".join(lines) + "\n", rows.getvalue()


def read_judgment_worksheet(
    sheet: str,
    pool: PdfJudgmentPool,
    *,
    dataset_id: str,
    reviewer: str,
    status: str,
) -> PdfMultibookJudgmentSet:
    """Turn a filled CSV sheet into judgments; blank rows stay unjudged."""
    if status not in {"pilot", "draft", "frozen"}:
        raise ValueError(f"Unknown judgment status {status}")
    expected = {
        f"W{number:04d}": entry
        for number, entry in enumerate(pool.entries, start=1)
    }
    reader = csv.DictReader(io.StringIO(sheet))
    if tuple(reader.fieldnames or ()) != WORKSHEET_COLUMNS:
        raise ValueError("Worksheet columns differ from the template")
    judgments = []
    seen = set()
    for row in reader:
        entry = expected.get(row["item"])
        if entry is None or (
            row["case_id"], row["unit_id"], row["record_id"]
        ) != entry.key:
            raise ValueError(f"Worksheet row {row['item']} does not match")
        if row["item"] in seen:
            raise ValueError(f"Worksheet repeats row {row['item']}")
        seen.add(row["item"])
        relevance = row["relevance"].strip().lower()
        supplies = row["supplies_evidence"].strip().lower()
        if not relevance:
            if supplies:
                raise ValueError(
                    f"{row['item']}: supplies_evidence without relevance"
                )
            continue
        judgments.append(PdfMultibookJudgment(
            case_id=entry.case_id,
            unit_id=entry.unit_id,
            record_id=entry.record_id,
            relevance=PdfRecordRelevance(relevance),
            supplies_evidence=_supplies(row["item"], supplies),
            reviewer=reviewer,
            note=row["note"].strip(),
        ))
    if seen != set(expected):
        raise ValueError("Worksheet rows are missing")
    return PdfMultibookJudgmentSet(
        dataset_id=dataset_id,
        golden_content_fingerprint=pool.golden_content_fingerprint,
        library_fingerprint=pool.library_fingerprint,
        pool_fingerprint=pool.fingerprint,
        status=cast(Literal["pilot", "draft", "frozen"], status),
        judgments=tuple(judgments),
    )


@dataclass(frozen=True, slots=True)
class PdfWorksheetCarryOver:
    """Hold labels moved to a regenerated pool and the row mapping."""

    sheet: str
    row_map: tuple[tuple[str, str, tuple[str, str, str]], ...]


def carry_over_worksheet(
    old_pool: PdfJudgmentPool,
    new_pool: PdfJudgmentPool,
    labelled_sheet: str,
    expected_sheet_sha256: str,
) -> PdfWorksheetCarryOver:
    """Move human labels to a regenerated pool, unchanged, by key.

    The old sheet must hash as recorded, both pools must name the same
    library, depth, and entries with the same sources, and every row must
    match its old pool entry. Labels and notes are copied verbatim; rows are
    renumbered in the new pool's order. Any difference stops the transfer.
    """
    digest = sha256(labelled_sheet.encode("utf-8")).hexdigest()
    if digest != expected_sheet_sha256:
        raise ValueError("Labelled worksheet differs from its recorded hash")
    if old_pool.library_fingerprint != new_pool.library_fingerprint:
        raise ValueError("Regenerated pool names a different library")
    if old_pool.retrieval_depth != new_pool.retrieval_depth:
        raise ValueError("Regenerated pool uses a different depth")
    old_sources = {entry.key: entry.sources for entry in old_pool.entries}
    new_sources = {entry.key: entry.sources for entry in new_pool.entries}
    if old_sources != new_sources:
        missing = sorted(set(old_sources) - set(new_sources))
        added = sorted(set(new_sources) - set(old_sources))
        raise ValueError(
            "Regenerated pool differs: "
            f"missing {missing[:3]}, added {added[:3]}, or changed sources"
        )
    old_items = {
        f"W{number:04d}": entry
        for number, entry in enumerate(old_pool.entries, start=1)
    }
    reader = csv.DictReader(io.StringIO(labelled_sheet))
    if tuple(reader.fieldnames or ()) != WORKSHEET_COLUMNS:
        raise ValueError("Worksheet columns differ from the template")
    rows: dict[tuple[str, str, str], tuple[str, dict[str, str]]] = {}
    for row in reader:
        entry = old_items.get(row["item"])
        key = (row["case_id"], row["unit_id"], row["record_id"])
        if entry is None or entry.key != key or key in rows:
            raise ValueError(f"Worksheet row {row['item']} does not match")
        rows[key] = (row["item"], row)
    if len(rows) != len(old_items):
        raise ValueError("Worksheet rows are missing")
    output = io.StringIO()
    writer = csv.DictWriter(
        output, fieldnames=list(WORKSHEET_COLUMNS), lineterminator="\n",
    )
    writer.writeheader()
    mapping = []
    for number, entry in enumerate(new_pool.entries, start=1):
        old_item, row = rows[entry.key]
        item = f"W{number:04d}"
        writer.writerow({**row, "item": item})
        mapping.append((old_item, item, entry.key))
    return PdfWorksheetCarryOver(output.getvalue(), tuple(mapping))


def _selected_cases(
    golden: PdfMultibookGolden,
    case_ids: Sequence[str],
) -> tuple[PdfMultibookCase, ...]:
    known = {case.case_id: case for case in golden.cases}
    unknown = [case_id for case_id in case_ids if case_id not in known]
    if unknown:
        raise ValueError(f"Unknown cases: {', '.join(unknown)}")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Pool cases must be unique")
    return tuple(known[case_id] for case_id in case_ids)


def _page_records(
    locator: PdfEvidenceLocator,
    records: Sequence[PdfIndexRecord],
) -> tuple[str, ...]:
    return tuple(
        record.record_id
        for record in records
        if record.book_path == locator.book_path
        and any(span.page_number == locator.physical_page
                for span in record.page_spans)
    )


def _ordered(sources: Iterable[PdfPoolSource]) -> tuple[PdfPoolSource, ...]:
    return tuple(sorted(set(sources), key=_SOURCE_ORDER.__getitem__))


def _supplies(item: str, value: str) -> bool | None:
    if not value:
        return None
    if value in _YES:
        return True
    if value in _NO:
        return False
    raise ValueError(f"{item}: supplies_evidence must be yes or no")


def _unit_lines(
    golden: PdfMultibookGolden,
    unit_id: str,
    case: PdfMultibookCase,
) -> list[str]:
    unit = golden.units[unit_id]
    lines = []
    for interpretation in case.interpretations:
        if unit_id in interpretation.unit_ids:
            lines.extend([
                f"### Interpretation `{interpretation.interpretation_id}`: "
                f"{interpretation.label}", "",
                "Judge only for this meaning, as if the user had chosen it.",
                "",
            ])
    lines += [
        f"### Evidence unit `{unit_id}` ({unit.modality.value})", "",
        f"Required content: {unit.required_content}", "",
        "Verified locators:",
    ]
    for locator in unit.locators:
        lines.append(
            f"- {locator.book_path}, physical page {locator.physical_page}: "
            f"`{locator.exact_anchor}`"
        )
    for locator in unit.supporting_locators:
        lines.append(
            f"- supporting, {locator.book_path}, physical page "
            f"{locator.physical_page}: `{locator.exact_anchor}`"
        )
    lines.extend(f"- note: {note}" for note in unit.notes)
    lines.append("")
    return lines


def _record_lines(item: str, record: PdfIndexRecord) -> list[str]:
    pages = sorted({span.page_number for span in record.page_spans})
    section = " > ".join(record.section_path) or "(no verified section)"
    return [
        f"#### {item}", "",
        f"{record.book_path}, physical pages "
        f"{', '.join(map(str, pages))}; section: {section}", "",
        "```text", record.text.rstrip("\n"), "```", "",
    ]
