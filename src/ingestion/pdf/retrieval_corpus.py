"""Select PDF records for retrieval without altering the source records."""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import re

from src.ingestion.pdf.index_records import PdfIndexRecord
from src.ingestion.pdf.outline import PdfOutlineEntry
from src.ingestion.pdf.sections import PdfSectionResolution


PDF_RETRIEVAL_CORPUS_VERSION = "pdf-retrieval-corpus-v1"

# Navigation shape is judged over a whole section interval, never over one
# short record, so a record holding only a heading cannot qualify. Every
# condition narrows navigation; uncertain intervals stay searchable.
MIN_NAVIGATION_LINES = 20
MIN_INDEX_REFERENCE_RATIO = 0.2
# An index is alphabetical. Sub-entries add noise, so the trend is measured
# on the median initial letter of consecutive bins of reference lines.
INDEX_TREND_BINS = 10
MIN_INDEX_LETTER_ENTRIES = 10
MIN_INDEX_DISTINCT_INITIALS = 5
MIN_INDEX_ALPHABET_TREND = 0.5
# Page locators cannot exceed the book's last page; years and data cannot
# pass as an index.
MAX_INDEX_OUT_OF_RANGE_LOCATORS = 0.05
MIN_CONTENTS_TITLE_RATIO = 0.6
MIN_CONTENTS_DISTINCT_TITLES = 10

# Repeated boilerplate is a block of consecutive substantial lines that
# recurs unchanged across most top-level sections of one book.
BOILERPLATE_WINDOW_LINES = 3
MIN_BOILERPLATE_LINE_LENGTH = 20
MIN_BOILERPLATE_WINDOW_LENGTH = 120
MIN_BOILERPLATE_SECTIONS = 3
MIN_BOILERPLATE_SECTION_SHARE = 0.5

_PAGE_LOCATOR = re.compile(
    r",\s*(?:\d+(?:[-–]\d+)?|[ivxlc]+)(?:,\s*\d+(?:[-–]\d+)?)*\s*$"
)
_TRAILING_PAGE = re.compile(r"[\s.]*\d+$")
_TITLE_NUMBER = re.compile(r"^(?:\d+|[A-Z])\.\s*")
_NUMBER = re.compile(r"\d+")
_WHITESPACE = re.compile(r"\s+")
_HYPHENS = str.maketrans({"‐": "-", "‑": "-"})


class PdfRecordRole(str, Enum):
    """Describe what kind of source text a record holds.

    Only INDEX and TABLE_OF_CONTENTS are navigation-only. Unsectioned front
    matter precedes every verified heading; it is reported, not excluded.
    """

    CONTENT = "content"
    INDEX = "index"
    TABLE_OF_CONTENTS = "table_of_contents"
    UNSECTIONED_FRONT_MATTER = "unsectioned_front_matter"


NAVIGATION_ROLES: frozenset[PdfRecordRole] = frozenset(
    {PdfRecordRole.INDEX, PdfRecordRole.TABLE_OF_CONTENTS}
)


class PdfRetrievalPolicy(str, Enum):
    """Choose which records and record text enter retrieval indexes."""

    ALL_RECORDS = "all_records"
    EXCLUDE_NAVIGATION = "exclude_navigation"
    EXCLUDE_NAVIGATION_AND_BOILERPLATE = "exclude_navigation_and_boilerplate"


@dataclass(frozen=True, slots=True)
class PdfRecordClassification:
    """Store one record's role and its repeated boilerplate text ranges.

    Ranges are half-open offsets into the unchanged record text.
    """

    record_id: str
    role: PdfRecordRole
    boilerplate_ranges: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class PdfRetrievalCorpus:
    """Hold eligible records, in source order, and their searchable text.

    retrieval_texts aligns with records. A text differs from its record's
    text only when the policy blanks confirmed boilerplate lines; offsets
    still match the record text.
    """

    policy: PdfRetrievalPolicy
    records: tuple[PdfIndexRecord, ...]
    retrieval_texts: tuple[str, ...]
    excluded_record_ids: tuple[str, ...]

    @property
    def fingerprint(self) -> str:
        """Hash everything that can change a ranking over this corpus."""
        digest = sha256()
        for value in (PDF_RETRIEVAL_CORPUS_VERSION, self.policy.value):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
        for record, text in zip(self.records, self.retrieval_texts):
            for value in (record.record_id, text):
                digest.update(value.encode("utf-8"))
                digest.update(b"\0")
        return digest.hexdigest()


def classify_pdf_records(
    records: Sequence[PdfIndexRecord],
    outline: Sequence[PdfOutlineEntry],
) -> tuple[PdfRecordClassification, ...]:
    """Classify records of one book from observable structure.

    Navigation roles depend on the shape of a whole section interval: index
    lines end in page or section references, and contents lines repeat
    outline titles. Titles are never matched against a fixed keyword list.
    """
    _require_one_book(records)
    titles = _outline_titles(outline)
    roles = _interval_roles(records, titles)
    boilerplate = _boilerplate_ranges(records)
    return tuple(
        PdfRecordClassification(
            record_id=record.record_id,
            role=roles[record.record_id],
            boilerplate_ranges=boilerplate.get(record.record_id, ()),
        )
        for record in records
    )


def build_pdf_retrieval_corpus(
    records: Sequence[PdfIndexRecord],
    classifications: Sequence[PdfRecordClassification],
    policy: PdfRetrievalPolicy,
) -> PdfRetrievalCorpus:
    """Apply one policy; excluded records stay available to the caller."""
    by_id = {item.record_id: item for item in classifications}
    if len(by_id) != len(classifications) or set(by_id) != {
        record.record_id for record in records
    }:
        raise ValueError("PDF record classifications must match the records")

    eligible: list[PdfIndexRecord] = []
    texts: list[str] = []
    excluded: list[str] = []
    for record in records:
        classification = by_id[record.record_id]
        text = record.text
        if policy is not PdfRetrievalPolicy.ALL_RECORDS and (
            classification.role in NAVIGATION_ROLES
        ):
            excluded.append(record.record_id)
            continue
        if policy is PdfRetrievalPolicy.EXCLUDE_NAVIGATION_AND_BOILERPLATE:
            text = _without_ranges(text, classification.boilerplate_ranges)
            if not text.strip():
                excluded.append(record.record_id)
                continue
        eligible.append(record)
        texts.append(text)
    return PdfRetrievalCorpus(
        policy=policy,
        records=tuple(eligible),
        retrieval_texts=tuple(texts),
        excluded_record_ids=tuple(excluded),
    )


def _interval_roles(
    records: Sequence[PdfIndexRecord],
    titles: frozenset[str],
) -> dict[str, PdfRecordRole]:
    groups: dict[str, list[PdfIndexRecord]] = defaultdict(list)
    for record in records:
        key = record.section_interval_id or f"record:{record.record_id}"
        groups[key].append(record)
    last_page = max(
        (span.page_number for record in records for span in record.page_spans),
        default=0,
    )
    first_verified = min(
        (
            (record.flow_index, record.flow_start)
            for record in records
            if record.section_resolution is PdfSectionResolution.VERIFIED
        ),
        default=None,
    )

    roles = {}
    for group in groups.values():
        role = _navigation_role(group, titles, last_page)
        if role is None:
            role = (
                PdfRecordRole.UNSECTIONED_FRONT_MATTER
                if _precedes(group[0], first_verified)
                else PdfRecordRole.CONTENT
            )
        for record in group:
            roles[record.record_id] = role
    return roles


def _navigation_role(
    group: Sequence[PdfIndexRecord],
    titles: frozenset[str],
    last_page: int,
) -> PdfRecordRole | None:
    lines = [
        line.strip()
        for record in group
        for line in record.text.splitlines()
        if line.strip()
    ]
    if len(lines) < MIN_NAVIGATION_LINES:
        return None
    references = [line for line in lines if _is_reference_line(line, titles)]
    if (
        len(references) / len(lines) >= MIN_INDEX_REFERENCE_RATIO
        and _is_alphabetical(references)
        and _locators_within_book(references, last_page)
    ):
        return PdfRecordRole.INDEX
    title_lines = [
        title for title in (
            _normalize(_TRAILING_PAGE.sub("", line)) for line in lines
        )
        if title in titles
    ]
    if (
        len(title_lines) / len(lines) >= MIN_CONTENTS_TITLE_RATIO
        and len(set(title_lines)) >= MIN_CONTENTS_DISTINCT_TITLES
    ):
        return PdfRecordRole.TABLE_OF_CONTENTS
    return None


def _is_alphabetical(references: Sequence[str]) -> bool:
    """Return whether entry initials rise from the start to the end."""
    initials = [
        line[0].casefold() for line in references if line[0].isalpha()
    ]
    if (
        len(initials) < MIN_INDEX_LETTER_ENTRIES
        or len(set(initials)) < MIN_INDEX_DISTINCT_INITIALS
    ):
        return False
    size = len(initials) / INDEX_TREND_BINS
    medians = [
        sorted(part)[len(part) // 2]
        for part in (
            initials[int(index * size):int((index + 1) * size)]
            for index in range(INDEX_TREND_BINS)
        )
        if part
    ]
    return _rank_correlation(medians) >= MIN_INDEX_ALPHABET_TREND


def _rank_correlation(values: Sequence[str]) -> float:
    """Return Spearman's correlation between values and their positions."""
    ordered = sorted(set(values))
    if len(ordered) < 2:
        return 0.0
    ties: dict[str, list[int]] = defaultdict(list)
    for position, value in enumerate(sorted(values)):
        ties[value].append(position)
    rank = {value: sum(items) / len(items) for value, items in ties.items()}
    ranks = [rank[value] for value in values]
    mean = (len(values) - 1) / 2
    covariance = sum(
        (position - mean) * (value - mean)
        for position, value in enumerate(ranks)
    )
    spread = sum((value - mean) ** 2 for value in ranks)
    positions = sum((position - mean) ** 2 for position in range(len(ranks)))
    return float(covariance / (spread * positions) ** 0.5)


def _locators_within_book(references: Sequence[str], last_page: int) -> bool:
    """Reject numeric references that cannot be pages of this book."""
    numbers = [
        int(number)
        for line in references
        if (locator := _PAGE_LOCATOR.search(line)) is not None
        for number in _NUMBER.findall(locator.group())
    ]
    if not numbers:
        return True
    beyond = sum(1 for number in numbers if number > last_page)
    return beyond / len(numbers) <= MAX_INDEX_OUT_OF_RANGE_LOCATORS


def _is_reference_line(line: str, titles: frozenset[str]) -> bool:
    """Return whether a line ends in page numbers or section references."""
    if _PAGE_LOCATOR.search(line):
        return True
    if ", " not in line:
        return False
    references = line.split(", ", 1)[1]
    return any(
        _normalize(part) in titles
        for item in references.split(", ")
        for part in item.split("-")
        if part.strip()
    )


def _precedes(
    record: PdfIndexRecord,
    first_verified: tuple[int, int] | None,
) -> bool:
    """Return whether unsectioned text lies before the first verified one."""
    return (
        first_verified is not None
        and record.section_resolution is PdfSectionResolution.NONE
        and (record.flow_index, record.flow_start) < first_verified
    )


def _boilerplate_ranges(
    records: Sequence[PdfIndexRecord],
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Locate line blocks repeated unchanged across most top-level units."""
    units: dict[tuple[str, ...], set[tuple[str, ...]]] = defaultdict(set)
    record_windows: dict[str, list[tuple[tuple[str, ...], int, int]]] = {}
    all_units: set[tuple[str, ...]] = set()
    for record in records:
        unit = _top_level_unit(record)
        all_units.add(unit)
        windows = list(_line_windows(record.text))
        record_windows[record.record_id] = windows
        for block, _, _ in windows:
            units[block].add(unit)

    required = max(
        MIN_BOILERPLATE_SECTIONS,
        MIN_BOILERPLATE_SECTION_SHARE * len(all_units),
    )
    confirmed = {block for block, found in units.items()
                 if len(found) >= required}
    ranges = {}
    for record_id, windows in record_windows.items():
        spans = [(start, end) for block, start, end in windows
                 if block in confirmed]
        if spans:
            ranges[record_id] = _merge_ranges(spans)
    return ranges


def _line_windows(
    text: str,
) -> list[tuple[tuple[str, ...], int, int]]:
    """Return substantial consecutive line blocks with their text range."""
    lines = []
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        lines.append((raw_line.strip(), offset, offset + len(raw_line)))
        offset += len(raw_line)

    windows = []
    for index in range(len(lines) - BOILERPLATE_WINDOW_LINES + 1):
        window = lines[index:index + BOILERPLATE_WINDOW_LINES]
        block = tuple(line for line, _, _ in window)
        if (
            all(len(line) >= MIN_BOILERPLATE_LINE_LENGTH for line in block)
            and sum(map(len, block)) >= MIN_BOILERPLATE_WINDOW_LENGTH
        ):
            windows.append((block, window[0][1], window[-1][2]))
    return windows


def _top_level_unit(record: PdfIndexRecord) -> tuple[str, ...]:
    if record.section_path:
        return ("section", record.section_path[0])
    return ("interval", record.section_interval_id or record.record_id)


def _merge_ranges(
    ranges: list[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return tuple(merged)


def _without_ranges(
    text: str,
    ranges: tuple[tuple[int, int], ...],
) -> str:
    """Blank boilerplate lines while keeping every offset and line break."""
    characters = list(text)
    for start, end in ranges:
        for index in range(start, end):
            if characters[index] not in "\r\n":
                characters[index] = " "
    return "".join(characters)


def _outline_titles(outline: Sequence[PdfOutlineEntry]) -> frozenset[str]:
    titles = set()
    for entry in outline:
        for title in (entry.title, _TITLE_NUMBER.sub("", entry.title)):
            normalized = _normalize(title)
            if normalized:
                titles.add(normalized)
    return frozenset(titles)


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", text.translate(_HYPHENS)).strip().casefold()


def _require_one_book(records: Sequence[PdfIndexRecord]) -> None:
    if len({record.book_path for record in records}) > 1:
        raise ValueError("PDF record classification requires one book")
    if len({record.record_id for record in records}) != len(records):
        raise ValueError("PDF record classification requires unique IDs")
