"""Tests for immutable chunk audit result models."""

import pytest

from src.ingestion import (
    ChunkAuditIssue,
    ChunkAuditIssueKind,
    ChunkAuditReport,
    ChunkSizeSummary,
)


def make_summary() -> ChunkSizeSummary:
    """Create one valid representative size distribution."""
    return ChunkSizeSummary(
        count=10,
        minimum=12,
        median=240.5,
        p95=900,
        maximum=1200,
    )


def test_successful_report_exposes_derived_counts() -> None:
    """A report derives chunk count and pass status from its contents."""
    report = ChunkAuditReport(
        document_count=4,
        size_summary=make_summary(),
    )

    assert report.chunk_count == 10
    assert report.passed


def test_empty_report_has_no_chunk_distribution() -> None:
    """An audit without source content can represent zero chunks."""
    report = ChunkAuditReport(document_count=0, size_summary=None)

    assert report.chunk_count == 0
    assert report.passed


def test_issue_makes_report_fail() -> None:
    """One actionable invariant failure changes the report status."""
    issue = ChunkAuditIssue(
        kind=ChunkAuditIssueKind.OVERSIZED,
        file_path="docs/guide.md",
        chunk_index=3,
        detail="Chunk length 2100 exceeds maximum 2000",
    )

    report = ChunkAuditReport(
        document_count=1,
        size_summary=make_summary(),
        issues=(issue,),
    )

    assert not report.passed
    assert report.issues[0].chunk_index == 3


@pytest.mark.parametrize(
    ("minimum", "median", "p95", "maximum"),
    [
        (0, 1.0, 2, 3),
        (5, 4.0, 6, 7),
        (1, 5.0, 4, 7),
        (1, 4.0, 8, 7),
    ],
)
def test_size_summary_rejects_invalid_distribution(
    minimum: int,
    median: float,
    p95: int,
    maximum: int,
) -> None:
    """Size statistics must describe positive ordered chunks."""
    with pytest.raises(ValueError):
        ChunkSizeSummary(
            count=1,
            minimum=minimum,
            median=median,
            p95=p95,
            maximum=maximum,
        )


def test_issue_rejects_negative_chunk_index() -> None:
    """An issue cannot point outside the produced chunk sequence."""
    with pytest.raises(
        ValueError,
        match="Audit issue chunk index must not be negative",
    ):
        ChunkAuditIssue(
            ChunkAuditIssueKind.EMPTY_TEXT,
            "notes.txt",
            -1,
            "Chunk contains whitespace only",
        )
