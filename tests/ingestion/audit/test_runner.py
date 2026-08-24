"""Tests for deterministic chunk invariant audits."""

from src.ingestion import (
    Chunk,
    ChunkAuditIssueKind,
    FileKind,
    SourceDocument,
    audit_documents,
)


def test_real_chunkers_pass_all_audit_invariants() -> None:
    """The orchestrator produces bounded exact deterministic chunks."""
    documents = [
        SourceDocument(
            "src/example.py",
            FileKind.PYTHON,
            "def add(left, right):\n    return left + right\n",
        ),
        SourceDocument(
            "docs/guide.md",
            FileKind.TEXT,
            "# Guide\n\nFirst paragraph.\n\nSecond paragraph.\n",
        ),
    ]

    report = audit_documents(documents, max_chunk_size=30)

    assert report.document_count == 2
    assert report.chunk_count > 0
    assert report.size_summary is not None
    assert report.size_summary.maximum <= 30
    assert report.passed


def test_audit_reports_each_failed_chunk_invariant() -> None:
    """Invalid ranges, excessive size, and empty text remain distinct."""
    document = SourceDocument("notes.txt", FileKind.TEXT, "content")

    def faulty_chunker(
        source: SourceDocument,
        maximum: int,
    ) -> list[Chunk]:
        del maximum
        return [
            Chunk(source.file_path, 0, 9, "different"),
            Chunk(source.file_path, 0, 3, "   "),
        ]

    report = audit_documents(
        [document],
        max_chunk_size=5,
        chunker=faulty_chunker,
    )

    assert {issue.kind for issue in report.issues} == {
        ChunkAuditIssueKind.INVALID_RANGE,
        ChunkAuditIssueKind.OVERSIZED,
        ChunkAuditIssueKind.TEXT_MISMATCH,
        ChunkAuditIssueKind.EMPTY_TEXT,
    }
    assert not report.passed


def test_audit_detects_nondeterministic_chunking() -> None:
    """Repeated calls must return the same chunks in the same order."""
    document = SourceDocument("notes.txt", FileKind.TEXT, "abcdef")
    calls = 0

    def alternating_chunker(
        source: SourceDocument,
        maximum: int,
    ) -> list[Chunk]:
        nonlocal calls
        del maximum
        calls += 1
        end = 3 if calls % 2 else 4
        return [Chunk(source.file_path, 0, end, source.text[:end])]

    report = audit_documents([document], chunker=alternating_chunker)

    assert [issue.kind for issue in report.issues] == [
        ChunkAuditIssueKind.NONDETERMINISTIC
    ]


def test_audit_uses_nearest_rank_p95() -> None:
    """The summary reports stable statistics for a known distribution."""
    document = SourceDocument("sizes.txt", FileKind.TEXT, "x" * 210)

    def sized_chunker(
        source: SourceDocument,
        maximum: int,
    ) -> list[Chunk]:
        del maximum
        chunks = []
        start = 0
        for size in range(1, 21):
            end = start + size
            chunks.append(
                Chunk(source.file_path, start, end, source.text[start:end])
            )
            start = end
        return chunks

    report = audit_documents([document], chunker=sized_chunker)

    assert report.size_summary is not None
    assert report.size_summary.minimum == 1
    assert report.size_summary.median == 10.5
    assert report.size_summary.p95 == 19
    assert report.size_summary.maximum == 20


def test_audit_rejects_non_positive_chunk_limit() -> None:
    """An audit cannot validate chunks against an invalid limit."""
    try:
        audit_documents([], max_chunk_size=0)
    except ValueError as error:
        assert str(error) == "Maximum chunk size must be positive"
    else:
        raise AssertionError("Expected ValueError")
