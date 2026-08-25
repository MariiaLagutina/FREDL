"""Persist complete BM25 experiment evidence as machine-readable JSON."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

from src.evaluation.bm25.models import ExperimentResult, QueryRunResult
from src.evaluation.bm25.suite import fingerprint_suite
from src.evaluation.retrieval import sources_match


def write_json_report(
    output_path: Path,
    suite_root: Path,
    results: list[ExperimentResult],
) -> None:
    """Write parameters, environment, metrics, and rankings to JSON."""
    if not results:
        raise ValueError("BM25 report requires at least one experiment")
    git_commit, git_dirty = _git_state()
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "suite_name": results[0].suite_name,
        "suite_fingerprint": fingerprint_suite(suite_root),
        "environment": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "runs": [_serialize_result(result) for result in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _serialize_result(result: ExperimentResult) -> dict[str, Any]:
    """Convert one immutable experiment result to JSON-ready values."""
    return {
        "run_id": result.run_id,
        "parameters": asdict(result.parameters),
        "source_file_count": result.source_file_count,
        "document_count": result.document_count,
        "documentation_metrics": asdict(result.documentation_metrics),
        "code_metrics": asdict(result.code_metrics),
        "build_time_ms": result.build_time_ms,
        "index_size_bytes": result.index_size_bytes,
        "peak_build_memory_bytes": result.peak_build_memory_bytes,
        "median_latency_ms": result.median_latency_ms,
        "p95_latency_ms": result.p95_latency_ms,
        "queries": [
            _serialize_query(query_result)
            for query_result in result.query_results
        ],
    }


def _serialize_query(result: QueryRunResult) -> dict[str, Any]:
    """Convert one query, its labels, metrics, and ranking to JSON."""
    return {
        "query_id": result.query.query_id,
        "kind": result.query.kind.value,
        "question": result.query.question,
        "sources": [source.model_dump() for source in result.query.sources],
        "metrics": asdict(result.metrics),
        "median_latency_ms": result.median_latency_ms,
        "p95_latency_ms": result.p95_latency_ms,
        "hits": [
            {
                "rank": rank,
                "file_path": hit.document.chunk.file_path,
                "start": hit.document.chunk.start,
                "end": hit.document.chunk.end,
                "score": hit.score,
                "content_score": hit.content_score,
                "metadata_score": hit.metadata_score,
                "relevant": any(
                    sources_match(hit.document.chunk, source)
                    for source in result.query.sources
                ),
            }
            for rank, hit in enumerate(result.hits, start=1)
        ],
    }


def _git_state() -> tuple[str, bool | None]:
    """Return commit and dirty state without making report creation fail."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None
    return commit.stdout.strip(), bool(status.stdout.strip())
