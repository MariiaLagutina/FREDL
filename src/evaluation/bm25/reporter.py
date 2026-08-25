"""Render reproducible BM25 experiment summaries for the terminal."""

from collections.abc import Sequence

from src.evaluation.bm25.models import ExperimentResult, QueryRunResult
from src.evaluation.retrieval import sources_match


def print_results(
    results: Sequence[ExperimentResult],
    *,
    verbose: bool = False,
) -> None:
    """Print a compact comparison and optional per-query explanations."""
    if not results:
        raise ValueError("BM25 reporter requires at least one experiment")
    _print_table(results)
    if len(results) > 1:
        _print_changes(results[0], results[1:])
    if verbose:
        for result in results:
            _print_query_details(result)


def _print_table(results: Sequence[ExperimentResult]) -> None:
    """Print required docs, code, and latency metrics per run."""
    first = results[0]
    print(
        f"Suite: {first.suite_name}; "
        f"files={first.source_file_count}; "
        f"chunks={first.document_count}"
    )
    header = (
        "Run  k1   b     meta  "
        "Docs R@1 R@3 R@5 R@10 MRR  "
        "Code R@1 R@3 R@5 R@10 MRR  "
        "Buildms IndexKB PeakKB P50ms  P95ms"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        docs = result.documentation_metrics
        code = result.code_metrics
        parameters = result.parameters
        print(
            f"{result.run_id:<4} "
            f"{parameters.k1:<4.1f} "
            f"{parameters.b:<5.2f} "
            f"{parameters.metadata_weight:<5.1f} "
            f"{docs.recall_at_1:<4.2f} "
            f"{docs.recall_at_3:<4.2f} "
            f"{docs.recall_at_5:<4.2f} "
            f"{docs.recall_at_10:<5.2f} "
            f"{docs.mean_reciprocal_rank:<4.2f} "
            f"{code.recall_at_1:<4.2f} "
            f"{code.recall_at_3:<4.2f} "
            f"{code.recall_at_5:<4.2f} "
            f"{code.recall_at_10:<5.2f} "
            f"{code.mean_reciprocal_rank:<4.2f} "
            f"{result.build_time_ms:<7.3f} "
            f"{result.index_size_bytes / 1024:<7.1f} "
            f"{result.peak_build_memory_bytes / 1024:<6.1f} "
            f"{result.median_latency_ms:<6.3f} "
            f"{result.p95_latency_ms:<6.3f}"
        )


def _print_changes(
    control: ExperimentResult,
    candidates: Sequence[ExperimentResult],
) -> None:
    """Compare per-query Recall@5 against the named control run."""
    control_values = {
        result.query.query_id: result.metrics.recall_at_5
        for result in control.query_results
    }
    print(f"\nRecall@5 changes against {control.run_id}:")
    for candidate in candidates:
        changes = [
            _compare(
                result.metrics.recall_at_5,
                control_values[result.query.query_id],
            )
            for result in candidate.query_results
        ]
        print(
            f"{candidate.run_id}: "
            f"improved={changes.count(1)} "
            f"regressed={changes.count(-1)} "
            f"unchanged={changes.count(0)}"
        )
    print(f"\nFirst-relevant rank changes against {control.run_id}:")
    control_ranks = {
        result.query.query_id: result.metrics.reciprocal_rank
        for result in control.query_results
    }
    for candidate in candidates:
        changes = [
            _compare(
                result.metrics.reciprocal_rank,
                control_ranks[result.query.query_id],
            )
            for result in candidate.query_results
        ]
        print(
            f"{candidate.run_id}: "
            f"improved={changes.count(1)} "
            f"regressed={changes.count(-1)} "
            f"unchanged={changes.count(0)}"
        )


def _print_query_details(result: ExperimentResult) -> None:
    """Print source labels, ranks, and separate field scores."""
    print(f"\n{result.run_id} query details:")
    for query_result in result.query_results:
        print(
            f"\n[{query_result.query.kind.value}] "
            f"{query_result.query.query_id}: "
            f"{query_result.query.question}"
        )
        print(
            "Expected: "
            + ", ".join(
                f"{source.file_path} "
                f"[{source.first_character_index}, "
                f"{source.last_character_index})"
                for source in query_result.query.sources
            )
        )
        print(
            f"First relevant rank: {_first_relevant_rank(query_result)}; "
            f"Recall@5={query_result.metrics.recall_at_5:.2f}"
        )
        for rank, hit in enumerate(query_result.hits[:3], start=1):
            marker = "relevant" if any(
                sources_match(hit.document.chunk, source)
                for source in query_result.query.sources
            ) else "not relevant"
            print(
                f"  {rank}. {hit.document.chunk.file_path} "
                f"total={hit.score:.4f} "
                f"content={hit.content_score:.4f} "
                f"metadata={hit.metadata_score:.4f} "
                f"({marker})"
            )


def _first_relevant_rank(result: QueryRunResult) -> int | None:
    """Return the first matching rank for terminal diagnostics."""
    for rank, hit in enumerate(result.hits, start=1):
        if any(
            sources_match(hit.document.chunk, source)
            for source in result.query.sources
        ):
            return rank
    return None


def _compare(candidate: float, control: float) -> int:
    """Return a compact direction for one per-query metric change."""
    return (candidate > control) - (candidate < control)
