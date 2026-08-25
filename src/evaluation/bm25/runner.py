"""Run one controlled BM25 configuration over a prepared fixed suite."""

from collections.abc import Sequence
from math import ceil
from statistics import median
from time import perf_counter_ns

from src.evaluation.bm25.models import (
    EvaluationSuite,
    ExperimentResult,
    QueryKind,
    QueryRunResult,
)
from src.evaluation.retrieval import (
    RetrievalMetrics,
    aggregate_query_metrics,
    evaluate_query,
)
from src.retrieval.bm25 import (
    BM25Document,
    BM25Index,
    BM25Parameters,
    BM25Retriever,
)


def run_experiment(
    suite: EvaluationSuite,
    documents: Sequence[BM25Document],
    run_id: str,
    parameters: BM25Parameters,
    *,
    latency_repetitions: int = 30,
) -> ExperimentResult:
    """Build, query, and measure one BM25 parameter configuration."""
    if latency_repetitions < 1:
        raise ValueError("Latency repetitions must be positive")

    build_started = perf_counter_ns()
    index = BM25Index(documents, parameters)
    build_time_ms = _milliseconds_since(build_started)
    retriever = BM25Retriever(index)

    query_results: list[QueryRunResult] = []
    all_latencies: list[float] = []
    for query in suite.queries:
        hits = retriever.search(query.question, top_k=10)
        metrics = evaluate_query(
            [hit.document.chunk for hit in hits],
            query.sources,
        )
        latencies = _measure_query(
            retriever,
            query.question,
            latency_repetitions,
        )
        all_latencies.extend(latencies)
        query_results.append(
            QueryRunResult(
                query=query,
                metrics=metrics,
                hits=tuple(hits),
                median_latency_ms=median(latencies),
                p95_latency_ms=_percentile(latencies, 0.95),
            )
        )

    return ExperimentResult(
        run_id=run_id,
        suite_name=suite.name,
        parameters=parameters,
        documentation_metrics=_aggregate_kind(
            query_results,
            QueryKind.DOCUMENTATION,
        ),
        code_metrics=_aggregate_kind(query_results, QueryKind.CODE),
        build_time_ms=build_time_ms,
        median_latency_ms=median(all_latencies),
        p95_latency_ms=_percentile(all_latencies, 0.95),
        query_results=tuple(query_results),
    )


def _measure_query(
    retriever: BM25Retriever,
    question: str,
    repetitions: int,
) -> list[float]:
    """Warm up once, then collect independent query latency samples."""
    retriever.search(question, top_k=10)
    samples: list[float] = []
    for _ in range(repetitions):
        started = perf_counter_ns()
        retriever.search(question, top_k=10)
        samples.append(_milliseconds_since(started))
    return samples


def _aggregate_kind(
    query_results: Sequence[QueryRunResult],
    kind: QueryKind,
) -> RetrievalMetrics:
    """Aggregate one required docs or code partition independently."""
    return aggregate_query_metrics(
        [
            result.metrics
            for result in query_results
            if result.query.kind is kind
        ]
    )


def _percentile(values: Sequence[float], fraction: float) -> float:
    """Return a deterministic nearest-rank percentile."""
    ordered = sorted(values)
    index = max(0, ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def _milliseconds_since(started_ns: int) -> float:
    """Convert a monotonic nanosecond duration to milliseconds."""
    return (perf_counter_ns() - started_ns) / 1_000_000
