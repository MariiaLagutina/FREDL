"""Validated inputs and immutable outputs for BM25 experiments."""

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel

from src.evaluation.retrieval import RetrievalMetrics, RetrievalQueryMetrics
from src.models import MinimalSource
from src.retrieval.bm25 import BM25Hit, BM25Parameters


class QueryKind(str, Enum):
    """Separate documentation and source-code retrieval queries."""

    DOCUMENTATION = "documentation"
    CODE = "code"


class EvaluationQuery(BaseModel):
    """Describe one fixed query and its relevant source spans."""

    query_id: str
    kind: QueryKind
    question: str
    sources: list[MinimalSource]


class EvaluationSuite(BaseModel):
    """Describe one versioned corpus and its retrieval questions."""

    name: str
    max_chunk_size: int
    queries: list[EvaluationQuery]


@dataclass(frozen=True, slots=True)
class QueryRunResult:
    """Keep one query's metrics, ranking, and measured latency."""

    query: EvaluationQuery
    metrics: RetrievalQueryMetrics
    hits: tuple[BM25Hit, ...]
    median_latency_ms: float
    p95_latency_ms: float


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Keep one reproducible parameter run and separated metrics."""

    run_id: str
    suite_name: str
    parameters: BM25Parameters
    documentation_metrics: RetrievalMetrics
    code_metrics: RetrievalMetrics
    source_file_count: int
    document_count: int
    build_time_ms: float
    index_size_bytes: int
    peak_build_memory_bytes: int
    median_latency_ms: float
    p95_latency_ms: float
    query_results: tuple[QueryRunResult, ...]
