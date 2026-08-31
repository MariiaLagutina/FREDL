"""Immutable metrics for one query and an aggregated retrieval dataset."""

from dataclasses import dataclass
from enum import Enum

from src.models import MinimalSource


class RetrievalDatasetKind(str, Enum):
    """Identify one independently reported retrieval dataset."""

    DOCS = "Docs"
    CODE = "Code"


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationCase:
    """Align one labelled question with its ranked retrieved sources."""

    question_id: str
    question: str
    references: tuple[MinimalSource, ...]
    retrieved: tuple[MinimalSource, ...]


@dataclass(frozen=True, slots=True)
class RetrievalQueryMetrics:
    """Store source recall and first-relevant rank for one query."""

    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    """Store mean retrieval metrics for one fixed query group."""

    query_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mean_reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationReport:
    """Label aggregate metrics for one retrieval dataset."""

    dataset: RetrievalDatasetKind
    metrics: RetrievalMetrics
