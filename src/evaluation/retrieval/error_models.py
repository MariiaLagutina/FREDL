"""Immutable models for classifying retrieval misses."""

from dataclasses import dataclass
from enum import Enum

from src.evaluation.retrieval.models import RetrievalDatasetKind
from src.models import MinimalSource


class RetrievalErrorCategory(str, Enum):
    """Identify the dominant reason for one top-five retrieval miss."""

    WRONG_FILE = "wrong_file"
    RELEVANT_BELOW_TOP_5 = "relevant_below_top_5"
    CHUNK_BOUNDARY = "chunk_boundary"
    LOST_IDENTIFIER = "lost_identifier"
    PARAPHRASE = "paraphrase"
    DUPLICATE_RESULTS = "duplicate_results"
    NOISY_METADATA = "noisy_metadata"


@dataclass(frozen=True, slots=True)
class RetrievalMissEvidence:
    """Capture ranked evidence for one question missed in the top five."""

    question_id: str
    question: str
    references: tuple[MinimalSource, ...]
    retrieved: tuple[MinimalSource, ...]
    relevant_rank: int | None

    def __post_init__(self) -> None:
        """Keep top-five miss evidence internally consistent."""
        if not self.question_id.strip():
            raise ValueError("Retrieval miss question ID must not be empty")
        if not self.question.strip():
            raise ValueError("Retrieval miss question must not be empty")
        if not self.references:
            raise ValueError("Retrieval miss must have reference sources")
        if self.relevant_rank is not None:
            if self.relevant_rank <= 5:
                raise ValueError(
                    "Relevant rank for a top-five miss must be greater "
                    "than five"
                )
            if self.relevant_rank > len(self.retrieved):
                raise ValueError(
                    "Relevant rank must refer to a retrieved source"
                )


@dataclass(frozen=True, slots=True)
class RetrievalMissAnalysis:
    """Record one classified top-five miss and its follow-up action."""

    question_id: str
    question: str
    category: RetrievalErrorCategory
    relevant_rank: int | None
    hypothesis: str
    proposed_fix: str
    next_test: str

    def __post_init__(self) -> None:
        """Require actionable evidence for every classified miss."""
        required_text = {
            "question ID": self.question_id,
            "question": self.question,
            "hypothesis": self.hypothesis,
            "proposed fix": self.proposed_fix,
            "next test": self.next_test,
        }
        for field_name, value in required_text.items():
            if not value.strip():
                message = f"Retrieval miss {field_name} must not be empty"
                raise ValueError(message)

        if self.relevant_rank is not None and self.relevant_rank <= 5:
            raise ValueError(
                "Relevant rank for a top-five miss must be greater than five"
            )


@dataclass(frozen=True, slots=True)
class RetrievalErrorAnalysisReport:
    """Group classified misses for one independently analyzed dataset."""

    dataset: RetrievalDatasetKind
    misses: tuple[RetrievalMissAnalysis, ...]

    def __post_init__(self) -> None:
        """Reject reports that classify the same question more than once."""
        question_ids = [miss.question_id for miss in self.misses]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError(
                "Retrieval error analysis question IDs must be unique"
            )

    @property
    def category_counts(
        self,
    ) -> tuple[tuple[RetrievalErrorCategory, int], ...]:
        """Count misses by category in a stable presentation order."""
        return tuple(
            (
                category,
                sum(miss.category is category for miss in self.misses),
            )
            for category in RetrievalErrorCategory
        )

    @property
    def dominant_category(self) -> RetrievalErrorCategory | None:
        """Return the most frequent category, using enum order for ties."""
        if not self.misses:
            return None
        return max(self.category_counts, key=lambda item: item[1])[0]
