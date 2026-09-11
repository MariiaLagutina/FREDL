"""Source-by-source binary evidence selection."""

from src.generation.evidence.dataset import (
    EvidenceDatasetSplit,
    EvidenceExampleKind,
    EvidenceTrainingDataset,
    EvidenceTrainingExample,
    load_evidence_training_dataset,
)
from src.generation.evidence.models import (
    BinaryEvidenceScorer,
    EvidenceAssessment,
    EvidenceCandidate,
    EvidenceDecision,
    EvidenceDecisionScore,
    EvidenceSelection,
    EvidenceSelectionError,
)
from src.generation.evidence.selector import BinaryEvidenceSelector
from src.generation.evidence.scorer import HuggingFaceBinaryEvidenceScorer

__all__ = [
    "BinaryEvidenceScorer",
    "BinaryEvidenceSelector",
    "EvidenceAssessment",
    "EvidenceCandidate",
    "EvidenceDecision",
    "EvidenceDecisionScore",
    "EvidenceDatasetSplit",
    "EvidenceExampleKind",
    "EvidenceSelection",
    "EvidenceSelectionError",
    "EvidenceTrainingDataset",
    "EvidenceTrainingExample",
    "HuggingFaceBinaryEvidenceScorer",
    "load_evidence_training_dataset",
]
