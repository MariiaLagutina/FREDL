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
from src.generation.evidence.review import (
    EvidenceCandidateOrigin,
    EvidenceReviewCandidate,
    EvidenceReviewQueue,
    build_evidence_review_queue,
    save_evidence_review_queue,
)

__all__ = [
    "BinaryEvidenceScorer",
    "BinaryEvidenceSelector",
    "EvidenceAssessment",
    "EvidenceCandidate",
    "EvidenceCandidateOrigin",
    "EvidenceDecision",
    "EvidenceDecisionScore",
    "EvidenceDatasetSplit",
    "EvidenceExampleKind",
    "EvidenceReviewCandidate",
    "EvidenceReviewQueue",
    "EvidenceSelection",
    "EvidenceSelectionError",
    "EvidenceTrainingDataset",
    "EvidenceTrainingExample",
    "HuggingFaceBinaryEvidenceScorer",
    "build_evidence_review_queue",
    "load_evidence_training_dataset",
    "save_evidence_review_queue",
]
