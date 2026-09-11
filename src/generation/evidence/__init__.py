"""Source-by-source binary evidence selection."""

from src.generation.evidence.annotations import (
    EvidenceReviewAnnotation,
    EvidenceReviewAnnotations,
    build_evidence_review_annotations,
    load_evidence_review_annotations,
    promote_reviewed_evidence,
    save_evidence_review_annotations,
)
from src.generation.evidence.dataset import (
    EvidenceDatasetSplit,
    EvidenceExampleKind,
    EvidenceTrainingDataset,
    EvidenceTrainingExample,
    load_evidence_training_dataset,
    save_evidence_training_dataset,
    validate_evidence_training_dataset,
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
    "EvidenceReviewAnnotation",
    "EvidenceReviewAnnotations",
    "EvidenceReviewQueue",
    "EvidenceSelection",
    "EvidenceSelectionError",
    "EvidenceTrainingDataset",
    "EvidenceTrainingExample",
    "HuggingFaceBinaryEvidenceScorer",
    "build_evidence_review_queue",
    "build_evidence_review_annotations",
    "load_evidence_training_dataset",
    "load_evidence_review_annotations",
    "promote_reviewed_evidence",
    "save_evidence_review_annotations",
    "save_evidence_review_queue",
    "save_evidence_training_dataset",
    "validate_evidence_training_dataset",
]
