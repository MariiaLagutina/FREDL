"""Golden evaluation data for local PDF retrieval."""

from src.evaluation.pdf.evidence import (
    find_pdf_evidence_candidates,
    PdfEvidenceCandidates,
)
from src.evaluation.pdf.loader import load_pdf_golden_dataset
from src.evaluation.pdf.models import (
    PdfGoldenAnswerability,
    PdfGoldenCase,
    PdfGoldenDataset,
    PdfGoldenEvidence,
    PdfGoldenEvidenceRole,
    PdfGoldenModality,
)

__all__ = [
    "PdfGoldenAnswerability",
    "PdfGoldenCase",
    "PdfGoldenDataset",
    "PdfGoldenEvidence",
    "PdfGoldenEvidenceRole",
    "PdfGoldenModality",
    "PdfEvidenceCandidates",
    "find_pdf_evidence_candidates",
    "load_pdf_golden_dataset",
]
