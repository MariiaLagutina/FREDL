"""Golden evaluation data for local PDF retrieval."""

from src.evaluation.pdf.answerability import (
    apply_pdf_comparison_guard,
    build_pdf_answerability_messages,
    build_pdf_comparison_messages,
    evaluate_pdf_answerability,
    parse_pdf_answerability_decision,
    parse_pdf_comparison_decision,
    PDF_ANSWERABILITY_PROMPT_VERSION,
    PdfAnswerabilityCaseScore,
    PdfAnswerabilityDecision,
    PdfAnswerabilityEvaluation,
    PdfAnswerabilityResult,
    PdfComparisonDecision,
    find_pdf_comparison_subjects,
)
from src.evaluation.pdf.bm25_baseline import (
    PdfBm25Baseline,
    PdfBm25Hit,
)
from src.evaluation.pdf.context import find_pdf_neighbor_context
from src.evaluation.pdf.evidence import (
    find_pdf_evidence_candidates,
    PdfEvidenceCandidates,
)
from src.evaluation.pdf.hybrid_baseline import (
    PDF_HYBRID_PARAMETERS,
    PdfHybridBaseline,
    PdfHybridHit,
)
from src.evaluation.pdf.loader import load_pdf_golden_dataset
from src.ingestion.pdf.fingerprints import fingerprint_pdf_records
from src.evaluation.pdf.judgments import (
    load_pdf_judgment_set,
    PdfJudgmentSet,
    PdfRecordJudgment,
    PdfRecordRelevance,
)
from src.evaluation.pdf.minilm_baseline import (
    PdfMiniLmBaseline,
    PdfMiniLmHit,
)
from src.evaluation.pdf.models import (
    PdfGoldenAnswerability,
    PdfGoldenCase,
    PdfGoldenDataset,
    PdfGoldenEvidence,
    PdfGoldenEvidenceRole,
    PdfGoldenModality,
)
from src.evaluation.pdf.retrieval_metrics import (
    evaluate_pdf_retrieval,
    PdfRetrievalCaseScore,
    PdfRetrievalEvaluation,
    PdfRetrievalResult,
)
from src.evaluation.pdf.review import (
    render_pdf_evidence_review,
    write_pdf_evidence_review,
)

__all__ = [
    "PDF_ANSWERABILITY_PROMPT_VERSION",
    "PdfAnswerabilityCaseScore",
    "PdfAnswerabilityDecision",
    "PdfAnswerabilityEvaluation",
    "PdfAnswerabilityResult",
    "PdfComparisonDecision",
    "PdfBm25Baseline",
    "PdfBm25Hit",
    "PdfGoldenAnswerability",
    "PdfGoldenCase",
    "PdfGoldenDataset",
    "PdfGoldenEvidence",
    "PdfGoldenEvidenceRole",
    "PdfGoldenModality",
    "PDF_HYBRID_PARAMETERS",
    "PdfHybridBaseline",
    "PdfHybridHit",
    "PdfJudgmentSet",
    "PdfMiniLmBaseline",
    "PdfMiniLmHit",
    "PdfRecordJudgment",
    "PdfRecordRelevance",
    "PdfRetrievalCaseScore",
    "PdfRetrievalEvaluation",
    "PdfRetrievalResult",
    "PdfEvidenceCandidates",
    "apply_pdf_comparison_guard",
    "build_pdf_answerability_messages",
    "build_pdf_comparison_messages",
    "evaluate_pdf_answerability",
    "evaluate_pdf_retrieval",
    "find_pdf_neighbor_context",
    "find_pdf_comparison_subjects",
    "find_pdf_evidence_candidates",
    "fingerprint_pdf_records",
    "load_pdf_golden_dataset",
    "load_pdf_judgment_set",
    "parse_pdf_answerability_decision",
    "parse_pdf_comparison_decision",
    "render_pdf_evidence_review",
    "write_pdf_evidence_review",
]
