"""Run the public Python Fire command-line interface."""

from collections.abc import Sequence
import sys

import fire

from src.cli import (
    CliError,
    analyze_retrieval_errors,
    answer,
    answer_dataset,
    diagnose_answer_quality,
    evaluate,
    evaluate_all,
    index,
    index_semantic,
    search,
    search_dataset,
    search_dataset_hybrid,
    search_dataset_semantic,
    search_semantic,
    serve,
    validate_sources,
)
from src.evaluation.pdf.cli import (
    pdf_compare_runs,
    pdf_evaluate_answerability,
    pdf_evaluate_retrieval,
    pdf_migrate_judgments,
    pdf_run_answerability,
    pdf_run_retrieval,
    pdf_snapshot_records,
)
from src.evaluation.pdf.multibook_cli import (
    pdf_collect_judgments,
    pdf_freeze_multibook_golden,
    pdf_import_multibook_golden,
    pdf_prepare_judgment_pilot,
)


def main(argv: Sequence[str] | None = None) -> None:
    """Expose FREDL retrieval commands through Python Fire."""
    try:
        fire.Fire(
            {
                "analyze_retrieval_errors": analyze_retrieval_errors,
                "answer": answer,
                "answer_dataset": answer_dataset,
                "diagnose_answer_quality": diagnose_answer_quality,
                "evaluate": evaluate,
                "evaluate_all": evaluate_all,
                "index": index,
                "index_semantic": index_semantic,
                "pdf_collect_judgments": pdf_collect_judgments,
                "pdf_compare_runs": pdf_compare_runs,
                "pdf_evaluate_answerability": pdf_evaluate_answerability,
                "pdf_evaluate_retrieval": pdf_evaluate_retrieval,
                "pdf_freeze_multibook_golden": pdf_freeze_multibook_golden,
                "pdf_import_multibook_golden": pdf_import_multibook_golden,
                "pdf_migrate_judgments": pdf_migrate_judgments,
                "pdf_prepare_judgment_pilot": pdf_prepare_judgment_pilot,
                "pdf_run_answerability": pdf_run_answerability,
                "pdf_run_retrieval": pdf_run_retrieval,
                "pdf_snapshot_records": pdf_snapshot_records,
                "search": search,
                "search_dataset": search_dataset,
                "search_dataset_hybrid": search_dataset_hybrid,
                "search_dataset_semantic": search_dataset_semantic,
                "search_semantic": search_semantic,
                "serve": serve,
                "validate_sources": validate_sources,
            },
            command=argv,
        )
    except CliError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
