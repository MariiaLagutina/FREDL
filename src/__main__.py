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
    search_semantic,
    validate_sources,
)


def main(argv: Sequence[str] | None = None) -> None:
    """Expose assignment-compatible RAG commands through Python Fire."""
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
                "search": search,
                "search_dataset": search_dataset,
                "search_semantic": search_semantic,
                "validate_sources": validate_sources,
            },
            command=argv,
        )
    except CliError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
