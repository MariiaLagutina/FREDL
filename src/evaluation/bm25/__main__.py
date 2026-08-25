"""Run controlled BM25 experiments from a terminal."""

import argparse
from pathlib import Path

from src.evaluation.bm25.reporter import print_results
from src.evaluation.bm25.runner import run_experiment
from src.evaluation.bm25.suite import build_suite_documents, load_suite
from src.retrieval.bm25 import BM25Parameters


RUNS = {
    "M0": BM25Parameters(k1=1.5, b=0.75, metadata_weight=1.0),
    "M1": BM25Parameters(k1=1.5, b=0.75, metadata_weight=1.5),
    "M2": BM25Parameters(k1=1.5, b=0.75, metadata_weight=2.0),
    "M3": BM25Parameters(k1=1.5, b=0.75, metadata_weight=3.0),
}


def main() -> None:
    """Load one suite, execute requested runs, and print their evidence."""
    arguments = _parse_arguments()
    project_root = Path(__file__).resolve().parents[3]
    suite_root = project_root / "evals" / "bm25" / arguments.suite
    suite = load_suite(suite_root)
    documents = build_suite_documents(project_root, suite_root, suite)
    run_ids = arguments.compare or [arguments.run or "M0"]
    results = [
        run_experiment(
            suite,
            documents,
            run_id,
            RUNS[run_id],
        )
        for run_id in run_ids
    ]
    print_results(results, verbose=arguments.verbose)


def _parse_arguments() -> argparse.Namespace:
    """Parse a single run or a controlled multi-run comparison."""
    parser = argparse.ArgumentParser(
        description="Run fixed BM25 retrieval experiments",
    )
    parser.add_argument("--suite", default="mini", choices=("mini",))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--run", choices=tuple(RUNS))
    selection.add_argument("--compare", nargs="+", choices=tuple(RUNS))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
