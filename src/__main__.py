"""Command-line entry point for the RAG application."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from src.retrieval import run_stored_retrieval


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line interface."""
    parser = argparse.ArgumentParser(prog="rag")
    commands = parser.add_subparsers(dest="command", required=True)

    search = commands.add_parser(
        "search",
        help="search a validated question dataset",
    )
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--fingerprint", required=True)
    search.add_argument("--input", type=Path, required=True)
    search.add_argument("--output", type=Path, required=True)
    search.add_argument("--k", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the selected RAG command."""
    arguments = build_parser().parse_args(argv)
    if arguments.command == "search":
        run_stored_retrieval(
            arguments.index,
            arguments.fingerprint,
            arguments.input,
            arguments.output,
            k=arguments.k,
        )


if __name__ == "__main__":
    main()
