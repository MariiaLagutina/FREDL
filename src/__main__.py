"""Run the public Python Fire command-line interface."""

from collections.abc import Sequence
import sys

import fire

from src.cli import CliError, search, search_dataset, validate_sources


def main(argv: Sequence[str] | None = None) -> None:
    """Expose assignment-compatible RAG commands through Python Fire."""
    try:
        fire.Fire(
            {
                "search": search,
                "search_dataset": search_dataset,
                "validate_sources": validate_sources,
            },
            command=argv,
        )
    except CliError as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
