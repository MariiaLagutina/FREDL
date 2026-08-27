"""Run the public Python Fire command-line interface."""

from collections.abc import Sequence

import fire

from src.cli import search, search_dataset


def main(argv: Sequence[str] | None = None) -> None:
    """Expose assignment-compatible RAG commands through Python Fire."""
    fire.Fire(
        {
            "search": search,
            "search_dataset": search_dataset,
        },
        command=argv,
    )


if __name__ == "__main__":
    main()
