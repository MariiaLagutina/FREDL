*This project has been created as part of the 42 curriculum by mlagutin.*

# RAG against the machine

## Description

This project is a local Retrieval-Augmented Generation (RAG) system for
answering questions about the vLLM codebase. Development is incremental: each
pipeline stage is implemented, tested, reviewed, and merged before work begins
on the next stage.

The current implementation provides validated data models and safe corpus file
discovery. It does not yet provide a complete RAG pipeline.

## Current Status

Implemented:

- project bootstrap with `uv`;
- pytest, flake8, and mypy checks;
- the required Pydantic data models;
- safe and deterministic corpus file discovery;
- project-relative POSIX source paths;
- filtering for supported file types, hidden entries, and symbolic links;
- automated tests for models and file discovery.

Current work:

- reviewing and finalizing the corpus file manifest.

Next step:

- preserve exact character offsets when source files are split into chunks.

## Requirements

- Python 3.10 or later;
- `uv` for dependency and environment management.

## Installation

Clone the repository and install the locked dependencies from its root:

```bash
make install
```

The `make install` target runs:

```bash
uv sync
```

## Local Data Layout

The supplied corpus, datasets, generated outputs, and local evaluation tools
are not committed to Git. The current local layout is:

```text
data/
├── raw/
│   └── vllm-0.10.1/
├── datasets/
│   ├── AnsweredQuestions/
│   └── UnansweredQuestions/
├── processed/
└── output/
    ├── search_results/
    └── search_results_and_answer/
```

Evaluation-facing source paths are relative to the project root:

```text
data/raw/vllm-0.10.1/docs/features/lora.md
```

## Development Commands

Install dependencies:

```bash
make install
```

Run the test suite:

```bash
make test
```

Run the mandatory lint and type checks:

```bash
make lint
```

Run stricter type checking:

```bash
make lint-strict
```

## Implemented File Discovery

File discovery currently supports:

- `.py` files as Python sources;
- `.md`, `.rst`, and `.txt` files as text sources;
- stable sorting by project-relative path;
- exclusion of unsupported files and hidden entries;
- exclusion of symbolic links;
- rejection of a corpus outside the project root;
- rejection of the project root itself as a corpus.

Example:

```python
from pathlib import Path

from src.ingestion import discover_files

project_root = Path.cwd()
corpus_root = project_root / "data" / "raw" / "vllm-0.10.1"
manifest = discover_files(project_root, corpus_root)

print(f"Discovered {len(manifest)} supported files")
```

The current vLLM corpus produces 1,952 manifest entries. All 147 unique source
files referenced by the supplied AnsweredQuestions datasets are included.

## Verification

The current checks pass:

```text
pytest: 12 passed
flake8: passed
mypy: passed
```

These results cover the current implementation only.

## Design Decisions

- Store portable project-relative paths instead of machine-specific absolute
  paths.
- Serialize paths with POSIX `/` separators on every operating system.
- Return a stable manifest order for reproducible processing.
- Reject symbolic links instead of reading files outside the intended corpus.
- Require the corpus to be a strict descendant of the project root.
- Use Pydantic for data exchanged between pipeline stages.

Reconsidered choices and their consequences are recorded in
`docs/decision-log.md`.

## Resources

- [Python pathlib documentation](https://docs.python.org/3/library/pathlib.html)
- [Python os.walk documentation](https://docs.python.org/3/library/os.html#os.walk)
- [Pydantic documentation](https://docs.pydantic.dev/)

### AI Usage

AI was used as a collaborative learning and review tool to discuss the subject,
identify implementation risks, propose tests, and improve documentation. Every
accepted change is reviewed, discussed, and tested before it is committed. The
author remains responsible for understanding, explaining, and maintaining all
submitted code.
