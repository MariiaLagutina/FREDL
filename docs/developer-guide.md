# Developer Guide

This guide explains how to set up FREDL, run its checks, find the main entry
points and keep experiments reproducible. The current capabilities and their
limits are summarized in the [README](../README.md#current-project-state).

## Setup

Requirements: Python 3.10 or later and [`uv`](https://docs.astral.sh/uv/).
The repository pins every dependency in `uv.lock`.

```bash
make install            # uv sync: create .venv from the lock file
uv sync --locked        # the same, failing if uv.lock is out of date
```

The environment includes PyTorch and Transformers. Semantic retrieval runs on
CPU; no GPU is required for the test suite.

## Checks

```bash
make test               # pytest
make lint-strict        # flake8 and mypy --strict (the CI gate)
make lint               # flake8 and a less strict mypy configuration
make clean              # remove Python check caches
```

CI runs `uv sync --locked`, `make lint-strict` and `make test` on Python 3.10
for every pull request and push to `main`
([workflow](../.github/workflows/ci.yml)).

The tests are deterministic and use synthetic inputs only: generated PDFs,
synthetic records and fake model backends. They need no network access, no
model download and no private book. Passing tests verify contracts and
mechanics, not retrieval or answer quality.

Optional Git hooks are defined in `.pre-commit-config.yaml`. Install the
`pre-commit` tool separately, then run
`pre-commit install --hook-type pre-commit --hook-type pre-push`. The
private-file hook checks file names and paths only; it cannot detect book text
copied into an allowed file.

## Repository entry points

| Entry point | Purpose |
| --- | --- |
| `python -m src` ([`src/__main__.py`](../src/__main__.py), [`src/cli.py`](../src/cli.py)) | Fire CLI; `python -m src --help` lists all commands |
| `python -m src pdf_*` ([`src/evaluation/pdf/cli.py`](../src/evaluation/pdf/cli.py) and `src/evaluation/pdf/multibook/*/cli.py`) | PDF record snapshots, retrieval runs, pooling, judgments, metrics and answerability |
| `python -m src index`, `search`, `answer` | Code, Markdown and plain-text pipeline; it does not ingest PDFs |
| `python -m src serve` ([`src/api/app.py`](../src/api/app.py)) | Local FastAPI server on `127.0.0.1` for the code and text pipeline |

Source layout:

| Package | Contents |
| --- | --- |
| `src/ingestion/pdf/` | PDF pages, reading text, outline and heading matching, section intervals, chunking, index records, retrieval corpus policies, fingerprints |
| `src/ingestion/chunking/` | Python and text chunking for the code and text pipeline |
| `src/retrieval/` | BM25, semantic encoder and index, hybrid RRF, persisted index stores, tokenization |
| `src/generation/` | Context building, prompts, citation validation, Transformers backend, answer cache |
| `src/evaluation/pdf/` | Single-book PDF baselines and evaluation contracts; `multibook/` holds the library corpus, golden contract, runner, pooling and scoring |
| `src/evaluation/retrieval/`, `src/evaluation/answer_quality/` | Evaluation of the code and text pipeline |

Command usage and contracts for PDF work are described in
[PDF Evaluation and CLI](pdf-evaluation.md).

## Private data boundary

PDFs, extracted text, questions, judgments, model outputs and run artifacts
stay outside the repository in a user-controlled library directory, written
as `<library_root>` in the documentation. Never commit or publish them.
`data/`, `.local/`, `models/` and weight files are ignored by Git.

Model weights are loaded from the local Hugging Face cache. The multi-book
retrieval runner uses `local_files_only=True`; set `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` to make any other run fail instead of downloading.
Answerability and model-comparison experiments use a llama.cpp server on
loopback; it is not a FREDL dependency, and the answerability client rejects
non-loopback addresses.

## Reproducible experiments

- Run from a clean working tree. Run manifests record the code revision, a
  `code_dirty` flag and relevant package versions.
- Inputs are pinned by content: PDF SHA-256 values, record-set and corpus
  fingerprints, Golden Set hashes and retrieval configuration fingerprints.
  A mismatch is rejected rather than silently tolerated.
- Write each run to a new output path. Run, judgment and metrics outputs
  refuse to replace an existing file; not every helper report does, so do not
  rely on this as a backup.
- Use `python -m src pdf_compare_runs FIRST SECOND` to check whether two runs
  are directly comparable and why not.
- Label every result with its evidence class from
  [PD-009](product-decisions.md#pd-009-evaluate-without-mandatory-manual-annotation):
  behavioral diagnostic, synthetic known-provenance check, model judgment or
  human judgment.

Known reproducibility gaps:

- Some recorded experiments (M2-A, M2-B and R120 in the
  [research reports](pdf-retrieval-experiments.md)) used operator scripts
  outside the repository; they cannot be rerun from FREDL commands alone.
- Persisted PDF embedding indexes are not yet supported by FREDL commands:
  PDF retrieval runs re-encode the corpus each time.
- Python is not pinned to one minor version; CI uses 3.10, and local work has
  also used 3.13.

## Documentation map

| Question | Document |
| --- | --- |
| What works today? | [README](../README.md#current-project-state) |
| What does FREDL aim to be? | [Product Direction](product-direction.md) |
| Which decisions constrain the work? | [Product Decisions](product-decisions.md) |
| How is the code structured and what are its limits? | [Technical Foundation](technical-foundation.md) |
| How do I run PDF evaluation commands? | [PDF Evaluation and CLI](pdf-evaluation.md) |
| What has been measured, and how? | [PDF Retrieval Experiments](pdf-retrieval-experiments.md), [Inference Performance Diagnostics](inference-performance-diagnostics.md) |
| What is planned next? | [Retrieval and Inference Roadmap](retrieval-inference-roadmap.md) |

Research reports are dated records. Later decisions add dated notes instead
of rewriting them.

## Resuming development

The next technical work block is *Embedding Comparison Readiness*
([roadmap, Investigation 3](retrieval-inference-roadmap.md#3-embedding-model-alternatives)).
Start with the embedding profile and encoder changes in `src/retrieval/semantic/`
and the semantic configuration in `src/evaluation/pdf/manifest.py`, keeping
the current MiniLM behavior reproducible. Diagnostic tools follow as a
separate change.
