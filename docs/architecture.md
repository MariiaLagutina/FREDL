# RAG Pipeline Architecture

## Purpose

The project indexes the provided vLLM corpus, retrieves source spans for a
question, and uses a local Qwen model to generate an answer grounded in those
spans. The pipeline keeps evaluation-facing JSON models separate from internal
indexing and ranking data.

## Pipeline

```mermaid
flowchart LR
    A["Corpus files"] --> B["File discovery"]
    B --> C["Python or text chunker"]
    C --> D["Lexical index"]
    D --> E["Retriever"]
    E --> F["Context builder"]
    F --> G["Qwen backend"]
    G --> H["Validated JSON output"]
```

The same retriever serves both search commands and both answer commands. Batch
operations orchestrate repeated single-question operations rather than
implementing a second retrieval path.

## Component Boundaries

### CLI

The CLI parses Python Fire arguments, reports friendly errors, and calls the
pipeline services. It does not implement chunking, ranking, evaluation, or model
inference.

Planned module: `src/cli.py`.

### File discovery

File discovery walks the configured corpus root, applies an explicit file
allowlist, skips unsafe or irrelevant entries, and returns paths in a stable
order. Stable ordering makes indexing deterministic.

Planned module: `src/ingestion/files.py`.

### Chunking

Chunking has two implementations behind one interface:

- Python-aware chunking for `.py` files.
- Markdown/text chunking for documentation files.

Every internal chunk carries its text, exact project-relative file path, and
character coordinates in the original file. The public `MinimalSource` model is
created from those coordinates only when producing search output.

Planned modules:

- `src/chunking/base.py`
- `src/chunking/python.py`
- `src/chunking/text.py`

### Indexing and storage

The indexer receives chunks and builds the mandatory lexical index. The index
store persists everything required for retrieval under `data/processed/` and
loads it without reading the full corpus again.

Planned modules:

- `src/indexing/lexical.py`
- `src/indexing/store.py`

The exact lexical implementation and serialized representation are deferred to
their dedicated plan steps.

### Retrieval

The retriever accepts one query and `k`, ranks indexed chunks, and returns
internal ranked results. The output adapter removes internal text and scores and
creates `MinimalSearchResults` containing only the required source locations.

Batch search loads a `RagDataset`, calls the same single-query retriever for each
question, and writes one `StudentSearchResults` file.

Planned module: `src/retrieval/service.py`.

### Context and generation

The context builder selects retrieved chunk text within a configurable token
budget and labels every source. The generation backend owns Qwen loading and
inference. Prompt construction remains independent from the model backend so it
can be tested without loading model weights.

Planned modules:

- `src/generation/context.py`
- `src/generation/prompt.py`
- `src/generation/qwen.py`

### Evaluation

The local evaluator compares retrieved and reference spans and reports recall at
`k`. It is an independent implementation for development only and must never
import or execute Moulinette.

Planned module: `src/evaluation/recall.py`.

### Orchestration

The pipeline wires the components together and implements the six required
operations: `index`, `search`, `search_dataset`, `answer`, `answer_dataset`, and
`evaluate`.

Planned module: `src/pipeline.py`.

## Data Boundaries

Evaluation-facing models live in `src/models.py` and retain the exact field names
required by the assignment. Internal models may add chunk text, ranking scores,
token counts, or index metadata, but those fields must not leak into required
JSON output unless explicitly intended.

All source paths are normalized once during ingestion. They remain POSIX-style,
relative to the project root, and are reused unchanged through indexing,
retrieval, and serialization.

Character coordinates refer to the original decoded file content. Chunking must
never calculate offsets from normalized, stripped, or reformatted text.

## Dependency Direction

Dependencies point toward data contracts and lower-level services:

```text
CLI -> pipeline -> retrieval/generation/evaluation
                    |
                    v
              indexing/chunking
                    |
                    v
                  models
```

Lower-level modules do not import the CLI or pipeline. This keeps chunking,
ranking, and prompt construction independently testable.

## Error Handling

Library code raises specific application exceptions. The CLI catches those
exceptions and prints concise user-facing messages without an unhandled
traceback. Missing files, malformed JSON, empty queries, invalid `k`, and an
unavailable model backend are handled at their nearest input boundary.

## Test Strategy

- Unit tests cover models, path normalization, chunk boundaries, ranking, span
  overlap, prompt construction, and degenerate inputs.
- Integration tests cover index-save-load and each pipeline operation on a small
  fixture corpus.
- End-to-end tests exercise the required CLI commands without importing
  Moulinette.
- Local full-corpus runs measure recall and performance separately from the fast
  committed test suite.

## Deferred Decisions

The following choices belong to later plan steps and are intentionally not fixed
here:

- the exact file allowlist;
- Python and documentation chunk boundary algorithms;
- BM25 or TF-IDF library and parameters;
- index serialization format;
- context token budget and prompt wording;
- Qwen runtime configuration.
