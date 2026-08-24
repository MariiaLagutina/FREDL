*This project has been created as part of the 42 curriculum by mlagutin.*

# RAG against the machine

## Description

This project is a local Retrieval-Augmented Generation (RAG) system for
answering questions about the vLLM codebase. Development is incremental: each
pipeline stage is implemented, tested, reviewed, and merged before work begins
on the next stage.

The current implementation provides validated exchange models, safe corpus
file discovery, exact source reading, immutable chunks with character offsets,
structural chunking for Python, Markdown, and plain text, and a full-corpus
chunk invariant audit. It does not yet provide a complete RAG pipeline.

## Current Status

Implemented:

- project bootstrap with `uv`;
- pytest, flake8, and mypy checks;
- the required Pydantic data models;
- safe and deterministic corpus file discovery;
- project-relative POSIX source paths;
- filtering for supported, readable, non-binary files within a configurable
  size limit;
- exact UTF-8 source reading without newline normalization;
- immutable source documents and half-open chunk spans;
- Python-aware AST chunking with safe fallbacks;
- Markdown and plain-text chunking with section metadata and bounded overlap;
- a shared orchestrator for selecting format-specific chunkers;
- deterministic chunk audits with invariant failures and size statistics;
- automated tests organized by pipeline component.

Current work:

- preparing the chunk audit stage for review and merge.

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

## Ingestion Architecture

The public ingestion API is exported from `src.ingestion`, while internal
format-specific implementations are separated by responsibility:

```text
src/ingestion/
├── audit/
│   ├── invariants.py
│   ├── models.py
│   ├── runner.py
│   └── statistics.py
├── chunking/
│   ├── orchestrator.py
│   ├── python/
│   └── text/
├── documents.py
└── files.py
```

The orchestrator selects a chunker from `SourceDocument.kind`. Chunking
implementations do not know how the corpus was discovered, and the audit does
not contain format-specific branches.

## Implemented File Discovery

File discovery currently supports:

- `.py` files as Python sources;
- `.md`, `.rst`, and `.txt` files as text sources;
- stable sorting by project-relative path;
- exclusion of unsupported files and hidden entries;
- exclusion of symbolic links;
- exclusion of files above a configurable size limit;
- exclusion of binary-looking or unreadable files;
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

The default maximum source-file size is 10 MiB. It can be changed explicitly:

```python
manifest = discover_files(
    project_root,
    corpus_root,
    max_file_size_bytes=5 * 1024 * 1024,
)
```

The current vLLM corpus produces 1,952 manifest entries. All 147 unique source
files referenced by the supplied AnsweredQuestions datasets are included.

## Exact Source Offsets

Discovered files are read as strict UTF-8 without newline normalization. This
keeps character positions stable across ingestion and retrieval. The reader
also verifies that every source path is the canonical, project-relative POSIX
path expected by the evaluator.

`SourceDocument` and `Chunk` are frozen, slotted dataclasses. A chunk uses the
standard Python half-open interval `[start:end)` and must satisfy:

```text
0 <= start < end
len(chunk.text) == end - start
chunk.text == document.text[start:end]
```

Example:

```python
from pathlib import Path

from src.ingestion import discover_files, make_chunk, read_document

project_root = Path.cwd()
corpus_root = project_root / "data" / "raw" / "vllm-0.10.1"
corpus_file = discover_files(project_root, corpus_root)[0]
document = read_document(project_root, corpus_file)
chunk = make_chunk(document, start=0, end=min(2000, len(document.text)))

assert chunk.text == document.text[chunk.start:chunk.end]
```

Overlapping chunks are supported because each chunk independently stores an
exact source range. Chunking strategies decide the boundaries and overlap;
the source-offset layer only guarantees their correctness.

## Python-aware Chunking

Python sources are parsed with the standard-library AST and divided at exact
structural boundaries. Top-level functions and classes retain their
decorators, signatures, docstrings, and directly preceding comment blocks.
Oversized classes are divided around direct methods; oversized functions are
divided around direct body statements. Adjacent small units are packed while
the configured maximum size is respected.

If parsing fails because a source file is incomplete or syntactically invalid,
the chunker falls back to exact line boundaries and then to character limits.
The fallback preserves Unicode text and original LF, CRLF, or CR newlines.

```python
from src.ingestion import chunk_python_document

chunks = chunk_python_document(document, max_chunk_size=2000)

assert all(len(chunk.text) <= 2000 for chunk in chunks)
assert all(
    chunk.text == document.text[chunk.start:chunk.end]
    for chunk in chunks
)
```

Chunk text never contains synthetic class or function context. Structural
names can later be stored as retrieval metadata without invalidating exact
source coordinates.

## Markdown and Plain-Text Chunking

Markdown files are partitioned into headings, paragraphs, lists, fenced code
blocks, and whitespace ranges. Heading-like text inside backtick or tilde code
fences remains code. Each resulting chunk stores its active heading hierarchy
as an immutable `section_path` without adding synthetic text to the exact
source slice.

Plain `.txt` and `.rst` files use paragraph boundaries and do not interpret
hash-prefixed lines as Markdown headings. Oversized text blocks prefer line
boundaries and fall back to character limits when necessary.

```python
from src.ingestion import chunk_text_document

chunks = chunk_text_document(
    document,
    max_chunk_size=2000,
)

assert all(len(chunk.text) <= 2000 for chunk in chunks)
assert all(
    chunk.text == document.text[chunk.start:chunk.end]
    for chunk in chunks
)
```

Overlap is limited to forced splits inside one oversized block. Its default is
the smaller of 100 characters and ten percent of `max_chunk_size`; callers can
disable it with `overlap_size=0`. Natural structural boundaries do not create
duplicate chunks. Original Markdown markup is preserved until retrieval
evaluation provides evidence that a separate normalization layer improves
Recall@5.

## Chunk Audit

`audit_documents()` checks an in-memory document stream. `audit_corpus()` runs
the complete discovery, reading, chunking, and audit flow without retaining
every source document in memory.

Each document is chunked twice to verify deterministic output. Every chunk is
then checked for:

- a valid half-open source range;
- exact equality with its source slice;
- a size at or below `max_chunk_size`;
- at least one non-whitespace character.

The resulting immutable `ChunkAuditReport` contains actionable issues and a
size distribution with the minimum, median, nearest-rank P95, and maximum.

```python
from pathlib import Path

from src.ingestion import audit_corpus

project_root = Path.cwd()
corpus_root = project_root / "data" / "raw" / "vllm-0.10.1"
report = audit_corpus(project_root, corpus_root)

assert report.passed
assert report.size_summary is not None
assert report.size_summary.maximum <= 2000
```

The current local vLLM corpus audit reports:

```text
documents: 1,952
chunks: 20,099
invalid chunks: 0
minimum size: 5
median size: 931
P95 size: 1,978
maximum size: 2,000
```

## Verification

The current checks pass:

```text
pytest: 131 passed
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
- Read complete sources as strict UTF-8 and preserve original newline
  characters.
- Use Pydantic for assignment-facing JSON and frozen, slotted dataclasses for
  high-volume internal ingestion records.
- Represent chunk coordinates as half-open Python ranges.
- Keep Python chunk text exact and reserve synthetic retrieval context for
  metadata.
- Store Markdown heading paths as metadata instead of synthetic chunk text.
- Preserve Markdown markup until retrieval evaluation justifies normalization.
- Apply overlap only to forced splits inside oversized text blocks.
- Keep format-specific chunkers behind one shared orchestrator.
- Audit format-independent invariants through the public chunking path.

Reconsidered choices and their consequences are recorded in
`docs/decision-log.md`.

## Resources

- [Python pathlib documentation](https://docs.python.org/3/library/pathlib.html)
- [Python os.walk documentation](https://docs.python.org/3/library/os.html#os.walk)
- [Python dataclasses documentation](https://docs.python.org/3/library/dataclasses.html)
- [Python statistics documentation](https://docs.python.org/3/library/statistics.html)
- [Pydantic documentation](https://docs.pydantic.dev/)

### AI Usage

AI was used as a collaborative learning and review tool to discuss the subject,
identify implementation risks, propose tests, and improve documentation. Every
accepted change is reviewed, discussed, and tested before it is committed. The
author remains responsible for understanding, explaining, and maintaining all
submitted code.
