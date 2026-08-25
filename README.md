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
chunk invariant audit. It also provides separate lexical tokenization for code
and documentation, explainable two-field BM25 retrieval, and reproducible
source-aware retrieval evaluation. It does not yet provide a complete RAG
pipeline.

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
- code-aware identifier expansion and conservative documentation tokenization;
- content and structural metadata terms stored as separate BM25 fields;
- an inverted BM25 index with stable ranking and inspectable field scores;
- mixed natural-language and code query tokenization;
- Moulinette-compatible source IoU, Recall@K, and MRR metrics;
- a fixed documentation/code mini-suite and BM25 experiment CLI;
- optional JSON reports with corpus, Git, environment, latency, and memory
  evidence;
- automated tests organized by pipeline component.

Current work:

- finalizing the BM25 lexical baseline for merge.

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

The supplied corpus, datasets, and generated outputs are not committed to Git.
The current local layout is:

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

The small deterministic BM25 acceptance suite is committed separately:

```text
evals/bm25/mini/
├── corpus/
│   ├── docs/
│   └── src/
└── suite.json
```

It validates experiment mechanics and expected parameter effects. It is not a
replacement for the complete evaluation datasets used to select final values.

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

## Lexical Tokenization

Lexical retrieval uses separate tokenizers for code and documentation. Both
preserve source order and repeated terms so a later BM25 index can measure term
frequency.

`CodeTokenizer` retains each complete normalized identifier and adds subword
signals for snake_case, dotted names, CamelCase, acronyms, and numeric suffixes:

```text
gpu_memory_utilization
→ gpu_memory_utilization, gpu, memory, utilization

PagedAttention.forward
→ pagedattention.forward, pagedattention, paged, attention, forward

HTTPServer2
→ httpserver2, http, server, 2
```

Capitalization is preserved until structural boundaries have been extracted.
Unicode components remain complete instead of being partially interpreted by
the ASCII CamelCase rule.

`TextTokenizer` lowercases Unicode text, discards surrounding punctuation,
splits hyphenated words, preserves technical underscore and dotted forms, and
normalizes typographic apostrophes. It does not apply stemming or remove stop
words before retrieval measurements justify those transformations.

Possessive expansion uses a bounded heuristic. The complete apostrophe form is
always retained. A suffix-free base is added only when it contains at least
four letters or the original base is uppercase:

```text
model's → model's, model
GPU's   → gpu's, gpu
it's    → it's
```

This threshold is not a grammatical rule. It removes common contraction noise
without adding a language parser, and it remains subject to retrieval
evaluation.

The current local corpus sanity check reports:

```text
Python: 1,753 files, 18,578 chunks, 2,919,369 tokens
        median 149, P95 322, maximum 607 tokens per chunk
        3 punctuation-only chunks with no lexical terms
Text:     199 files,  1,521 chunks,   141,492 tokens
        median 66, P95 256, maximum 393 tokens per chunk
        0 chunks with no lexical terms
```

Punctuation-only chunks are valid exact source slices but should be skipped by
the lexical index because they provide no searchable terms.

## BM25 Lexical Retrieval

Each searchable chunk keeps exact source evidence and two independent lexical
fields:

```text
content terms  = normalized terms from the exact chunk text
metadata terms = path, heading hierarchy, and overlapping Python symbols
```

Content preserves repeated terms for term-frequency scoring. Metadata terms
are stably deduplicated so repeated structural sources do not create an
undeclared weight. Python symbol spans come from the AST and use the same exact
character-offset conversion as Python chunking.

The index calculates separate document frequencies and average lengths for the
two fields. Metadata is combined only after both field scores are known:

```text
final_score = content_score + metadata_weight * metadata_score
```

This supports fractional weights exactly and keeps every retrieval result
explainable. Token repetition is not used as an approximation of metadata
weight. The default control parameters are `k1=1.5`, `b=0.75`, and
`metadata_weight=1.0`.

The inverted index stores postings from each term to matching document indexes
and term frequencies. Query execution visits matching candidates instead of
recounting every document. Zero-score documents are omitted, and equal scores
use `(file_path, start, end)` for deterministic ordering.

`QueryTokenizer` combines natural-language normalization with code identifier
expansion. `BM25Retriever` keeps this preparation outside the mathematical
index while accepting normal string queries.

## BM25 Evaluation

Run the neutral mini-suite control:

```bash
.venv/bin/python -m src.evaluation.bm25 --suite mini --run M0
```

Compare the planned metadata weights:

```bash
.venv/bin/python -m src.evaluation.bm25 \
  --suite mini \
  --compare M0 M1 M2 M3
```

Add `--verbose` to inspect every query, expected source, relevant rank, and
separate content and metadata scores. Save complete machine-readable evidence
only when needed:

```bash
.venv/bin/python -m src.evaluation.bm25 \
  --suite mini \
  --compare M0 M1 M2 M3 \
  --verbose \
  --output reports/bm25-mini.json
```

The report records the Git commit and dirty state, suite fingerprint,
environment, parameters, file and chunk counts, documentation and code metrics,
ranked hits, build time, recursive in-memory index size, traced peak build
memory, and median/P95 query latency.

Retrieval relevance requires an exact file path and source-range IoU of at
least `0.05`. Documentation and code metrics are reported separately. The
mini-suite demonstrates both useful structural boosts and metadata dominance;
it does not select a production weight. Hardware-dependent measurements are
repeated on one Linux machine before performance conclusions are recorded.

Controlled parameter history and provisional measurements are recorded in
`docs/bm25-tuning-log.md`.

## Verification

The current checks pass:

```text
pytest: 201 passed
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
- Preserve identifier capitalization until code structure is extracted.
- Bound possessive expansion and preserve the original apostrophe form.
- Score content and structural metadata as independent BM25 fields.
- Apply fractional metadata weight after field scoring.
- Use postings to avoid scanning every document for every query.
- Compare parameters on fixed, fingerprinted docs/code evaluation suites.

Reconsidered choices and their consequences are recorded in
`docs/decision-log.md`.

## Resources

- [Python pathlib documentation](https://docs.python.org/3/library/pathlib.html)
- [Python os.walk documentation](https://docs.python.org/3/library/os.html#os.walk)
- [Python dataclasses documentation](https://docs.python.org/3/library/dataclasses.html)
- [Python statistics documentation](https://docs.python.org/3/library/statistics.html)
- [Python regular expression documentation](https://docs.python.org/3/library/re.html)
- [Pydantic documentation](https://docs.pydantic.dev/)

### AI Usage

AI was used as a collaborative learning and review tool to discuss the subject,
identify implementation risks, propose tests, and improve documentation. Every
accepted change is reviewed, discussed, and tested before it is committed. The
author remains responsible for understanding, explaining, and maintaining all
submitted code.
