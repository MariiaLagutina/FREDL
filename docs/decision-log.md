# Engineering Decision Log

This log records implementation choices that were reconsidered during the
project. Every entry uses the same structure: status, initial approach, reason
for reconsideration, replacement decision, consequences, and lesson.

## 2026-08-23 - Require the corpus to be below the project root

**Status:** Accepted

### Initial approach

File discovery required the corpus path to be inside the project root. The
check also allowed the corpus root and project root to be the same directory.

### Why the approach was reconsidered

Passing the project root as the corpus root would scan the student project's
own source code and tests together with the intended vLLM corpus. The operation
was read-only, but it could contaminate the index, reduce retrieval quality,
increase indexing time, and expose unrelated local text to later pipeline
stages.

### Decision

File discovery now requires the corpus root to be a strict descendant of the
project root. Equal paths are rejected before directory traversal begins.

### Consequences

- Accidental whole-project indexing fails early with a clear error.
- Test fixture corpora and the required `data/raw/vllm-0.10.1` layout remain
  supported.
- Callers must distinguish explicitly between the project root and corpus root.

### Lesson

Containment checks should distinguish between "inside or equal" and "strictly
inside." When a broad path can increase scope silently, validate the narrowest
safe relationship before performing any traversal or data processing.

## 2026-08-23 - Validate incomplete UTF-8 samples incrementally

**Status:** Accepted

### Initial approach

Binary-file detection read a fixed-size prefix and decoded that sample as a
complete UTF-8 string.

### Why the approach was reconsidered

The sample boundary could split a valid multi-byte character. A normal text
file would then raise `UnicodeDecodeError`, be classified as binary, and be
silently omitted from the retrieval corpus. Missing source files can reduce
recall without producing an obvious indexing failure.

### Decision

Decode the sample with an incremental UTF-8 decoder and mark the input as
non-final. Invalid byte sequences still identify non-text content, while an
incomplete character at the end of the sample remains valid pending more data.

### Consequences

- Valid UTF-8 text is not rejected because of an arbitrary sample boundary.
- NUL bytes and genuinely invalid UTF-8 sequences are still filtered out.
- Binary detection reads at most a small prefix instead of the complete file.
- A regression test fixes a multi-byte character across the exact boundary.

### Lesson

When validating a prefix of streamed data, do not apply checks that assume the
prefix is a complete input. Buffer boundaries are implementation details and
must not change how valid content is classified.

## 2026-08-23 - Use slotted dataclasses for internal ingestion records

**Status:** Accepted

### Initial approach

`SourceDocument` and `Chunk` were implemented as frozen Pydantic models, like
the external JSON models required by the assignment.

### Why the approach was reconsidered

Pydantic validation is valuable when data crosses an untrusted input or output
boundary. Internal ingestion records are constructed by controlled project
functions, however, and `Chunk` may be instantiated tens of thousands of
times. Repeating general-purpose model parsing and storing per-instance model
state would add avoidable indexing time and memory use.

### Decision

Keep Pydantic for the assignment-facing JSON models and use frozen, slotted
dataclasses for internal `SourceDocument` and `Chunk` records. Enforce span
invariants in `Chunk.__post_init__` and continue using static type checking for
the controlled construction paths.

### Consequences

- Internal records remain immutable.
- `slots=True` removes the per-instance `__dict__` overhead.
- Chunk construction retains explicit coordinate and text-length validation.
- JSON boundary models keep Pydantic parsing and serialization.
- Tests distinguish internal dataclass errors from external validation errors.

### Lesson

Choose validation machinery according to the trust boundary and object volume.
A single modeling tool across every layer can look consistent while imposing
costs that the internal data flow does not need.

## 2026-08-23 - Keep structural context out of exact chunk text

**Status:** Accepted

### Initial approach

Repeat a containing class or function header inside every structural chunk so
that each method or statement carries readable context during retrieval.

### Why the approach was reconsidered

Prepended text would no longer be the exact source slice identified by the
chunk's `start` and `end` coordinates. Repeating a large prefix through overlap
would preserve exactness, but could exceed the size limit and create excessive
duplicate content.

### Decision

Keep `Chunk.text` equal to `source_text[start:end]`. Split oversized classes and
functions on structural AST boundaries, and add class or function names later
as separate retrieval metadata rather than synthetic source text.

### Consequences

- Every chunk remains traceable to one exact source range.
- Structural splitting does not inflate chunks with repeated prefixes.
- Retrieval metadata will need a separate representation from chunk text.
- Context enrichment cannot silently change evaluator-facing coordinates.

### Lesson

Search context and source evidence serve different purposes. Preserve source
evidence exactly and enrich retrieval through explicit metadata.

## 2026-08-23 - Fail on invalid complete UTF-8 source text

**Status:** Accepted

### Initial approach

Invalid bytes encountered while reading a complete source file could be
ignored or replaced to let indexing continue.

### Why the approach was reconsidered

Both `errors="ignore"` and `errors="replace"` silently change the decoded
source. The resulting text can differ from the corpus used by the evaluator,
making later character offsets unreliable while hiding the original data
problem.

### Decision

Read complete source files with strict UTF-8 decoding and preserve their
original newline characters. If a file passes the lightweight discovery sample
but contains invalid UTF-8 later, stop with `UnicodeDecodeError` instead of
building an index from modified text.

### Consequences

- Source text and character offsets share one unmodified representation.
- Invalid corpus content fails visibly at the ingestion boundary.
- Callers can report the failing path instead of receiving a corrupted chunk.
- The corpus is required to contain valid UTF-8 source files.

### Lesson

Recovery that mutates source data is unsafe when downstream identifiers depend
on exact positions. Prefer a visible ingestion failure to silently producing
coordinates for content that no longer matches the source of truth.

## 2026-08-24 - Defer Markdown normalization until retrieval evaluation

**Status:** Accepted

### Initial approach

Strip emphasis markers, link syntax, heading markers, and code-fence delimiters
before sending Markdown chunks to the embedding model.

### Why the approach was reconsidered

Correct inline Markdown normalization requires contextual parsing. Simple
regular expressions can corrupt identifiers, escaped markers, arithmetic,
inline code, and fenced source code. Adding that transformation before any
retrieval measurement would increase complexity without evidence that the
original markup reduces search quality.

### Decision

Keep exact Markdown source text in the first retrieval representation and use
parsed headings and block types only for chunk boundaries and metadata. Add a
separate parser-based normalization layer only if controlled experiments show
an improvement in Recall@5.

### Consequences

- Chunking remains responsible for structure rather than text rewriting.
- Markdown code and emphasis remain available to models trained on technical
  text.
- Exact source coordinates remain directly auditable.
- A future normalizer must be evaluated against the unchanged baseline.

### Lesson

Do not add irreversible preprocessing because it appears cleaner. Preserve a
simple baseline and require retrieval metrics to justify normalization.

## 2026-08-24 - Limit overlap to oversized text block fallback

**Status:** Accepted

### Initial approach

Apply a fixed overlap between every pair of Markdown or plain-text chunks to
retain context around all boundaries.

### Why the approach was reconsidered

Structural blocks already preserve headings, paragraphs, lists, and fenced
code boundaries. Repeating text across those natural boundaries would increase
index size, duplicate complete semantic units, and risk mixing content from
different heading paths without repairing a real context loss.

### Decision

Use overlap only when one structural block is itself larger than the maximum
chunk size and must use line or character fallback. The default overlap is the
smaller of 100 characters and ten percent of the configured chunk size.
Callers may set it to zero. Prefer a complete trailing line only when doing so
does not exceed twice the requested overlap; otherwise use the exact character
target.

### Consequences

- Natural structural chunks remain disjoint and compact.
- Long paragraphs and code blocks retain limited boundary context.
- Whole-line overlap avoids unnecessary mid-line starts when duplication stays
  bounded.
- Every overlapping chunk remains an exact source slice with valid offsets.

### Lesson

Overlap should repair a forced split, not compensate for boundaries that are
already semantically meaningful. Bound duplication explicitly and measure its
retrieval value later.

## 2026-08-24 - Separate chunkers behind a shared orchestrator

**Status:** Accepted

### Initial approach

Python, Markdown, and plain-text chunking modules all lived directly inside
`src/ingestion`. The public dispatcher was another file in the same flat
directory.

### Why the approach was reconsidered

As parsing, fallback, overlap, and audit modules accumulated, unrelated
implementation details became visually indistinguishable. Adding another
source format would increase that flat module list and make internal imports
harder to navigate. The audit runner also needed one stable entry point rather
than knowledge of every format-specific function.

### Decision

Group chunking code below `src/ingestion/chunking`, place Python and text
implementations in separate subpackages, and keep format selection in
`orchestrator.py`. Store audit models, invariant checks, execution, and
statistics in a separate `src/ingestion/audit` package. Preserve the existing
public exports from `src.ingestion`.

### Consequences

- Public imports remain stable while internal ownership becomes explicit.
- The audit calls one orchestrator instead of branching on file formats.
- Format-specific tests can mirror the source package structure.
- A new source format can be introduced as a sibling chunking strategy.
- Moving modules requires internal import updates but no behavior changes.

### Lesson

A flat package is useful while responsibilities are still small. Split it when
new formats and cross-cutting services create distinct reasons for modules to
change, while keeping a narrow public API stable across the refactor.

## 2026-08-24 - Preserve capitalization until identifier expansion

**Status:** Accepted

### Initial approach

The shared lexical scanner converted every matched unit to lowercase as soon
as it was extracted. This produced stable case-insensitive tokens for ordinary
text and exact identifier forms.

### Why the approach was reconsidered

Capitalization is structural data inside code identifiers. Converting
`SamplingParams` to `samplingparams` before code-specific processing removes
the boundary between `Sampling` and `Params`. That boundary cannot be inferred
reliably from the normalized string, so later subword expansion would either
miss useful search terms or require unsafe dictionary-based guesses.

### Decision

Separate lexical extraction from normalization. `scan_lexemes()` preserves
the original capitalization and delimiters, while `scan_tokens()` remains the
lowercase wrapper for callers that only need normalized units. Code identifier
expansion operates on the preserved lexeme and lowercases exact and component
signals only after snake_case, dotted-name, CamelCase, acronym, and numeric
boundaries have been identified.

### Consequences

- Existing normalized scanning behavior remains stable.
- CamelCase and acronym boundaries are available to the code tokenizer.
- Exact normalized identifiers and readable subwords can coexist.
- The shared scanner exposes one additional intermediate representation.
- Golden tests must protect both preserved lexemes and normalized tokens.

### Lesson

Apply irreversible normalization only after every downstream consumer has
extracted the structure it needs. Case may be irrelevant for matching while
still carrying essential information during parsing.

## 2026-08-24 - Bound possessive expansion with a simple heuristic

**Status:** Accepted

### Initial approach

Preserve every apostrophe form as one normalized token. This avoids fragments
such as `isn` and `t`, but a possessive such as `model's` then cannot match a
query containing only `model`.

### Why the approach was reconsidered

Removing `'s` from every token would treat contractions as possessives.
Forms such as `it's`, `he's`, and `she's` would create speculative base terms
that do not represent the same grammatical operation. A full language parser
would add dependencies and complexity before retrieval evaluation shows that
the distinction materially affects search quality.

### Decision

Always retain the complete normalized apostrophe form. Add the suffix-free
base as a second search signal only when the base contains at least four
letters or when the original base is uppercase, which covers technical
acronyms such as `GPU's` and `API's`. Count alphabetic characters instead of
raw string length so punctuation and digits do not satisfy the threshold.

### Consequences

- `model's` contributes both `model's` and `model`.
- Short contractions such as `it's` and `she's` remain single tokens.
- Uppercase technical acronyms remain searchable without their suffix.
- The rule removes common noise without requiring an NLP library.
- The threshold is a heuristic, not a grammatical law; forms such as `that's`
  may still contribute a useful but linguistically simplified base token.
- Retrieval metrics may justify changing or removing the heuristic later.

### Lesson

When a lightweight heuristic replaces full linguistic analysis, keep it
bounded, preserve the original signal, document known errors, and make the
behavior executable through exact tests.

## 2026-08-25 - Score content and metadata as separate BM25 fields

**Status:** Accepted

### Initial approach

Append path, heading, and symbol tokens to chunk content. Approximate metadata
weight by repeating those tokens before building one BM25 field.

### Why the approach was reconsidered

Token repetition cannot represent a fractional weight such as `1.5` exactly.
It also changes term frequency before `k1` saturation, field length before `b`
normalization, and document frequency inside the content corpus. A score change
could not then be attributed cleanly to body evidence or structural context.

### Decision

Store immutable `content_terms` and stably deduplicated `metadata_terms` as two
independent fields. Calculate separate document frequencies, average lengths,
and BM25 scores. Combine them only after field scoring:

```text
final_score = content_score + metadata_weight * metadata_score
```

Expose content, metadata, and final scores on every hit. Use inverted postings
for each field so query execution visits matching candidates rather than
recounting all document terms.

### Consequences

- Fractional metadata weights are represented exactly.
- Structural boosts cannot alter content term frequency or saturation.
- Field scores explain why a result moved between experiment runs.
- Metadata and content retain independent length normalization statistics.
- Two postings maps and two sets of corpus statistics increase index state.
- A high metadata weight can still promote an attractive but irrelevant path,
  so fixed docs/code evaluation remains required.

### Lesson

When two evidence sources need different weights, preserve them as explicit
signals until the final combination step. Do not encode importance through
data duplication when the scoring model can represent it directly.

## 2026-08-26 - Persist validated lexical inputs instead of runtime internals

**Status:** Accepted

### Initial approach

Serialize the complete live `BM25Index` object, including its private postings
maps and derived statistics, with Python pickle. Loading would restore the
runtime object graph directly with minimal reconstruction work.

### Why the approach was reconsidered

Pickle couples stored data to Python class layout and executes a Python object
deserialization protocol that is inappropriate for an index file that may be
stale or damaged. Persisting private derived structures would also duplicate
the source of truth and make schema compatibility difficult to inspect. A
class refactor could make an old cache fail without a clear reindex message.

### Decision

Persist a versioned, Pydantic-validated JSON snapshot containing the corpus
fingerprint, all BM25 parameters, exact chunks and section paths, and the
already-tokenized content and metadata fields. Reconstruct postings and corpus
statistics from those stored lexical fields at load time; do not reopen,
reparse, rechunk, or retokenize the corpus. Reject unknown schema versions and
corpus fingerprint mismatches with an explicit `reindex required` error.

Calculate the corpus fingerprint from every discovered file's canonical
project-relative path and exact bytes in sorted path order. Save through a
temporary sibling file and atomically replace the target only after the JSON
snapshot has been written successfully.

### Consequences

- Stored data is portable, inspectable, and validated before runtime objects
  are created.
- Source text, offsets, section metadata, lexical fields, and scoring
  parameters round-trip without loss.
- Loading rebuilds derived postings but avoids the much more expensive corpus
  parsing and tokenization stages.
- The JSON representation favors transparency over compactness. The full
  20,096-document snapshot measured 90,700,944 bytes and loaded in 1.874
  seconds on the Linux development machine.
- Schema changes require an intentional version update and reindex behavior.

**Verification evidence:**

On the 1,952-file vLLM corpus with fingerprint
`1745355afa9ef90effcc67802ad356e439dbf8a5d954e36ad4095d88b05bf1f2`,
the existing pipeline produced 20,096 lexical documents in 12.994 seconds.
Saving took 0.878 seconds. Top-10 source rankings for fixed documentation and
code queries were identical before and after loading. A subprocess test also
proves that a separate Python process reproduces exact top-k scores.

### Lesson

Persist the smallest validated source of truth that can rebuild runtime state.
Treat cache compatibility as an explicit contract rather than relying on a
serializer to preserve private implementation details.

## 2026-08-27 - Use domain names for retrieval output models

**Status:** Accepted

### Initial approach

Name the output models `MinimalSearchResults`, `StudentSearchResults`,
`MinimalAnswer`, and `StudentSearchResultsAndAnswer`, following the terminology
of the original assignment schema.

### Why the approach was reconsidered

The `Student` prefix describes who produced the data rather than what the data
represents. It makes reusable retrieval components appear tied to coursework
and gives portfolio readers less information about whether a model represents
one query or a complete retrieval run. `Minimal` is similarly ambiguous at the
Python API boundary.

### Decision

Use `QuerySearchResult` for one query and `RetrievalResults` for a complete
retrieval run. Use `QueryAnswer` and `RetrievalResultsWithAnswers` for the
corresponding answer-bearing models. Preserve the exact assignment-compatible
JSON field names and structure; only Python class names change.

### Consequences

- Public Python APIs communicate retrieval concepts without coursework-specific
  terminology.
- Singular and batch result types are easier to distinguish.
- Serialized JSON remains compatible with the required submission schema.
- References to the former class names must be updated together during the
  rename.

### Lesson

Public model names should describe their domain role, while serialization
contracts should remain stable at external boundaries. When an assignment also
checks specific Python symbols, compatibility aliases can preserve both goals.

## 2026-08-27 - Reject invalid retrieval limits at the CLI boundary

**Status:** Accepted

### Initial approach

Parse `--k` as any integer and rely on the retrieval result boundary to reject
values less than one. This kept the invariant in the domain layer and still
prevented an invalid result from being produced.

### Why the approach was reconsidered

The real Linux acceptance run loads an approximately 87 MB persisted BM25
snapshot before control reaches the domain validation. A command with `--k 0`
would therefore perform avoidable index deserialization before reporting a
simple input error.

### Decision

Keep the domain validation as the final invariant, and also parse CLI `--k`
with a positive-integer validator. Reject non-positive values during argument
parsing, before the index path is opened or the retrieval workflow is called.

### Consequences

- Invalid interactive commands fail immediately with a focused argument error.
- Large index files are not loaded for a request that cannot succeed.
- Python callers remain protected by the existing domain-layer validation.
- The same rule intentionally exists at both the user boundary and the domain
  boundary because they protect different callers.

### Lesson

Validate inexpensive user constraints before starting expensive work, while
retaining domain-level checks for callers that bypass the command-line layer.

## 2026-08-27 - Match the evaluator contract at the CLI boundary

**Status:** Accepted

### Initial approach

Expose one `argparse` command named `search` for batch retrieval. Require users
to pass the index path, corpus fingerprint, input dataset path, output file,
and retrieval limit explicitly.

### Why the approach was reconsidered

The assignment invokes commands through Python Fire and assigns different
contracts to `search <query>` and `search_dataset`. The explicit internal
parameters made the implementation inspectable, but the evaluator could not
call the public interface it required. Expected file and validation failures
also escaped as unhandled tracebacks.

### Decision

Keep explicit paths and fingerprints inside reusable Python workflows. Expose
assignment-compatible Python Fire commands for single-query and batch search,
using the prescribed data directories by default and calculating the corpus
fingerprint automatically. Convert expected boundary failures into concise
stderr messages with a non-zero exit status. Retain domain-oriented result
models while providing the exact assignment model names as compatibility
wrappers.

### Consequences

- Reference evaluation scripts can call the required command names and
  arguments directly.
- Single-query and batch behavior are no longer conflated.
- Internal workflows remain independently testable and configurable.
- Expected invalid input, missing files, malformed JSON, and incompatible
  indexes do not produce an unhandled traceback.
- Assignment terminology remains isolated from the models used by retrieval
  internals.

### Lesson

A sound internal API does not compensate for an incompatible external
contract. Preserve strong domain boundaries, then adapt the public boundary to
the system that must invoke it.

## 2026-08-27 - Bind persisted indexes to the complete build pipeline

**Status:** Accepted

### Initial approach

Determine persisted-index compatibility from the JSON schema version and a
fingerprint of the discovered corpus paths and bytes. Raise `reindex required`
when either value differs from the current application and corpus.

### Why the approach was reconsidered

An indexing algorithm can change without changing either the source files or
the JSON field structure. For example, a tokenizer update can produce different
`content_terms` while the stored field remains a list of strings. A chunker
change can produce different source spans while every stored chunk remains
valid JSON. In both cases, the previous snapshot would pass the existing
compatibility checks even though it no longer represents the current build
pipeline.

Relying on developers to raise the schema version for every algorithm or
configuration change would make correctness depend on a manual convention that
the persistence layer could not verify.

### Decision

Define an immutable `PipelineConfig` containing the maximum chunk size, BM25
parameters, and explicit chunker and tokenizer versions. Calculate a canonical
SHA-256 pipeline fingerprint from that configuration together with the index
schema version.

Persist this fingerprint in schema version 2 and require it when loading an
index. Treat the schema version, corpus fingerprint, and pipeline fingerprint
as three independent compatibility checks. Build production indexes through a
single builder that consumes `PipelineConfig` and returns both fingerprints
with the runtime index.

### Consequences

- Changes to corpus bytes, stored representation, or declared build behavior
  independently require reindexing.
- Tokenizer, chunker, chunk-size, and BM25 changes cannot silently reuse stale
  lexical fields.
- Compatibility behavior is deterministic and can be tested without building
  the full corpus.
- Algorithm changes require an intentional tokenizer or chunker version update
  when their configuration fields do not otherwise change.
- Existing schema version 1 snapshots are intentionally incompatible and must
  be rebuilt as schema version 2.
- The public search boundary must calculate the same default pipeline identity
  used by the production index builder.

**Verification evidence:**

The production command rebuilt the 1,952-file vLLM corpus as 20,096 lexical
documents in schema version 2. The corpus fingerprint remained
`1745355afa9ef90effcc67802ad356e439dbf8a5d954e36ad4095d88b05bf1f2`,
and the declared pipeline fingerprint was
`03dbf67f929d95d8e405759fb5f5fbf7effcdc89c196d4ea410018484c8046d4`.
The documentation and code retrieval result files were byte-for-byte identical
to the schema version 1 baseline. Source validation accepted all 500
documentation sources and all 495 code sources.

### Lesson

Cache compatibility must describe how stored data was produced, not only what
its serialized shape looks like or which input files existed. Make every
behavioral input explicit, canonicalize it, and validate that identity before
reusing derived data.
