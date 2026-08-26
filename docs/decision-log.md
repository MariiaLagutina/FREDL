# Engineering Decision Log

This log records implementation choices that were reconsidered during the
project. Each entry explains the original approach, the problem discovered,
the replacement decision, and the lesson to carry into later work.

## 2026-08-23 - Require the corpus to be below the project root

**Status:** Accepted

### Initial approach

File discovery required the corpus path to be inside the project root. The
check also allowed the corpus root and project root to be the same directory.

### Why the approach was unsafe

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

### Why the approach was unsafe

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

### Why the approach was inefficient

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

### Initial approach considered

Repeat a containing class or function header inside every structural chunk so
that each method or statement carries readable context during retrieval.

### Why the approach was unsafe

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

### Initial approach considered

Invalid bytes encountered while reading a complete source file could be
ignored or replaced to let indexing continue.

### Why the approach was unsafe

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

### Initial approach considered

Strip emphasis markers, link syntax, heading markers, and code-fence delimiters
before sending Markdown chunks to the embedding model.

### Why the approach was premature

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

### Initial approach considered

Apply a fixed overlap between every pair of Markdown or plain-text chunks to
retain context around all boundaries.

### Why the approach was inefficient

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

### Why the approach stopped scaling

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

### Why the approach lost information

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

### Initial approach considered

Preserve every apostrophe form as one normalized token. This avoids fragments
such as `isn` and `t`, but a possessive such as `model's` then cannot match a
query containing only `model`.

### Why unconditional suffix removal was unsafe

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

### Initial approach considered

Append path, heading, and symbol tokens to chunk content. Approximate metadata
weight by repeating those tokens before building one BM25 field.

### Why the approach was misleading

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

### Initial approach considered

Serialize the complete live `BM25Index` object, including its private postings
maps and derived statistics, with Python pickle. Loading would restore the
runtime object graph directly with minimal reconstruction work.

### Why the approach was rejected

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

### Verification

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
