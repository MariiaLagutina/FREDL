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
