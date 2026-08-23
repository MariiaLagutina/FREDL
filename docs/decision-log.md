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
