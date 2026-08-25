# BM25 Tuning Log

This log records controlled BM25 retrieval experiments. Its purpose is to
preserve not only the selected values, but also the rejected values, measured
effects, parameter interactions, and reasoning behind each decision.

The engineering decision log records accepted project decisions. This file
keeps the quantitative path that leads to those decisions.

## Rules for a valid comparison

- Change one factor per experiment whenever possible.
- Keep the corpus, dataset split, chunking, tokenizer, and `k` fixed unless
  one of them is the factor under test.
- Record the Git commit and corpus fingerprint for every measured run.
- Report documentation and code datasets separately before any combined view.
- Compare retrieval metrics together with build time, index size, peak memory,
  and query latency.
- Inspect failed queries instead of choosing a configuration from one average.
- Repeat a surprising result before treating it as evidence.
- Do not replace the current baseline until the candidate wins on the same
  fixed evaluation set.

## Parameters and expected effects

### `k1`: term-frequency saturation

`k1` controls how much repeated occurrences of a query term can continue to
increase a document score.

- A lower value saturates quickly. The first match matters, while additional
  repetitions add little.
- A higher value allows repeated terms to contribute more.
- Too low can hide genuinely focused chunks.
- Too high can reward generated lists, repeated identifiers, or boilerplate.

Values should initially be tested around `1.2`, `1.5`, and `1.8`.

### `b`: document-length normalization

`b` controls how strongly BM25 compensates for document length.

- `0.0` disables length normalization.
- `1.0` applies full normalization relative to average document length.
- A higher value protects short focused chunks from long chunks that match by
  chance.
- A value that is too high can penalize long but necessary code or conceptual
  documentation sections.

Values should initially be tested around `0.5`, `0.75`, and `1.0`.

### `metadata_weight`: structural-signal strength

`metadata_weight` controls the contribution of terms derived from file paths,
section headings, class names, and function names.

- `1.0` is the neutral baseline.
- A value above `1.0` favors structural matches over ordinary body matches.
- A value that is too low may miss a relevant chunk whose body assumes context
  supplied by its class or section.
- A value that is too high can retrieve a file because its path looks relevant
  even when the chunk content does not answer the query.

Values should initially be tested at `1.0`, `1.5`, `2.0`, and `3.0`.

The weight must support fractional values exactly. It must not be implemented
by repeating metadata tokens because repetition cannot express `1.5` exactly
and also changes term-frequency saturation through `k1`.

Content and metadata are scored independently, then combined:

```text
final_score = content_score + metadata_weight * metadata_score
```

The score explanation for every hit must expose all three values. This keeps a
metadata boost observable and prevents it from silently changing content term
frequency.

## Interactions to watch

### `k1` and `b`

Long chunks naturally contain more repeated terms. Increasing `k1` lets those
repetitions matter more, while increasing `b` penalizes the length that made
them possible. A configuration can therefore look stable only because the two
effects cancel each other. After selecting each parameter independently, test
the neighboring `k1` and `b` combinations around the apparent winner.

### `k1` and `metadata_weight`

Content and metadata use separate term-frequency calculations, so metadata
cannot make content saturate early. However, a higher `k1` can still increase
both field scores before `metadata_weight` is applied. Inspect the separate
field scores to distinguish an actual structural improvement from a global
term-frequency effect.

### `b` and `metadata_weight`

Content and metadata must have separate field lengths and normalization
statistics. The multiplier is applied only after both field scores have been
calculated, so changing `metadata_weight` does not change either field length.
The implementation must still state whether metadata uses the same `b` value
as content. That choice remains fixed throughout the first weight experiment.

### Chunking and every BM25 parameter

Changing chunk size or overlap changes average document length, term frequency,
and document frequency. BM25 values selected for one chunking strategy cannot
be assumed to remain optimal after chunking changes. Retune only after a
chunking change has independently justified a new baseline.

## Metrics and evidence

Every metric is reported separately for documentation and code datasets. A
combined number may be added, but it never replaces the two required views.

Record at least:

- Recall@1, Recall@3, Recall@5, and Recall@10;
- mean reciprocal rank when a single first relevant hit is meaningful;
- index build time;
- serialized index size when persistence exists, plus the in-memory index size
  using a documented measurement method;
- peak memory during index construction when the platform supports a reliable
  measurement;
- per-query latency after warm-up, reported as median and P95, plus total time
  for a fixed query count;
- count of empty or skipped lexical documents;
- counts of improved, regressed, and unchanged queries relative to the control;
- Git commit and corpus fingerprint;
- a short review of improved and regressed queries.

For improvement and regression counts, compare per-query Recall@5 against the
control on the same dataset. A higher value is an improvement, a lower value is
a regression, and an equal value is unchanged. First-relevant rank changes may
be recorded separately and must not be mixed into these counts.

## Moulinette-compatible overlap criterion

A retrieved source matches a reference source only when:

1. `file_path` is exactly equal; and
2. the character-range intersection over union is at least `0.05`.

For the project's half-open ranges `[start, end)`, calculate:

```text
intersection = max(0, min(result_end, reference_end)
                      - max(result_start, reference_start))
union = (result_end - result_start)
        + (reference_end - reference_start)
        - intersection
iou = intersection / union
match = same_file and iou >= 0.05
```

Any intersection is not sufficient. A diagnostic report may also show raw
intersection length or other IoU thresholds, but Recall@k used for configuration
selection must follow the `0.05` rule stated by the assignment.

## Planned baseline sequence

| Run | `k1` | `b` | Metadata weight | Purpose |
| --- | ---: | ---: | ---: | --- |
| M0 | 1.5 | 0.75 | 1.0 | Neutral metadata control |
| M1 | 1.5 | 0.75 | 1.5 | Moderate metadata boost |
| M2 | 1.5 | 0.75 | 2.0 | Strong metadata boost |
| M3 | 1.5 | 0.75 | 3.0 | Stress test for metadata dominance |

Only `metadata_weight` changes in this sequence. The best supported value then
becomes fixed while `k1` is tested, followed by `b`. A small interaction grid
around the apparent winner is the final step, not the starting point.

## Experiment entry template

### Run ID - short name

**Status:** Planned / Completed / Rejected

**Hypothesis:** State the expected measurable effect.

**Changed factor:** Record the single value changed from the control.

**Constants:** Record corpus fingerprint, dataset split, chunking, tokenizers,
`k`, and all unchanged BM25 parameters.

**Git commit:** Record the exact commit used for the run.

**Results:** Record docs and code metrics separately, median and P95 query
latency, build time, index size, peak memory, and skipped-document counts.

**Query review:** Note important improvements, regressions, and whether matches
came from content or metadata. Include improved, regressed, and unchanged query
counts against the named control run.

**Interpretation:** Explain the likely mechanism without claiming more than the
results demonstrate.

**Decision:** Keep, reject, repeat, or design the next controlled run.

## M0 - neutral metadata control

**Status:** Completed for mini-suite validation only. Not parameter-selection
evidence.

**Hypothesis:** Metadata terms can improve structural matching without an
explicit boost, providing a clean control for later weight experiments.

**Changed factor:** None. This is the initial measured baseline.

**Constants:** `k1=1.5`, `b=0.75`, `metadata_weight=1.0`; fixed `bm25-mini`
suite; corpus fingerprint
`64f10584eb6ff1d4d666294d79b97402d5e39bb261873e5173658e01069a99b9`;
8 source files; 9 indexed chunks; 2 documentation and 2 code queries;
`max_chunk_size=2000`; `top_k=10`; 1 warm-up and 30 measured searches per
query.

**Git commit:** `60b4170` with `git_dirty=false` in the generated report.

**Environment:** Provisional local run on `Darwin x86_64` with Python
`3.10.19`. Functional rankings are recorded below. Build and latency values
must be repeated on the shared Linux evaluation machine before performance
claims are made.

**Command:**

```bash
.venv/bin/python -m src.evaluation.bm25 \
  --suite mini \
  --compare M0 M1 M2 M3
```

**Results:**

| Run | Metadata weight | Docs R@1 | Docs R@3/5/10 | Docs MRR | Code R@1 | Code R@3/5/10 | Code MRR | Build ms | Index bytes | Peak build bytes | Median ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 | 1.0 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 0.75 | 0.226 | 43,730 | 33,160 | 0.036 | 0.042 |
| M1 | 1.5 | 0.50 | 1.00 | 0.75 | 0.50 | 1.00 | 0.75 | 0.141 | 43,706 | 33,160 | 0.037 | 0.057 |
| M2 | 2.0 | 0.50 | 1.00 | 0.75 | 0.00 | 1.00 | 0.50 | 0.148 | 43,730 | 33,160 | 0.035 | 0.038 |
| M3 | 3.0 | 0.50 | 1.00 | 0.75 | 0.50 | 1.00 | 0.75 | 0.129 | 43,730 | 33,160 | 0.035 | 0.044 |

Index size recursively applies `sys.getsizeof` to the object graph retained by
the index and counts shared objects once. Peak build memory is the incremental
Python allocation peak from a separately warmed `tracemalloc` build. The
24-byte M1 index-size difference is object-sharing noise at this scale; all
four indexes are effectively 42.7 KiB. These methods do not measure native
library allocations or process RSS.

All four runs retained every relevant source within the first three results,
so Recall@5 comparison reported 0 improvements, 0 regressions, and 4 unchanged
queries for M1, M2, and M3. First-relevant rank was more discriminating:

| Candidate | Improved | Regressed | Unchanged |
| --- | ---: | ---: | ---: |
| M1 | 0 | 1 | 3 |
| M2 | 0 | 2 | 2 |
| M3 | 1 | 2 | 1 |

**Query review:** M3 moved `code-cache-store` from rank 2 to rank 1 because
the relevant file-path metadata overcame repeated body terms in a distractor.
M1 and higher moved `docs-request-retry-delay` from rank 1 to rank 2 because an
irrelevant filename exactly matched the query. M2 and higher also moved
`code-request-timeout-validation` from rank 1 to rank 2 for the same reason.
The separate field scores in verbose output confirmed that these changes came
from metadata rather than content score changes.

**Interpretation:** The runner detects both intended effects: structural terms
can rescue a relevant result, and excessive structural weight can make an
attractive filename dominate relevant content. The mini-suite is deliberately
small, so Recall@5 saturates and latency is below a meaningful benchmarking
scale. The suite validates mechanics and explanations; it cannot identify the
best production weight.

**Decision:** Accept the runner and the mini-suite as reproducible acceptance
evidence. Keep M0 as the control configuration. Do not select a metadata weight
from this run. Repeat M0-M3 on the complete, separately labelled documentation
and code query sets on one Linux machine. Record index size and peak memory once
those measurements are implemented.

## M1-M3 - mini-suite metadata stress comparison

**Status:** Completed as part of the provisional mini-suite validation above.

**Changed factor:** Only `metadata_weight`: M1=`1.5`, M2=`2.0`, M3=`3.0`.
All other constants, evidence, results, and the decision are recorded in the M0
entry so the controlled comparison remains in one place.
