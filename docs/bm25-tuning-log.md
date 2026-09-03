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

## Linux reproduction - M0-M3 mini-suite

**Status:** Completed as a platform reproduction. Not parameter-selection
evidence.

**Constants:** The fixed `bm25-mini` suite, corpus fingerprint, queries,
chunking, tokenizers, `top_k`, warm-up, measured searches, and M0-M3 parameters
match the provisional run above.


**Environment:** Linux `7.0.0-30-generic` on `x86_64` with Python `3.14.4`.

**Results:**

| Run | Build ms | Index bytes | Peak build bytes | Median ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 | 0.160 | 44,214 | 34,568 | 0.020 | 0.025 |
| M1 | 0.079 | 44,166 | 34,480 | 0.020 | 0.036 |
| M2 | 0.080 | 44,166 | 34,448 | 0.020 | 0.028 |
| M3 | 0.076 | 44,142 | 34,424 | 0.020 | 0.041 |

**Ranking review:** Functional results were unchanged from the macOS run. All
four queries remained relevant within top-3 and top-5. M1-M3 each had zero
Recall@5 improvements, zero regressions, and four unchanged queries against M0.
First-relevant rank changes also matched exactly: M1=`0/1/3`, M2=`0/2/2`, and
M3=`1/2/1` for improved/regressed/unchanged.

**Interpretation:** The stable rankings confirm platform-independent functional
behavior for the fixed suite. Only build time, recursive Python object size,
peak Python allocations, and median/P95 latency were re-recorded because those
measurements depend on the runtime and hardware.

**Decision:** Keep M0 as the control and retain the earlier parameter decision.
Use the Linux report as reproduction evidence, not as justification for a new
metadata weight.

## Full-dataset observation after CLI repair

**Status:** Preliminary quality observation. Complete full-dataset metrics have
not yet been recorded.

**Constants:** The persisted 20,096-document BM25 index with `k1=1.5`,
`b=0.75`, `metadata_weight=1.0`, and `k=5`. The repaired Python Fire CLI
reproduced the earlier batch JSON byte for byte for both the 100-question
documentation dataset and the 99-question code dataset.

**Observation:** For code question
`189c8b8a-e59c-4fca-92ad-c02df42cbe40`, the labelled source is
`fused_batched_moe.py` at range `[28416, 28975)`. It was absent from the first
five retrieved sources, so this query has Recall@5 equal to zero.

**Interpretation:** Correct paths and ranges prove output validity, not
retrieval relevance. The four-query mini-suite verifies deterministic ranking
and controlled metadata effects but does not represent quality across all 199
public questions.

**Decision:** Preserve this miss as baseline evidence. Measure Recall@1/3/5/10
separately for the complete documentation and code datasets before changing
tokenization, chunking, metadata, or BM25 parameters.

## Full public datasets - lexical BM25 baseline

**Status:** Completed as the first full-dataset retrieval-quality baseline.

**Date:** 2026-08-31.

**Constants:** The persisted 20,096-document BM25 index with `k1=1.5`,
`b=0.75`, `metadata_weight=1.0`, and `k=5`. Retrieval uses lexical content
and structural metadata tokens only; embeddings and semantic vector search are
not part of this baseline. Relevance requires an exact `file_path` match and
half-open source-range IoU greater than or equal to `0.05`.

**Evaluator Git commit:**
`dfaf4aa479a81484b53f6435210ed9665d1d2911`.

**Inputs:**

| Role | Path | SHA-256 |
| --- | --- | --- |
| Docs ground truth | `data/datasets/AnsweredQuestions/dataset_docs_public.json` | `bbde6ed2efaf8966e8950568ee04e8ccaa3441569ba0b026642c9e6c45c2632c` |
| Docs retrieval results | `data/output/search_results/UnansweredQuestions/dataset_docs_public.json` | `d163856fd08ac34c649d125d2a1abd4ad2c010a85713fef4764d536600857127` |
| Code ground truth | `data/datasets/AnsweredQuestions/dataset_code_public.json` | `dfc00cd9707d23f9ebf8c604013222c1268bfcc672403d465d5246e7c6e0915c` |
| Code retrieval results | `data/output/search_results/UnansweredQuestions/dataset_code_public.json` | `1edd60ac96b948f48875ae23cc34d9fdfb4fc8ee65eeefd60d908b3e5b3d9fd4` |

**Command:**

```bash
uv run python -m src evaluate \
  --docs_ground_truth_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --docs_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --code_ground_truth_path data/datasets/AnsweredQuestions/dataset_code_public.json \
  --code_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json
```

**Results:**

| Dataset | Queries | R@1 | R@3 | R@5 | R@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Docs | 100 | 0.540000 | 0.740000 | 0.820000 | 0.820000 | 0.643000 |
| Code | 99 | 0.484848 | 0.676768 | 0.757576 | 0.757576 | 0.595623 |

**Interpretation:** The lexical baseline retrieves 82% of labelled
documentation sources and approximately 75.8% of labelled code sources within
the first five results. This run persisted only five sources per question, so
its R@10 necessarily equals R@5 and cannot measure the effect of ranks 6-10.
The later B0 run below corrects that limitation with `k=10`. Future gains must
be demonstrated through controlled changes to tokenization, chunking, metadata
weighting, or BM25 parameters on the same fixed inputs.

**Decision:** Preserve these values as the first full public-dataset control.
Do not change the baseline configuration from this run alone. Compare every
future candidate against the same input hashes and report Docs and Code
separately.

## B0 - full public-dataset lexical BM25 control

**Status:** Completed as the first official Moulinette-verified `k=10`
full-dataset control.

**Date:** 2026-09-01.

**Hypothesis:** The unchanged production BM25 configuration provides a strong,
reproducible lexical control when all ten required ranks are persisted and
evaluated independently for documentation and code.

**Changed factor:** Retrieval depth only. The earlier full-dataset observation
persisted `k=5`; B0 persists `k=10`. No ranking, chunking, tokenization, or BM25
parameter changed.

**Constants:** `k1=1.5`, `b=0.75`, `metadata_weight=1.0`;
`max_chunk_size=2000`; documentation overlap `160`; code overlap `80`;
20,096 indexed documents; `max_context_length=2000`; exact `file_path` and
source-range IoU `>=0.05`; lexical content and structural metadata only; no
embeddings or vector search.

**Git commit:** `97cefb76d7656149035138130780018df82399a2` with a clean
`rag-18-baseline-metrics` working tree before generated artifacts were created.

**Environment:** Linux `7.0.0-30-generic` on `x86_64` with Python `3.14.4`.

**Fingerprints:**

- corpus: `1745355afa9ef90effcc67802ad356e439dbf8a5d954e36ad4095d88b05bf1f2`;
- pipeline: `03dbf67f929d95d8e405759fb5f5fbf7effcdc89c196d4ea410018484c8046d4`;
- persisted B0 index SHA-256:
  `b68963b5c9c82a1e7fefe91feb33b96873967401c95ce08e244716a16960e3d3`;
- Docs result SHA-256:
  `1fe4df37e49506c901b23bf36144b4301ef41a0c7aa75ba2906cd85e12664457`;
- Code result SHA-256:
  `cb16428f094fdfdc116ce4a9ef50c3d8720bdf2d799f33340058e5466437a8b7`.

The ground-truth and question-dataset hashes remain those recorded in the
preceding full-dataset entry.

**Commands:**

```bash
uv run python -m src index \
  --index_path data/processed/experiments/B0/bm25-index.json

uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/experiments/B0/search_results \
  --index_path data/processed/experiments/B0/bm25-index.json \
  --k 10

uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
  --save_directory data/output/experiments/B0/search_results \
  --index_path data/processed/experiments/B0/bm25-index.json \
  --k 10

./moulinette/moulinette-ubuntu evaluate_student_search_results \
  data/output/experiments/B0/search_results/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k=10 --max_context_length=2000

./moulinette/moulinette-ubuntu evaluate_student_search_results \
  data/output/experiments/B0/search_results/dataset_code_public.json \
  data/datasets/AnsweredQuestions/dataset_code_public.json \
  --k=10 --max_context_length=2000
```

**Moulinette results:**

| Dataset | Queries | R@1 | R@3 | R@5 | R@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Docs | 100 | 0.540000 | 0.740000 | 0.820000 | 0.870000 |
| Code | 99 | 0.484848 | 0.676768 | 0.757576 | 0.828283 |

The local evaluator matched all eight Moulinette Recall values exactly. It
also reported Docs MRR `0.650123` and Code MRR `0.604678`; Moulinette does not
report MRR for this command.

**Measurements:**

| Stage | Elapsed seconds | Peak RSS KiB |
| --- | ---: | ---: |
| Index build | 15.18 | 1,193,388 |
| Docs search, 100 queries | 8.91 | 838,172 |
| Code search, 99 queries | 8.56 | 838,292 |

The combined search took `17.47` seconds for 199 queries, equivalent to
approximately `17.56` seconds per 200 queries. The persisted index size was
90,701,038 bytes. Moulinette validated both result files and evaluated every
question with a labelled and retrieved source list.

**Interpretation:** Ranks 6-10 recover five additional documentation questions
and seven additional code questions beyond R@5. This raises Docs recall from
`0.82` to `0.87` and Code recall from `0.757576` to `0.828283`. The exact local
and official metric agreement independently validates the local evaluator;
the metric levels themselves measure retrieval quality.

**Decision:** Adopt B0 as the fixed full-dataset control for subsequent
single-factor experiments. Preserve its commit, input hashes, corpus and
pipeline fingerprints, `k=10`, and Moulinette settings for every comparison.

## M1 - moderate metadata boost

**Status:** Completed and rejected. The planned M2 and M3 metadata-weight runs
are cancelled by the stopping rule below.

**Date:** 2026-09-01.

**Hypothesis:** Increasing `metadata_weight` from `1.0` to `1.5` may improve
exact path, heading, and symbol matches without reducing content relevance.

**Changed factor:** Only `metadata_weight`, from B0's `1.0` to `1.5`.

**Constants:** B0 corpus and datasets; `k1=1.5`; `b=0.75`;
`max_chunk_size=2000`; documentation overlap `160`; code overlap `80`; 20,096
indexed documents; `k=10`; `max_context_length=2000`; no embeddings or vector
search.

**Git commit:** `eecd2f6`, which added production-compatible CLI parameters and
ensured that index and search commands use the same parameter-derived pipeline
fingerprint.

**Fingerprints and artifact hashes:**

- corpus: `1745355afa9ef90effcc67802ad356e439dbf8a5d954e36ad4095d88b05bf1f2`;
- M1 pipeline: `e54708fd97578cc3e34388e4f2d6a8576ad9fee347dde097301bc30390e54a33`;
- persisted M1 index SHA-256:
  `5bc775c996dc7bf950ca3e6e456ea6f4ffbb969b317459f3f7b2e98c401e4e7e`;
- Docs result SHA-256:
  `820a40f798a7a157f085d597dd616f5a988190afeef96eee6b2ac92c4df7d7a7`;
- Code result SHA-256:
  `f7d27597c2b0d7032a7f7998e00692538ab5b402623cc4521c827eb4372289e2`.

**Moulinette results:**

| Dataset | R@1 | R@3 | R@5 | R@10 | R@10 change from B0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Docs | 0.470000 | 0.660000 | 0.750000 | 0.830000 | -0.040000 |
| Code | 0.494949 | 0.676768 | 0.737374 | 0.828283 | 0.000000 |

The local evaluator matched all eight Moulinette Recall values exactly. It
also reported Docs MRR `0.580440` and Code MRR `0.601527`.

**Measurements:** The index build took `25.03` seconds with peak RSS
`1,193,448` KiB. Sequential Docs and Code searches took `16.53` and `15.19`
seconds respectively, equivalent to `31.88` seconds per 200 queries. Runtime
measurements are recorded as operational evidence but are not used to select
the ranking parameter because system load differed from the B0 run.

**Interpretation:** M1 reduced every Docs Recall value. For Code, R@1 improved
by approximately one percentage point, R@3 and R@10 were unchanged, and R@5
fell by approximately two percentage points. The isolated metadata increase
therefore produced no dataset-level retrieval improvement and materially
damaged documentation retrieval.

**Stopping rule and decision:** Reject `metadata_weight=1.5` and retain the B0
value `1.0`. Do not run the planned M2 (`2.0`) or M3 (`3.0`) experiments: M1
already moved the primary Recall metrics in the wrong direction, while further
increases would amplify the same scoring component and have low expected
information value. Revisit higher metadata weights only if a later change
alters metadata tokenization or field composition; that would create a new
interaction hypothesis rather than continue this single-factor sequence.

## K1 - term-frequency saturation comparison

**Status:** Completed. Adopt `k1=1.4` as the ranking baseline for the next
single-factor experiment series.

**Date:** 2026-09-01.

**Hypothesis:** A small reduction from B0's `k1=1.5` may limit the influence of
repeated terms while preserving the strong exact lexical matches of the plain
BM25 baseline.

**Changed factor:** Only `k1`. The comparison covers `1.2`, `1.4`, B0's `1.5`,
and `1.8`.

**Constants:** B0 corpus and datasets; `b=0.75`; `metadata_weight=1.0`;
`max_chunk_size=2000`; documentation overlap `160`; code overlap `80`; 20,096
indexed documents; `k=10`; `max_context_length=2000`; no embeddings or vector
search.

**Moulinette and local evaluator results:**

| k1 | Dataset | R@1 | R@3 | R@5 | R@10 | Local MRR |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1.2 | Docs | 0.520000 | 0.730000 | 0.820000 | 0.860000 | 0.635583 |
| 1.2 | Code | 0.515152 | 0.666667 | 0.767677 | 0.838384 | 0.619929 |
| 1.4 | Docs | 0.530000 | 0.740000 | 0.820000 | 0.870000 | 0.645623 |
| 1.4 | Code | 0.494949 | 0.676768 | 0.767677 | 0.838384 | 0.611344 |
| 1.5 | Docs | 0.540000 | 0.740000 | 0.820000 | 0.870000 | 0.650123 |
| 1.5 | Code | 0.484848 | 0.676768 | 0.757576 | 0.828283 | 0.604678 |
| 1.8 | Docs | 0.550000 | 0.740000 | 0.810000 | 0.880000 | 0.656413 |
| 1.8 | Code | 0.474747 | 0.676768 | 0.747475 | 0.808081 | 0.594440 |

The local evaluator matched all Moulinette Recall values exactly for all six
new result files.

**Fingerprints and artifact hashes:**

| k1 | Pipeline fingerprint | Index SHA-256 | Docs result SHA-256 | Code result SHA-256 |
| ---: | --- | --- | --- | --- |
| 1.2 | `62beb08bd6a7a49f96a8300ef79d734e8e18ebf1e8e3d0e4a6d1546294641be8` | `d5a6af9a7b057a6e61aa022d06d49a79dbd1475a2bf3e4806fee678283446f2a` | `a0e743a22279b290e281d738d6d18e33372de737675674d5072d53d4946b806f` | `052bf1504422f634695af8d57c93c2dac8cfd9c66b4727fb52b009190e67982d` |
| 1.4 | `fffb23851d80668d302a001e19df45e2bd9f534968e721a85cf7396c78906875` | `d58ca26fd0ebee6605214ef63ae65e0b294859618c3f3924e52345c8f86b656f` | `81175b44d5631c5fa1074663ee37b491d9f3c873b6b0385f06d3d73b2390b0dc` | `359e2b975c323c137b769421bbe268fefaa77c67ed793e9827cfe4f70dd79e44` |
| 1.8 | `838074cbc4685965fe133d7527775487f20dd68e1a4103bd7e2c5c4b5ef4b121` | `c05f9ec7d80633e372198da1646ae1d1378bd017c823c90edc7e85f98677a447` | `e828acc6c50e27735497bb010947f0ce6cdcbe1d03214321a12b92453ada84db` | `4659456084c3f3154fa4f4cf472f21032aeb44019f5d69dd7399d7c1ea319128` |

**Measurements:**

| k1 | Index seconds | Docs search seconds | Code search seconds | Search seconds per 200 queries |
| ---: | ---: | ---: | ---: | ---: |
| 1.2 | 25.36 | 16.41 | 15.54 | 32.11 |
| 1.4 | 24.99 | 15.89 | 15.14 | 31.19 |
| 1.8 | 24.92 | 16.16 | 15.09 | 31.41 |

Runtime is retained as operational evidence but is not a ranking-selection
criterion because these small differences can reflect system load.

**Per-question evidence for `k1=1.4` versus B0:**

- Docs: the labelled source for the fastest matrix multiplication kernel
  question moved from rank 5 to rank 4. The labelled source for the sharding
  and quantization of model weights question moved from rank 1 to rank 2.
- Code: the labelled source for the default `z` value in
  `selective_state_update` entered the top ten at rank 9; it was absent from
  B0's top ten. The `OvisConfig.depths` source moved from rank 4 to rank 5.
  The supported `W4A16Sparse24` bit values source moved from rank 7 to rank 6,
  the `gate_up_proj` `shard_id` source moved from rank 2 to rank 1, and the
  default `lora_int_id` source moved from rank 8 to rank 5.

**Interpretation:** No candidate dominates both datasets at every cutoff.
`k1=1.8` slightly improves Docs but reduces Code at R@1, R@5, and R@10.
`k1=1.2` gives the highest Code R@1 but reduces Docs at R@1, R@3, and R@10.
Relative to B0, `k1=1.4` preserves Docs R@3, R@5, and R@10 while losing one
Docs R@1 hit. It preserves Code R@3, gains one Code hit at R@1, R@5, and R@10,
and raises Code MRR. Across both datasets the number of R@1 hits is unchanged,
while R@5 and R@10 each gain one hit. These are deterministic improvements on
the fixed public dataset, but their one-question size should not be treated as
evidence of broad statistical significance.

**Decision:** Adopt `k1=1.4` as the unified ranking baseline for the next
series. Keep `metadata_weight=1.0`, then vary only `b` around B0's `0.75` to
measure document-length normalization independently. Begin with `b=0.5` and
`b=1.0`; use the fixed datasets, corpus, chunking, `k=10`, and evaluator
settings above.

## B - document-length normalization comparison

**Status:** Completed. Adopt `b=0.65` and stop this single-factor series.

**Date:** 2026-09-01.

**Hypothesis:** Reducing length normalization from `b=0.75` may improve early
Code ranks without materially reducing documentation retrieval quality.

**Changed factor:** Only `b`. The comparison covers `0.5`, `0.6`, `0.65`,
`0.70`, the adopted K1 baseline's `0.75`, and `1.0`.

**Constants:** B0 corpus and datasets; `k1=1.4`;
`metadata_weight=1.0`; `max_chunk_size=2000`; documentation overlap `160`;
code overlap `80`; 20,096 indexed documents; `k=10`;
`max_context_length=2000`; no embeddings or vector search.

**Moulinette and local evaluator results:**

| b | Dataset | R@1 | R@3 | R@5 | R@10 | Local MRR |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.50 | Docs | 0.550000 | 0.690000 | 0.790000 | 0.860000 | 0.637980 |
| 0.50 | Code | 0.545455 | 0.696970 | 0.787879 | 0.838384 | 0.642825 |
| 0.60 | Docs | 0.540000 | 0.710000 | 0.790000 | 0.880000 | 0.642579 |
| 0.60 | Code | 0.535354 | 0.676768 | 0.777778 | 0.838384 | 0.631630 |
| 0.65 | Docs | 0.540000 | 0.730000 | 0.810000 | 0.880000 | 0.646496 |
| 0.65 | Code | 0.525253 | 0.676768 | 0.767677 | 0.838384 | 0.627746 |
| 0.70 | Docs | 0.530000 | 0.730000 | 0.810000 | 0.870000 | 0.641123 |
| 0.70 | Code | 0.505051 | 0.676768 | 0.767677 | 0.838384 | 0.617404 |
| 0.75 | Docs | 0.530000 | 0.740000 | 0.820000 | 0.870000 | 0.645623 |
| 0.75 | Code | 0.494949 | 0.676768 | 0.767677 | 0.838384 | 0.611344 |
| 1.00 | Docs | 0.540000 | 0.770000 | 0.820000 | 0.860000 | 0.658750 |
| 1.00 | Code | 0.444444 | 0.656566 | 0.707071 | 0.757576 | 0.560630 |

The local evaluator matched all forty Moulinette Recall values exactly for the
ten new result files.

**Fingerprints and artifact hashes:**

| b | Pipeline fingerprint | Index SHA-256 | Docs result SHA-256 | Code result SHA-256 |
| ---: | --- | --- | --- | --- |
| 0.50 | `365dfc507e32cd6c1752b96020dfa438c8a1da3b741f84d1567f0a8b3edd4fd5` | `b4f7a950169a1b46f98b42fd557d3af97fa0529e46ac053a2efccb838e66d8d3` | `95f3c9a0fcf5fdf5278de92d25c8d112d3e1ad1b027d4ad6be85f0353e843b44` | `6ade46c70fd5d6fd22a7b36f744f12f3dddfe4ea63784eacf210704559d5b5a5` |
| 0.60 | `3b659657a1099b2930b006247f7c64e4472ba83990986e8b4a1da50862d17f7e` | `d829e6883922c75b6810698c5112351e5f8a37f10c56746906eb1ff2e191b0e2` | `c028c0fa8608bdad89944509e1641c7fe338f8b045594c158855d01242254ccd` | `fa2d4e59eb4728e5683da7dd3546b1713cce0a939e0339466b07a33e9124e4f2` |
| 0.65 | `ebc6d3ffc6535965b94aab7651bd837468a9c4ebefb81570818179741afe511f` | `ccda3204f24b87706eb8a01d56d39e36b27d8485da0870659af2396d33212092` | `5a77414dde064181f14c70e3f695f15c8717ed6cc04ae629513fb1014dc0c652` | `b2ff9dacad41acb46cbf21a1b96b602ddb5d7d44693dcddccceb377efcce25d0` |
| 0.70 | `b93c76c69e7dc8861af02deb5a8e018ef225b2fc6ecff6c31fcc45bc1fa12002` | `5d0a9fae9f31eae9da880414ae05c1fcf3f543b464e7804e3c63c80d57b056fd` | `ac1643d8cfc9c5636718bf277e013101bc484ae4d5cd233c077a8a7c5913da47` | `349b8001f28ad662bfd6bfc1ae75ecae36869dc14ca06e63668f3f9eab665ea5` |
| 1.00 | `cd94787bb092a82fab2078aa9ec219454599de6cac6556182134e56550678fa7` | `9a37d9a00571c84fbd03256f0ef8c87eefc982e159776fa4cdbea32b9df658ad` | `8751016ece2e81926068970093fd287d8c38509dee8482f947e3a2e394876337` | `a3de89531b22848efdea48d97b5a1b246906934a91cb28ef1eed2323717a1218` |

**Measurements:**

| b | Index seconds | Docs search seconds | Code search seconds | Search seconds per 200 queries |
| ---: | ---: | ---: | ---: | ---: |
| 0.50 | 25.35 | 16.52 | 15.33 | 32.01 |
| 0.60 | 24.91 | 16.34 | 15.68 | 32.18 |
| 0.65 | 25.30 | 16.46 | 15.43 | 32.05 |
| 0.70 | 24.96 | 16.59 | 15.26 | 32.01 |
| 1.00 | 25.30 | 16.53 | 15.50 | 32.19 |

Runtime is retained as operational evidence but is not a ranking-selection
criterion because these differences can reflect system load.

**Interpretation:** Lower `b` values strongly improve early Code ranks but
eventually damage Docs R@3 and R@5. Full normalization at `b=1.0` improves
Docs R@3 but substantially reduces every Code Recall value. Relative to
`b=0.75`, `b=0.65` gains one Docs R@1 and R@10 hit, loses one Docs R@3 and R@5
hit, gains three Code R@1 hits, preserves Code R@3, R@5, and R@10, and raises
MRR for both datasets. The neighbouring `b=0.70` candidate is dominated by
`b=0.65`: all other Recall values tie, while `0.65` has higher Docs R@1 and
R@10, higher Code R@1, and higher MRR for both datasets.

**Stopping rule and decision:** Adopt `b=0.65` with `k1=1.4` and
`metadata_weight=1.0`. Do not split the `0.65-0.70` interval further: the
nearest tested neighbour produced no compensating gain, and the remaining
differences on this fixed public dataset are already one-question effects.
Preserve `b=0.75` as the prior K1 control and use `b=0.65` as the unified
ranking baseline for the next experiment family.

## Phase 19 - retrieval error analysis

**Status:** Completed for all top-five misses in the adopted Phase 18 run.

**Date:** 2026-09-03.

**Inputs:** The `FINAL` result files produced with `k1=1.4`, `b=0.65`,
`metadata_weight=1.0`, and `k=10`, together with the unchanged public Docs and
Code ground-truth datasets.

Generate a reviewable Markdown report from the adopted Phase 18 result files:

```bash
uv run python -m src analyze_retrieval_errors \
  --docs_ground_truth_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --docs_results_path data/output/experiments/FINAL/search_results/dataset_docs_public.json \
  --code_ground_truth_path data/datasets/AnsweredQuestions/dataset_code_public.json \
  --code_results_path data/output/experiments/FINAL/search_results/dataset_code_public.json
```

The command writes
`data/output/evaluation/retrieval-error-analysis.md`. The generated report is
local evidence and is excluded from Git. It preserves ground-truth question
order and lists every top-five miss with its labelled source, all ten ranked
sources, the first relevant rank when one exists, the exact reference excerpt,
and the top-three retrieved excerpts. Human-reviewed classifications are stored
separately from measured rankings in the local
`data/output/evaluation/retrieval-error-annotations.json` artifact.

**Classification method:** Rank positions, source paths, and half-open ranges
provide deterministic evidence for `relevant_below_top_5`, `chunk_boundary`,
and overlapping `duplicate_results`. Content review compares the question,
reference excerpt, and top-three retrieved excerpts before assigning one
dominant category to each remaining miss. A repeated file path alone is not a
duplicate: retrieved ranges must overlap. Each reviewed miss records a
hypothesis, proposed fix, and one next test.

**Results:**

| Category | Docs | Code | Total |
| --- | ---: | ---: | ---: |
| `wrong_file` | 6 | 1 | 7 |
| `relevant_below_top_5` | 7 | 7 | 14 |
| `chunk_boundary` | 2 | 1 | 3 |
| `lost_identifier` | 1 | 11 | 12 |
| `paraphrase` | 1 | 2 | 3 |
| `duplicate_results` | 0 | 0 | 0 |
| `noisy_metadata` | 2 | 1 | 3 |
| **Total top-five misses** | **19** | **23** | **42** |

All 42 misses have exactly one category. Docs' dominant category is
`relevant_below_top_5` with seven misses, closely followed by `wrong_file`
with six. Code's dominant category is `lost_identifier` with eleven misses.

**Representative evidence:**

- The Docs question asking which `LLM` method generates embeddings retrieves
  the implementation of `LLM.embed` at rank two, but the label requires
  `docs/models/pooling_models.md`. This is `wrong_file`: retrieval understood
  the subject but did not satisfy the dataset's source kind.
- The Code question asking for `FP8_MIN` and `FP8_MAX` retrieves general FP8
  utilities and benchmarks instead of their definitions in
  `triton_flash_attention.py`. This is `lost_identifier`.
- The Marlin MOE CUDA-architecture question retrieves an adjacent range in the
  correct `CMakeLists.txt`, separated from the label by two characters. This is
  `chunk_boundary`.
- Seven misses in each dataset already contain the labelled source at ranks
  6-10. Increasing retrieval depth exposes these sources but does not improve
  top-five recall; ranking must move them above the evaluation cutoff.

**Interpretation:** Documentation errors are split between insufficient early
ranking and returning a semantically useful source of the wrong kind. Code
errors are much more concentrated: exact class, method, parameter, attribute,
and constant identifiers are present in the questions but their labelled
definitions are not selected. No top-five miss contains overlapping retrieved
ranges, so duplicate results are not a supported explanation for this run.

**Next test:** Start with an identifier-aware lexical ranking experiment that
preserves complete snake_case, CamelCase, private-attribute, and uppercase
constant identifiers as high-value query terms. Measure Docs and Code
separately against the unchanged `FINAL` control. Treat a Docs-versus-Code path
preference as a separate later factor so that any effect remains attributable.
