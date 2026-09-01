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
