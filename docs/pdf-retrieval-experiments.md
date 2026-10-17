# PDF Retrieval Experiments

> **Research record.** This journal reports private experiments as aggregate
> counts only: no question text, book passages, model answers, book titles,
> page numbers, case identifiers, private paths or private fingerprints.
> Observations marked as assistant checks are not human judgments.

## B1: multi-book retrieval baseline (16 October 2026)

### Purpose and success criteria

B1 measures how well the existing BM25, MiniLM and Hybrid retrievers find
supporting evidence in the real four-book PDF library, using the B0 evaluation
pipeline without changing retrieval or evaluation code. The aim is retrieval
that reliably delivers sufficient evidence to a downstream LLM, not the
highest possible score.

Success criteria:

- the primary metric is `BER@5` for complete original questions
  (`headline_complete:original`, ranked-only and with direct context);
- `BER@10`, sufficient-set success, localization and the two MRR diagnostics
  explain the primary result;
- clarified interpretations are reported separately, weighted by case;
- a numeric "good enough for LLM evidence" threshold is **not yet defined**
  and needs a decision once the judged baseline exists.

### Methodology and limitations

- Metrics follow `docs/pdf-evaluation.md`: metric policy
  `declared-sets-query-macro-v2` and supply policy
  `required-and-explicit-yes-v1`. `required` relevance alone does not supply
  evidence, `supporting` never closes a unit, unjudged records are never
  irrelevant, and queries without an eligible sufficient set are null.
- Final metrics require a human label for every unit/record binding in each
  engine's top-10 and direct context. Without those labels the evaluator
  refuses to publish metrics, so B1 currently has **no B0 metrics**.
- The Golden Set is an unfrozen draft. Results computed from it remain
  provisional until it is frozen.
- The sample is small: 12 complete original questions carry the primary
  headline. Differences of one or two questions are not meaningful.
- The anchor-overlap counts below come from pool provenance. They show whether
  a record overlapping a verified evidence anchor was ranked; they are not
  relevance judgments, not BER and not evidence supply.

### Corpus

| Item | Value |
| --- | --- |
| Books | 4 technical PDF books on AI and machine-learning engineering |
| Extracted records | 2163 |
| Eligible records under P2 | 2043 (per book: 496, 756, 360, 431) |
| Corpus policy | P2 `exclude_navigation_and_boilerplate`, `pdf-retrieval-corpus-v1` |
| Chunking | PDF reading-flow chunks of at most 1600 characters (600 minimum target; 82 shorter records end a flow or section), sentence boundaries preferred |
| Median record length | 1434 characters; 304 MiniLM word pieces |

Book titles, hashes and record-set fingerprints are kept in the private run
manifests.

### Golden Set

| Item | Value |
| --- | --- |
| Schema | `fredl.pdf.multibook-golden` version 2 |
| Status | `draft` (not frozen) |
| Cases | 22 |
| Evidence units | 24 (19 text, 4 structured text, 1 figure) |
| Expected outcomes | 12 complete, 2 partial, 2 extraction-limited, 2 ambiguous, 3 unsupported, 1 source-incomplete |
| Cases with sufficient sets | 16 (two with alternative sets; set sizes 1–3 units) |
| Ambiguous cases | 2, with 2 and 3 approved clarified interpretations |
| Query variants | 27 (22 original, 5 clarified) |

Categories: 4 single-book, 2 multiple-sections single-book, 2 independent
multi-book, 3 complementary multi-book, 2 ambiguous terminology,
2 partially answerable, 1 unsupported library, 2 hard negative and
4 reliability-limited.

### Retrieval configuration

| Setting | Value |
| --- | --- |
| Code revision | `c52e402` (clean working tree) |
| BM25 | `k1 = 1.4`, `b = 0.65`, metadata weight 1.0, identifier weight 0.0, tokenization `text-content-code-metadata-v1` |
| MiniLM | `sentence-transformers/all-MiniLM-L6-v2` revision `1110a243`, CPU float32, mean pooling, L2, `max_length = 256`, batch 32, local files only |
| Hybrid | RRF, rank constant 60, lexical weight 1.0, semantic weight 1.0, global candidate rankings |
| Ranking | complete global ranking over all eligible records; no book quotas |
| Context | `same-book-flow-interval-touching-neighbors-v1`, cutoffs 5 and 10, original record text |
| Query mode | `all`: original and approved clarified questions |

Golden Set, library, corpus, query-plan, run and pool fingerprints are
recorded in the private artifacts and are withheld here until review.

### Experiment B1-1: baseline runs and judgment pool

Procedure, run offline (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`) from a
clean checkout, with private paths replaced by placeholders:

```bash
for retriever in bm25 minilm hybrid; do
  uv run python -m src pdf_run_multibook_retrieval \
    <library_root> <golden_v2_draft> "$retriever" <out>/run-$retriever.json
done
# <out>/runs.json lists the three files with their exact SHA256 values.
uv run python -m src pdf_pool_multibook_judgments \
  <golden_v2_draft> <library_root> <out>/runs.json <out>/pool --depth=10
```

| Run | Wall time | Peak memory |
| --- | ---: | ---: |
| BM25 | 38 s | 0.8 GB |
| MiniLM | 81 s | 1.2 GB |
| Hybrid | 80 s | 1.2 GB |

No ranking was empty. The pool contains 1019 query-record candidates and
1260 blank evidence-unit tasks.

### Judgment readiness

| Item | Count |
| --- | ---: |
| Pool tasks | 1260 |
| Tasks required before any metric can be published | 1233 |
| of which original clear questions | 856 |
| of which ambiguous originals (diagnostic only) | 217 |
| of which clarified interpretations | 160 |
| Anchor-only tasks outside the evaluated scope | 27 |
| Distinct query-record excerpts to read | 821 |
| Required tasks per query variant (23 variants with tasks) | 23–153, median 40 |
| Required tasks if only one engine were scored | BM25 635, MiniLM 696, Hybrid 664 |

Existing human judgments:

- The anchor-based pilot holds 29 human labels. 21 of their
  (question, unit, record) bindings also appear in the new pool, 19 of them
  inside the evaluated scope.
- There is no supported, provenance-preserving workflow that moves pilot
  labels into retrieval-pool worksheets, so none were reused. The pilot
  artifacts are unchanged.
- No retriever candidate in the new pool is judged yet.

Consequences:

- BER@5, BER@10 and every other B0 metric are **insufficient judgments** for
  all three engines until the 1233 required tasks are labelled.
- The evaluator scores only all three runs together; neither a single engine
  nor a subset of questions can be scored with the current contracts.
- Labelling time has not been measured. At the unmeasured pilot planning
  assumption of about 1.4 minutes per row, 1233 rows would take roughly
  28 hours; reading 821 distinct excerpts may make it faster.

### Results: B0 metrics

| Metric (`headline_complete:original`) | BM25 | MiniLM | Hybrid |
| --- | --- | --- | --- |
| BER@5 ranked / with context | insufficient judgments | insufficient judgments | insufficient judgments |
| BER@10 ranked / with context | insufficient judgments | insufficient judgments | insufficient judgments |
| Sufficient-set success @5 / @10 | insufficient judgments | insufficient judgments | insufficient judgments |
| Localization coverage @5 / @10 | insufficient judgments | insufficient judgments | insufficient judgments |
| MRR@10 localization / supply | insufficient judgments | insufficient judgments | insufficient judgments |

`headline_complete:clarified`, category and outcome groups: insufficient
judgments.

### Pre-judgment diagnostic: anchor overlap

For each eligible evidence unit, an anchor hit means that at least one record
overlapping one of the unit's verified locator anchors was ranked within the
cutoff. Units can also be supplied by other records, and an anchor record does
not necessarily supply the evidence, so these counts are neither a lower nor
an upper bound on BER.

Complete original questions (12 questions, 20 units):

| Count | BM25 | MiniLM | Hybrid |
| --- | ---: | ---: | ---: |
| Units with an anchor record in top-5 | 12 | 10 | 12 |
| Units with an anchor record in top-10 | 14 | 14 | 15 |
| Units with an anchor record in top-10 or its direct context | 14 | 14 | 15 |
| Questions with a whole sufficient set anchor-covered at 10 with context | 7 | 8 | 9 |

Clarified complete interpretations (3 units): top-10 anchor hits 2, 3 and 3.
Clarified partial interpretations (2 units): 2, 1 and 2. Extraction-limited
original questions (2 units): with context at 10, 1, 1 and 2.

Across all 29 eligible (query variant, unit) pairs:

| Anchor rank band, BM25 / MiniLM / Hybrid | Units |
| --- | ---: |
| top-5 / top-5 / top-5 | 10 |
| beyond top-10 for all three | 3 |
| top-5 / beyond 10 / beyond 10 or 10 | 4 |
| other mixed patterns | 12 |

### Preliminary failure analysis

These observations come from ranks and anchor provenance and await human
judgments:

- **Absent from top-5, present in top-10:** for complete originals, 2–4
  units per engine have anchor records only between ranks 6 and 10.
- **No anchor record in any engine's top-10:** 3 text units of complete
  original questions. Whether other records supply them is unknown.
- **Context:** direct neighbors add an anchor hit only rarely: one complete
  original unit at cutoff 5 for BM25 and for MiniLM, and one
  extraction-limited unit at cutoff 10 for MiniLM and for Hybrid.
- **Engines fall in different rank bands for 16 of 29 units.** In 3 units BM25 ranks an anchor
  record in its top-5 while MiniLM and Hybrid rank it beyond 10: equal-weight
  RRF does not always preserve a strong lexical hit.
- **MiniLM truncation:** 1346 of 2043 eligible records (66%) exceed the
  256-word-piece `max_length`, so most semantic embeddings ignore the end of
  their record.
- **Localization without supply, semantic similarity without sufficient
  evidence, extraction limitations:** not measurable without human
  `required` and `supplies_evidence` labels.

### BM25 tuning decision

Not justified now. There is no judged baseline, and the proxy shows no
BM25-specific weakness: BM25 is level with Hybrid at 5 and differs from the
other engines by at most one unit at 10. The visible issues concern fusion
weights and semantic truncation, not `k1` or `b`. Any parameter change would
also create new unjudged candidates. BM25 stays at `k1 = 1.4`, `b = 0.65`.

### Decision and next actions

- The baseline runs and the review worksheet exist; the judged baseline does
  not. B1 is stopped at the human-review boundary.
- Next: human labelling of the 1233 required worksheet rows, or a decision to
  reduce that scope first (see the open questions below).
- After labelling: `pdf_collect_multibook_judgments` and
  `pdf_evaluate_multibook_retrieval`, then fill the results tables above.

Open questions:

1. Should the 217 diagnostic-only tasks for ambiguous original questions be
   required before scoring? Making them optional needs an evaluator change.
2. Should the 19 matching pilot labels be re-entered by the reviewer, or
   should a reviewed carry-over tool be built?
3. What `BER@5` / sufficient-set success level is enough for LLM evidence
   delivery?

## B1-2: Qwen3-4B grounded-answer smoke test (16 October 2026)

### Purpose

A small practical check of whether the existing Hybrid retrieval gives the
local Qwen3-4B model enough evidence to answer real questions with grounded
citations. This is not a retrieval evaluation and computes no BER.

### Existing capabilities

- `pdf_run_answerability` only classifies stored single-book retrieval runs as
  `ANSWERABLE` or `NOT_FOUND` through an already-running loopback server; it
  does not read multi-book runs and does not generate answers.
- The general `answer` command generates text for the text/code index, not
  for PDFs.
- No grounded PDF answer-generation workflow exists. The smoke test therefore
  used a one-off script outside the repository, without changing production
  code, and the same operator-run llama.cpp server as the earlier
  answerability experiments.

### Configuration

| Setting | Value |
| --- | --- |
| Model | Qwen3-4B Q4_K_M GGUF, SHA256 `7485fe6f…` |
| Runtime | llama.cpp `llama-server` b11574, outside FREDL, loopback only |
| Server arguments | `-ngl 99 -c 16384 --reasoning off --reasoning-budget 0` |
| Generation | temperature 0, top-p 1, seed 0, max tokens 700, thinking disabled |
| Retrieval | B1-1 Hybrid run: ranked top-5, then direct context at cutoff 5, deduplicated |
| Passages per question | 9–13 (5 ranked plus 4–8 context records), 3.1k–5.0k prompt tokens |
| Passage labels | `[S1]`… with book file and physical page range; original record text |
| Not given to the model | evidence anchors, expected outcomes, expected behavior, reference notes, labels |
| Latency | 81–162 s per question; GPU offload was not confirmed at the time. [Investigation 1](inference-performance-diagnostics.md#finding-the-gpu-was-not-used) later showed that these runs, and B1-3, executed entirely on the CPU |

System prompt: answer only from the numbered passages; cite every claim with
passage identifiers that appear in the prompt; if evidence is insufficient,
do not guess and say what is missing; if the question has several materially
different meanings, ask which one is meant. The reply must begin with
`DECISION: ANSWER`, `DECISION: INSUFFICIENT_EVIDENCE` or
`DECISION: CLARIFICATION_NEEDED`.

### Sample

Six Golden Set questions, unchanged, selected by anchor ranks from B1-1:

| Question | Category | Expected | Reason for selection |
| --- | --- | --- | --- |
| Q1 | single-book | complete, answer | all engines rank the anchor first |
| Q2 | complementary multi-book | complete, answer | one unit: BM25 rank 1, MiniLM 41, Hybrid 6 |
| Q3 | single-book | complete, answer | Hybrid best: BM25 30, MiniLM 5, Hybrid 2 |
| Q4 | independent multi-book | complete, answer | no anchor record in any engine's top-10; best rank 11 (BM25) |
| Q5 | ambiguous terminology | ambiguous, clarification | three interpretations; Hybrid ranks two of their anchors 1 and 3 |
| Q6 | hard negative | unsupported, answer | near-topic question without supporting evidence |

### Observed outcomes

Facts are mechanical checks of the reply and passages. "Appears supported"
means that the key claim terms occur in the cited passages; it is an
assistant observation pending human verification, not a relevance judgment.

| Question | Decision | Citations | Cited IDs in context | Observation |
| --- | --- | --- | --- | --- |
| Q1 | answer | 1 | yes | Appears supported by the cited passage. |
| Q2 | answer | 0 | — | The framework the answer describes appears in none of the 11 passages; the answer relies on model knowledge. Unsupported. |
| Q3 | answer | 1 | yes | Appears supported by the cited passage. |
| Q4 | answer | 2, in `(S1)` form | yes | Answers from a related topic; the expected evidence is absent and the model did not flag missing evidence. Some phrasing is not in the passages. |
| Q5 | answer | 2 | yes | Chose one interpretation instead of asking; passages for a second interpretation were present. The chosen answer appears supported. |
| Q6 | insufficient evidence | 0 | — | Correct abstention, but without stating what is missing. |

Summary: 5 answers, 1 abstention, 0 clarification requests. Two answers
appear grounded, one partly grounded, one grounded but for a guessed meaning,
and one unsupported.

### Preliminary failure analysis

- **Missing multi-book evidence leads to unsupported answers.** In Q2 the
  second book's unit was ranked sixth by Hybrid, just outside the top-5, and
  the model filled the gap from prior knowledge without citations instead of
  abstaining.
- **Partial evidence is not flagged.** In Q4 the model answered from nearby
  material rather than reporting that the requested evidence was missing.
- **No clarification behavior.** Despite the instruction and passages for two
  meanings, Q5 was answered for one meaning. This matches PD-005: clarification
  needs a separate component, not only a prompt rule.
- **Citation format and refusal detail are inconsistent:** parenthesized
  identifiers in Q4 and a bare refusal in Q6.
- When the evidence is ranked first (Q1, Q3), the model gives short, cited
  answers.

### Limitations

Six questions, one engine, one prompt and one run at temperature 0. Support
was checked by term overlap and reading, not by human judgment of every
claim. The answers' correctness against the books is not verified. Results
do not generalize to the Golden Set and are not comparable with the
historical binary answerability metrics.

### Recommendation

The existing pipeline gives Qwen3-4B usable evidence for questions whose
evidence ranks near the top, but not reliably for multi-book questions, and
the model does not abstain or ask for clarification dependably. Before tuning
retrieval, a human should verify the six observations. The next measurement
should be the judged B1 baseline, or a targeted check of Hybrid top-10 versus
top-5 for multi-book questions, since Q2's missing unit was at rank 6.
Detailed per-question records are kept only in the private evaluation
directory.

## B1-3: Qwen3-4B with Hybrid top-5 versus top-10 (16 October 2026)

### Configuration

Identical to B1-2 (same six questions, model, server arguments, system
prompt and generation settings) except that the passages are Hybrid ranked
top-10 followed by direct context at cutoff 10, deduplicated. The one-off
script differs from B1-2 only in the cutoff and the private output folder.

| Question | Passages top-5 → top-10 | Prompt tokens top-5 → top-10 | Latency top-5 → top-10 |
| --- | ---: | ---: | ---: |
| Q1 | 9 → 15 | 3134 → 5052 | 81 → 161 s |
| Q2 | 11 → 18 | 3664 → 5709 | 162 → 248 s |
| Q3 | 9 → 22 | 3191 → 7657 | 93 → 242 s |
| Q4 | 11 → 22 | 3849 → 7427 | 135 → 283 s |
| Q5 | 11 → 25 | 4000 → 8538 | 113 → 331 s |
| Q6 | 13 → 22 | 4950 → 8027 | 131 → 236 s |

### Evidence reaching the model

Anchor presence is a provenance fact; whether the text supplies the evidence
still needs human judgment.

- **Q2:** the missing unit's locators span two records. At top-5 neither
  reached the model. At top-10 the rank-6 record is ranked and the other
  record (Hybrid rank 41) arrives as its direct context, so every anchored
  passage of the unit is supplied. The framework terms the B1-2 answer used
  without support now occur in two supplied passages.
- **Q4:** the expected units' anchor records are absent at both cutoffs.
- **Q5:** anchors for two of three interpretations are present at both
  cutoffs; the third is absent at both.
- Q1, Q3 and Q6: unchanged presence (Q6 has no evidence units).

### Observed behavior

| Question | Top-5 (B1-2) | Top-10 (B1-3) | Change |
| --- | --- | --- | --- |
| Q1 | answer, 1 citation, appears supported | same answer plus one uncited restating sentence | none |
| Q2 | answer, 0 citations, unsupported | answer, 0 citations; its framework claims now have matching passages, but illustrative examples are not checked | evidence improved; citation rule still violated |
| Q3 | answer, 1 citation, appears supported | answer, 1 citation, slightly less specific | none material |
| Q4 | answer from related topic, `(S1)` citations | same content, one `[S1]` citation | none material |
| Q5 | one meaning chosen, appears supported | one meaning chosen; one quotation attributed to a passage does not occur in any passage | **regression: invented quotation** |
| Q6 | abstained without detail | abstained without detail | none |

Every cited identifier exists in the supplied context in both runs. No run
asked for clarification.

### Assessment (pending human verification)

- Top-10 fixed the evidence gap it was meant to fix: Q2's missing unit now
  reaches the model.
- It did not fix downstream behavior: Q2 is still uncited, Q4 still answers
  without its evidence, Q5 still guesses a meaning and now contains an
  invented quotation, and Q6 is unchanged.
- Context grew by 1.6–2.4 times in prompt tokens, and latency roughly
  doubled on this runtime, which was later found to be CPU-only. GPU timings
  for the same prompts are in
  [Investigation 1B](inference-performance-diagnostics.md#investigation-1b-gpu-baseline-for-all-twelve-b1-prompts-16-october-2026).

The six questions do not justify switching to a larger retrieval window yet.
The dominant problems are now model behavior: missing citations, answering
without evidence, no clarification and an invented quotation. A human should
verify the twelve responses before any decision; retrieval-window changes
would need the judged baseline to show a measurable evidence gain.
