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

> **Superseded on 1 November 2026 by [PD-009](product-decisions.md#pd-009-evaluate-without-mandatory-manual-annotation):** labelling the 1233 worksheet
> rows is no longer a required next step, and the open questions below no
> longer block further work. The runs, the pool and the decision as recorded
> on 16 October 2026 are preserved unchanged.

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

## M2-A: Phi-4-mini compatibility smoke (17 October 2026)

**Status: executed once; technical compatibility gate passed, pending review.**
Authorization covered the pinned download, integrity check, twelve private B1
payloads, non-generating preflight and one uncached Phi request. No Qwen run,
M2-B comparison, retrieval change, prompt revision or production change occurred.
The [approved protocol](retrieval-inference-roadmap.md#m2-a-compatibility-and-context-fit-protocol-17-october-2026)
provides the preparation and stop conditions. This result applies to this
artifact, runtime, configuration and machine only.

### Artifact and input integrity

Downloaded only `microsoft_Phi-4-mini-instruct-Q4_K_M.gguf` from
`bartowski/microsoft_Phi-4-mini-instruct-GGUF`, pinned to
`915429cb42fe8eba71bd1d3117a7d63070892268`. Verified local size:
**2,491,874,688 bytes**; verified SHA256:
`01999f17c39cc3074afae5e9c539bc82d45f2dd7faa3917c66cbef76fce8c0c2`.
Both match the approved artifact. GGUF V3, file type Q4_K_M, architecture
`phi3`; the loader reports about 3.84B parameters. The model and all scripts,
request/response files, metadata, logs and answers remain outside Git.

The B1 records do not contain complete request bodies. Preparation recovered
passage text from existing extraction snapshots using an inert opcode parser,
without executing pickle globals or constructors, and reused the saved source
order, labels, book/page metadata, question and system instructions. All twelve
original body hashes matched before exporting new private messages files.
Frozen-file hashes were checked before tokenization. Original input-file
hashes were rechecked after the run and remained unchanged. No re-extraction,
reranking, evidence shortening, normalization or source relabelling occurred.

### Runtime, template and effective capacity

- Existing llama.cpp b11574, commit `8bb3cf0d2`, CUDA 12.8 backend; existing
  companion CUDA libraries supplied through the process library path. Binary
  hash and effective settings were saved privately. No dependencies installed.
- Quadro RTX 3000: 6,144 MiB physical; runtime device enumeration reported
  5,738 MiB visible and 5,648 MiB free before loading.
- Launch: `-ngl 99 --fit off -c 16384 -np 4 --kv-unified -ctk f16 -ctv f16
  -t 6 -tb 6 -b 2048 -ub 512 -fa auto --jinja --no-context-shift
  --no-warmup -lv 4`, loopback only. No warmup or automatic fit adjustment.
- Logs: `n_ctx = n_ctx_seq = n_ctx_slot = 16384`, `n_seq_max = 4`, four slots
  and unified KV. `/props` and each `/slots` entry independently reported
  16,384. The one request actually processed 8,267 input tokens, exceeding
  4,096, without truncation. This confirms capacity is not divided into four
  4k windows in this measured configuration; it does not test the 16k boundary
  or concurrent capacity.
- Tokenizer: BPE `gpt-4o` pre-tokenizer, 200,064 vocabulary entries; automatic
  BOS/EOS insertion disabled. Native role tokens: system 200022, user 200021,
  assistant 200019, end-of-turn 200020. Runtime EOG set: 199999
  (`<|endoftext|>`) and 200020 (`<|end|>`).
- `/apply-template` produced exactly the native system/content/end,
  user/content/end, assistant-prefix sequence for every input. Both contents
  occurred once, without relocation. Token-piece checks found the expected
  five role/turn markers and no extra BOS; detokenization exactly recovered
  every rendered prompt. Embedded template equalled `/props.chat_template`.
- The generation used `/completion` with those exact integer token IDs:
  no raw untemplated text, second template application or added BOS. The
  installed `/apply-template` handler uses the chat request formatter without
  inference, so the native template route was inspected before execution.
  No chat-completion generation request was sent.

The GGUF has 32 layers, 8 KV heads, 96 rotary dimensions of a 128-dimensional
head (0.75), and both 48-value short/long RoPE factor tensors. The generic
load log labels RoPE scaling `linear`; the Phi implementation additionally
uses these factor tensors according to context length. No metadata overrides
were applied. The loader warns that Phi sliding-window attention is disabled.
The artifact declares a 262,144-token sliding window, exceeding this entire
16,384-token context, so no in-window cutoff is lost here. This does not
validate all long-context behavior or certify upstream numerical equivalence.

### Measured token counts and context fit

Q1–Q6 retain the public B1 aliases; private case IDs are not included. Qwen
counts are historical measurements; Phi counts were measured with this GGUF
and its effective native template. The 700-token reserve is calculated.

| Question alias | Qwen top-5 input | Phi top-5 input | Phi top-5 + 700 | Qwen top-10 input | Phi top-10 input | Phi top-10 + 700 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1 | 3,134 | 3,077 | 3,777 | 5,052 | 4,965 | 5,665 |
| Q2 | 3,664 | 3,552 | 4,252 | 5,709 | 5,559 | 6,259 |
| Q3 | 3,191 | 3,089 | 3,789 | 7,657 | 7,392 | 8,092 |
| Q4 | 3,849 | 3,662 | 4,362 | 7,427 | 7,096 | 7,796 |
| Q5 | 4,000 | 3,819 | 4,519 | 8,538 | **8,267** | **8,967** |
| Q6 | 4,950 | 4,772 | 5,472 | 8,027 | 7,723 | 8,423 |

All twelve unchanged Phi inputs fit the measured 16,384-token single-request
capacity with the full reserve. The largest leaves 7,417 positions beyond
input plus reserve. All would also fit 12k arithmetically, but 12k was not
configured or tested. Phi Q5/Q6 top-10 would exceed 8k with the reserve;
changing evidence or context to address that was neither needed nor attempted.
Twelve template/tokenization/roundtrip checks generated no answers.

### GPU allocation and resource measurements

| Item | Measured |
| --- | ---: |
| GPU layer offload | 33/33, including output layer |
| CUDA0 model buffer | 2,368.57 MiB |
| CPU-mapped model buffer | 480.81 MiB |
| CUDA0 f16 KV buffer | 2,048.00 MiB: 1,024 K + 1,024 V |
| KV layout | 16,384 cells, 32 layers, shared across four sequences |
| CUDA0 compute buffer | 88.01 MiB |
| CUDA-host compute / output buffers | 40.02 / 3.05 MiB |
| Sampled GPU use after loading | about 4,609 MiB |
| Maximum sampled GPU use | 4,619 MiB |
| Minimum sampled free VRAM | **1,121 MiB**, above the 256 MiB gate |

Sampling ran every 200 ms from before loading through completion and review;
702 samples were summarized before shutdown. It can miss brief peaks. The
CPU-mapped file buffer and CUDA-host buffers are explicitly recorded; they do
not indicate a switch to CPU-only execution. Layer offload and KV placement
remained on CUDA0, with no CUDA/OOM error or fallback reported. No claim is
made that every host operation runs on the GPU. After shutdown, GPU use
returned to 4 MiB and the test port was free.

### Exactly one generation request

Selected the longest measured Phi input, Q5 top-10. Sent one non-streaming
request, with no automatic retry: temperature 0, top-p 1, seed 0, output limit
700, `cache_prompt: false`, `ignore_eos: false`, no custom stop strings and
raw output token IDs enabled. Effective settings confirmed those values;
additional recorded defaults included top-k 40, min-p about 0.05, repeat
penalty 1, zero presence/frequency penalties and no active logit biases.
The native endpoint reported content-only chat format and no extracted
reasoning; its generic `reasoning_format` field was `deepseek`, which does
not insert a thinking prompt into the already rendered token-ID input.
No Qwen-specific controls or prompt alterations were applied.

| Measurement | Result |
| --- | --- |
| HTTP | 200 |
| Prompt tokens supplied / evaluated | 8,267 / 8,267 |
| Prefix cache reuse | `timings.cache_n = 0` |
| Generated tokens | 73, including terminal token ID 200020 |
| Completion | `stop_type = eos`, terminal `<|end|>`, no custom stop word |
| Truncation | false; server release log also reports `truncated = 0` |
| Prefill | 13.176 s, 627.4 tokens/s |
| Decode | 1.304 s, 55.2 tokens/s |
| Client wall time | 14.493 s |
| First-token latency | not measured: non-streaming request |

The response was nonempty, ended with punctuation, contained no emitted role
markers and terminated naturally below the output limit. These are technical
completion checks, not human validation of its meaning or grounding. The
private answer is retained for review and is not reproduced here. Citation
correctness was not an M2-A acceptance criterion and no quality score was
assigned. `tokens_cached = 8339` describes retained KV tokens after completion,
not input reuse: all 8,267 prompt tokens were evaluated and `cache_n = 0`.

One expected protocol field needed a schema correction: this binary's native
response has a flat `generation_settings` object without `n_ctx`. Capacity was
verified in load logs, `/props` and `/slots`, then confirmed by actual prompt
processing; no request configuration changed. Do not use the preparation
snippet's `generation_settings.n_ctx` assertion with this response schema.

### Interpretation and next boundary

M2-A passed the approved technical gates for this one run. M2-B appears
feasible at the unchanged 16k/four-slot/f16-KV configuration with all twelve
frozen evidence payloads. This is feasibility evidence, not authorization to
run it, proof of every future output completing within 700 tokens, or a reason
to replace Qwen. No retry, second generation or Qwen comparison was performed;
the test server and monitor were stopped.

Before M2-B, approve the common endpoint, instrumentation and human review
protocol and run a fresh Qwen baseline under the same comparison conditions.
The native-token smoke is not directly comparable to historical chat-endpoint
B1 timings. Baseline instructions, retrieval and original B1/1B results remain
unchanged. Grounding, citations, abstention, clarification and usefulness still
need the separately approved controlled comparison and review.

## M2-B: controlled comparison preparation (17 October 2026)

**Status: planned, not executed.** Preparation only: no model loading,
tokenization, generation, download or dependency installation. Planned scope:
24 fresh responses from Qwen3-4B Q4_K_M and Phi-4-mini-instruct Q4_K_M, using
the same twelve B1 question/context payloads. The full
[M2-B protocol](retrieval-inference-roadmap.md#m2-b-controlled-comparison-protocol-17-october-2026)
defines controls, order, measurements, rubric and stop conditions. M1 and M2-A
remain separate completed research records; their results are not overwritten.

### Observation — verified preparation, no answer judgments

- Both existing local GGUF hashes were reverified by file reads, without
  loading either model. Qwen SHA256:
  `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`;
  Phi SHA256:
  `01999f17c39cc3074afae5e9c539bc82d45f2dd7faa3917c66cbef76fce8c0c2`.
- All twelve frozen messages and reconstructed legacy body hashes matched
  their B1/M2-A records. A new private input snapshot contains exact question
  text, system instructions, ordered evidence text and source/record mappings,
  per-component hashes, original generation settings and a package manifest.
  No passage, order, retrieval, ranking or chunk changed.
- Static GGUF reads pinned each embedded native chat template and its hash,
  tokenizer metadata, and model-specific controls. The planned runtime binary
  hash matches M2-A. These file checks are not fresh runtime validation.
- Six private draft behavior records include separate expectations for each
  context: twelve delivered-evidence targets, with source/text offsets and
  rationale. They are pending Maria's approval, not new human judgments or
  changes to the Golden Set. One previous preliminary support assessment is
  flagged for review because non-anchor passages may support an answer.
  Neither absence nor presence of an anchor alone settles sufficiency.
- A private rubric, 24 empty individual review cards, twelve empty pair-review
  forms, a 24-row measurement template, execution schedule, opaque review IDs
  and separate operator key were prepared. All ratings/measurements are null;
  no response or preference was scored. Approval fields remain unfilled and
  generation is explicitly unauthorized in this preparation snapshot.

### Interpretation — what this preparation permits us to propose

The same evidence can be compared without redesigning retrieval. Expected
actions must follow the supplied context: a full answer, bounded partial
answer, explicit insufficient-evidence refusal or clarification can each be
appropriate. Case-specific choices stay private and require review before
any generation. Existing Golden Set source support describes the wider corpus;
it cannot be substituted for support in the delivered top-5/top-10 contexts.

M2-A established Phi's fit at 16k with four slots and f16 unified KV. B1/1B
established the historical Qwen configuration. M2-B still needs fresh runtime
preflight for both models, especially Qwen's native-template rendering and
token counts under the common endpoint. Underlying message content is fixed;
native wrappers and tokenization differ. Exact historical rendered Qwen bytes
were not saved, so no byte-identical-wrapper claim is made.

The proposed order is Qwen set A, Phi set A, Phi set B, Qwen set B, six
requests per block; each set contains three top-5 and three top-10 inputs.
Only one server is resident, all requests are sequential and uncached, and
there are no extra warmup generations. This reverses model order across
matched input sets but cannot eliminate session or thermal effects.

Review should separate deterministic citation/quotation diagnostics from
human claim support, correctness and usefulness. One reviewer can complete
four sessions of six individual responses, then twelve blinded same-context
pairs. Preserve uncertain and not-applicable entries, rationale, confidence,
prior familiarity and any first/second-pass disagreement. Unreviewed nulls
are not scores. Treat the twelve contexts as related observations from six
questions, not independent evidence of statistically significant superiority.
No external evaluator, provider call or automatic self-assessment is planned.

### Decision — pending Maria's review and execution approval

Approve or revise the private delivered-context expectations and rubric;
confirm the common native-token `/completion` route, unchanged 700-token
budget, explicit sampler controls, 16k/four-slot/f16-KV configuration, execution
order and human-review procedure. The existing Qwen baseline must be rerun
fresh; no M2-A or B1 answer counts toward the 24 planned responses.

Technical validity is a separate gate from grounding, usefulness and resource
reporting. Invalid configuration or incomplete generation stops the run and
preserves partial artifacts; no automatic retries, shorter evidence or rescue
prompts. A citation/grounding defect remains a reviewable output, not a hidden
technical exclusion. There is no universal product latency threshold, overall
quality score or model-adoption rule.

### Implementation — none

No production changes, execution, new retrieval run or baseline-prompt edit.
Only private preparation artifacts and research documentation were created.
M2-B remains planned until explicitly authorized. Later observations and human
judgments must be recorded separately from this preparation, linked to
PD-006 (verified GPU use), PD-007 (sequential workflow/stage measurements) and
PD-008 (replaceable models with equivalent local evidence). A result will not
automatically create a product decision or implementation.

## M2-B: controlled comparison execution (17 October 2026)

**Status: executed, 24/24 requests technically complete; awaiting human review.**

> **Superseded on 1 November 2026 by [PD-009](product-decisions.md#pd-009-evaluate-without-mandatory-manual-annotation):** the human review described below
> is optional and no longer a prerequisite for further work. The execution
> record and its observations are unchanged.

The [approved comparison protocol](retrieval-inference-roadmap.md#m2-b-controlled-comparison-protocol-17-october-2026)
was executed with the two existing pinned GGUFs and twelve fixed B1 payloads.
No winning model, semantic grounding score or usefulness judgment is assigned.
Model labels in the aggregate table are anonymous; their key and response
mapping remain private and outside the reviewer package.

### Observation — input and preflight gates

- Both GGUF SHA256 values, the runtime binary hash, all twelve message/body
  hashes and the private preparation package were verified before loading.
  Original B1/M2-A records and the previous expectation draft remain unchanged.
- The approved delivered-context expectations were saved as a separate private
  version before execution, preserving the draft and recording the approved
  source-based clarification. This approves assessment targets, not historical
  model answers or new answer judgments.
- Both models passed non-generating preflight before the first answer. A
  preflight-only load was followed by the three planned execution loads; no
  warmup generation or extra smoke request was sent. Each later execution
  load repeated the template/context/offload checks.
- Native template text matched the pinned GGUF. The Qwen thinking-off wrapper
  was checked explicitly and all twelve Qwen input counts matched B1. Phi
  token IDs matched M2-A. All rendered prompts contained the exact unchanged
  messages and roundtripped through their tokenizers without change.
- Every slot reported 16,384 effective context positions with four slots and
  unified f16 KV, corroborated by load logs and properties. All twelve inputs
  plus the unchanged 700-token output reserve fitted for each model. No
  evidence truncation, reordering, context shift or configuration adjustment.
- Full expected model-layer and KV GPU offload was verified on each load.
  The 256 MiB sampled-free-VRAM gate passed throughout every request. Host
  mapped/bookkeeping buffers were recorded separately from layer placement.

### Observation — execution and technical validity

Executed four sequential batches of six in the approved balanced set order,
with one model resident at a time. Every question/context was generated once
per model; all 24 answers were fresh. The earlier M2-A answer and historical
B1 answers were not reused. There were exactly **24 generation attempts and
24 technical completions**, no retries and no technical stop conditions.

Both models used `/completion` with integer token IDs from the native
`/apply-template` → `/tokenize` route. No raw untemplated prompt or second
rendering was used. Approved settings were preserved: output budget 700,
temperature 0, top-p 1, seed 0, explicit common sampler controls, no custom
stop strings, `ignore_eos: false`, non-streaming, and prompt caching disabled.
Response-level effective settings and sampler chains were checked; inactive
common controls did not drift. Model-specific thinking controls remained as
specified in the protocol.

All responses returned HTTP 200, evaluated the complete supplied prompt,
reported `timings.cache_n = 0` and `truncated = false`, and terminated on a
native EOG token within the output budget. No timeout, OOM, CUDA failure,
unexpected CPU/KV fallback, capacity failure or automatic fit change occurred.
Natural EOG is technical termination evidence, not proof of semantic
completeness, correct refusal or useful content. First-token latency was not
measured because requests were non-streaming.

| Anonymous label | Requests | Client wall median (range), s | Prefill median, s | Decode median, s | Peak sampled GPU use, MiB | Minimum sampled free VRAM, MiB |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Model A | 12 | 5.534 (2.883–8.629) | 3.186 | 1.749 | 4,889 | 851 |
| Model B | 12 | 4.449 (2.494–9.460) | 2.599 | 1.359 | 4,621 | 1,119 |

The table is descriptive, **not a speed ranking**: tokenizers, prompt lengths,
output lengths and response actions differ. Per-request lengths and phase
throughput are retained privately for matched analysis. Phase medians do not
sum to the wall-time median. Model loading and preflight/tokenization are
outside request wall time. Requests were not uniformly cold: one execution
session covered twelve requests and two sessions covered six each, as planned.
First-request/CUDA-graph and thermal/order effects were not isolated. The
200 ms sampler can miss brief peaks. The known Phi sliding-window warning and
generic RoPE log terminology remain the M2-A limitations; no claim about the
advertised maximum context or upstream numerical equivalence follows.

Separate deterministic formatting observations flagged 12 responses whose
first line did not exactly match the required decision-line form. Two answers
contained only that full decision line and two contained only a bare decision
label. Five non-punctuation endings were decision-label endings; all ended
naturally rather than through a token limit or transport interruption. These
are preserved output-contract observations, not automatic grounding or
usefulness scores, and none triggered a hidden rerun. No emitted role markers
or candidate-name keyword cues were flagged; this heuristic does not prove
that identity is unguessable or content is semantically complete.

### Observation — private reviewer export

Raw response bytes are preserved unchanged. A separate private reviewer
package contains the exact questions/instructions/evidence, approved expected
actions, unmodified answer text, 24 blank individual review cards, twelve
matched pair forms and review guidance. Fresh opaque review IDs are mapped
only in operator records. All answer bytes were checked against their raw
response content; every human rating remains null/unreviewed.

The package excludes model/runtime identity metadata, execution order,
performance measurements, operator-only records, mapping keys and the full
experiment manifest. It contains no model-to-label key. Blinding remains
partial because style, source references or prior familiarity can suggest
identity; answers were not rewritten to hide such cues. Performance summaries
are kept outside the reviewer packet. No semantic answer judgments or model
preferences were generated automatically.

### Interpretation — what the execution establishes

The approved comparison is technically reproducible on the measured machine
with fixed local evidence and the recorded runtime settings. It supplies
matched outputs for review, not evidence that either model grounds claims
better. Six questions with related top-5/top-10 contexts remain a small,
purposefully selected sample; no statistical superiority or permanent model
choice follows from completion, timing or formatting checks.

### Decision — pending human evaluation

> **Superseded on 1 November 2026:** the review below is optional; see the
> note under this section's status.

Maria reviews individual responses before matched pairs, preserving uncertainty,
claim/source spans, first/revised judgments and suspected identity leaks. Keep
model mappings sealed until judgments are finalized; reveal performance after
content judgments. Compare models at the same question/cutoff, and compare
contexts within each model using record/text identity. Historical judgments
are not rewritten to match these new outputs. Any subsequent adoption or
implementation decision requires separate review.

### Implementation — no production change

Only private execution/reviewer artifacts and research documentation changed.
The server and sampler were stopped; the test port was free and GPU use
returned to its idle level. No dependencies, downloads, new retrieval runs,
prompt edits, production changes, commits or publication steps occurred.
This records observations under PD-006 (verified GPU use), PD-007 (sequential
requests and separate stage/resource reporting), and PD-008 (replaceable models
with equivalent private evidence), without creating a new product decision.
