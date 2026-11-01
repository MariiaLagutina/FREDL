# Retrieval and Inference Roadmap

> **Status: research roadmap; no implementation authorized.** Investigation 1
> is complete. The LLM comparison direction and M1 candidate research were
> approved on 17 October 2026; [M1 findings](#m1-findings-17-october-2026)
> provisionally identify Phi as the first alternative. M2-A has now passed its
> one-request compatibility gate; its result awaits review
> ([results](pdf-retrieval-experiments.md#m2-a-phi-4-mini-compatibility-smoke-17-october-2026)).
> M2-B has completed 24 requests and awaits human review
> ([execution record](pdf-retrieval-experiments.md#m2-b-controlled-comparison-execution-17-october-2026)).
> Investigations 2–6 and further M3–M4 analysis/decisions are not completed.
> [Investigation 3](#3-embedding-model-alternatives) is the next planned
> technical work block.
> Acceptance criteria are agreed before execution; the result may
> be to keep the current solution.
>
> Since 1 November 2026, human review of M2-B and judged retrieval subsets are
> optional, not prerequisites ([PD-009](product-decisions.md#pd-009-evaluate-without-mandatory-manual-annotation)).

This roadmap belongs to the current PDF evaluation work described in the
[README](../README.md#1-prepare-the-embedding-model-comparison). It does not change
the [product direction](product-direction.md), the
[product decisions](product-decisions.md) or the implemented boundary in the
[technical foundation](technical-foundation.md).

## Why

FREDL helps a reader work with technical literature and find reliable,
traceable information. Five qualities guide retrieval and inference work.

| Quality | Status |
| --- | --- |
| Evidence reliability: answers are supported by identifiable source material | Established requirement: source-grounded explanations and book/page provenance ([product direction](product-direction.md)); direct evidence before treating retrieval as an answer ([PD-002](product-decisions.md#pd-002-require-direct-evidence-before-treating-retrieval-as-an-answer)) |
| Privacy: local-first and offline operation | Established requirement: PDFs, passages and evaluation cases stay local, and any model integration is explicit about the text it receives ([privacy and model boundary](technical-foundation.md#privacy-and-model-boundary)) |
| Maintainability: retrieval, storage and inference stay separable | Established principle: models remain replaceable ([README](../README.md#product-principles)); separable storage is a design goal of this roadmap |
| Speed: short time to useful information and to a complete answer | Product goal; the earlier 10–20 s and 30 s figures are historical provisional targets only (see [Investigation 1](#1-local-inference-performance)); no acceptance threshold is defined |
| Future extensibility: multi-user use stays possible | Deferred direction, not a current requirement (see [Possible multi-user direction](#possible-multi-user-direction)) |

Speed and evidence reliability are judged together: a faster answer that is not
supported by its sources is not an improvement.

## Established findings from B1

These findings come from the experiment journal,
[PDF Retrieval Experiments](pdf-retrieval-experiments.md), whose sections give
the procedure and limitations, and from
[Inference Performance Diagnostics](inference-performance-diagnostics.md).

Corpus and configuration
([B1-1](pdf-retrieval-experiments.md#b1-multi-book-retrieval-baseline-16-october-2026)):

- the four-book PDF corpus has 2,043 eligible records under the P2 corpus
  policy;
- records are reading-flow chunks of at most 1,600 characters;
- the MiniLM configuration accepts at most 256 word pieces, and 1,346 of the
  2,043 eligible records (about 66%) are longer, so their semantic
  representations are truncated.

Retrieval:

- BM25, MiniLM and equal-weight Hybrid RRF ran over the whole library. Only
  anchor-overlap diagnostics exist: they show whether a record overlapping a
  verified evidence anchor was ranked. They are not human judgments of
  evidence sufficiency, and no judged BER baseline exists yet;
- in three evidence units, BM25 ranked an anchor record in its top-5 while
  MiniLM and Hybrid ranked it beyond 10: equal-weight fusion sometimes ranks
  strong lexical evidence below the cutoff.

Answers, from a six-question Qwen3-4B smoke test
([B1-2](pdf-retrieval-experiments.md#b1-2-qwen3-4b-grounded-answer-smoke-test-16-october-2026),
[B1-3](pdf-retrieval-experiments.md#b1-3-qwen3-4b-with-hybrid-top-5-versus-top-10-16-october-2026)):

- the model omitted citations, answered without sufficient evidence, did not
  ask for clarification of an ambiguous question and, with the larger
  context, invented a quotation;
- raising Hybrid from top-5 to top-10 delivered the missing anchor passages in
  one tested question but did not make the answers reliable;
- the larger context raised prompt size 1.6–2.4 times and roughly doubled
  latency (81–162 s per question at top-5, 161–331 s at top-10);
- Investigation 1 later showed that these runs executed entirely on the CPU:
  the CUDA backend could not load its runtime libraries, so no layer was
  offloaded. With the GPU active, the same twelve prompts took 2.9–9.0 s of
  model time, and the answer-behavior problems above remained.

These are observations from six questions, checked by term overlap and
reading rather than human judgment. They do not show that top-5 or top-10 is
the right cutoff in general, that MiniLM is the wrong model, or which storage
backend FREDL should use.

## Experimental decision process

Every investigation records the same fields. The fields up to the
acceptance criteria are written before it runs; the rest are written after.

| Field | Content |
| --- | --- |
| Experiment ID and research question | A stable identifier, such as B1-3 or Investigation 1A, and the question it answers |
| Related decisions | The `PD-` identifiers it depends on or may revisit |
| Hypothesis | What is expected to change, and why |
| Alternatives considered | The options compared, and options deliberately left out |
| Current baseline | The exact configuration and measurements it is compared against |
| Controlled experiment | What changes and which variables stay fixed |
| Hardware and runtime | Machine, accelerator use and how it was confirmed, inference runtime and build, server arguments |
| Model and quantization | Model, revision or file hash, quantization and model-specific settings |
| Prompt and context configuration | Instructions, passage selection, passage labels and generation settings |
| Evaluation dataset | Which questions, their version and status, and how expected behavior was established |
| Measurements | Quality, evidence, latency and resource metrics |
| Acceptance criteria | Thresholds agreed before execution |
| Observations | Measured results, kept apart from estimates and assistant checks |
| Interpretation and limitations | What the results support, and what they do not |
| Decision | Adopt, Reject or Defer with reasons, or the question left unresolved |
| Impact | Effects on other components, artifacts and judgments |
| Next experiment | Only when the result justifies one; it needs its own approval |

Traceability runs in both directions: an experiment names the decisions it
relies on, and a decision in the [product decision log](product-decisions.md)
cites the experiment sections in its evidence. A decision is recorded there
only after the product owner approves it; an experiment never records one on
its own. Experiments completed before this convention keep their original
structure.

Each result moves through four separate steps, and none implies the next:

1. **Observation:** what was measured or checked, labelled as a deterministic
   check, an assistant check or a human judgment;
2. **Interpretation:** what the observations support, with their limitations;
3. **Decision:** what the product owner approves, recorded as a `PD-` entry,
   or explicitly marked as pending when the evidence is insufficient;
4. **Implementation:** a separately approved change that follows a decision.

Principles:

- change one major variable at a time where feasible;
- keep the current configuration reproducible as a baseline;
- prefer a measured improvement to adopting a framework for its own sake;
- an experiment may conclude that the current solution stays;
- an experiment never implements its result automatically;
- judge performance and evidence reliability together;
- record, for every performance experiment, the actual hardware, runtime
  configuration, model settings and resource utilization, so that it can be
  reproduced;
- never let a product performance target depend on a single development
  machine.

### Evaluation scope

> **Superseded on 1 November 2026 by [PD-009](product-decisions.md#pd-009-evaluate-without-mandatory-manual-annotation):** a judged subset is no longer
> required before comparisons. Comparisons report behavioral diagnostics,
> synthetic known-provenance checks or model judgments with their evidence
> class; none of them establishes validated relevance. The text below is the
> scope as agreed on 16 October 2026.

Retrieval-quality decisions need human-judged evidence. Anchor overlap can
guide where to look but is never human-judged evidence sufficiency and cannot
accept or reject a retrieval change.

The complete judged B1 baseline is not required before preliminary controlled
experiments; it remains unfinished. Early comparative decisions may use a
small, deliberately selected judged subset:

- the subset includes questions where lexical retrieval does well, where
  semantic retrieval does well, where evidence from several sources is
  required and where evidence is absent;
- the subset and its judgment protocol are defined before alternatives are
  compared;
- subset results are reported as preliminary, separately from full B1
  metrics, with their sample size and uncertainty;
- conclusions from a small subset are revisited once broader judgments exist.

Any change to chunking or record identity creates new records. Existing pools
and judgments are bound to record identities and fingerprints, so they are
never silently reused: the affected evidence mappings and judgments are
established again.

## Investigations

Investigation 1 was the first technical investigation. Investigations
2–6 are not a strict sequence: they may run in a different order or in
parallel when they are independent. Reliability checks such as exact
quotation verification, for example, need no Qdrant or other infrastructure
decision. The order remains provisional, later findings may change it, and
every investigation still needs separate approval.

### 1. Local inference performance

- **Hypothesis:** Qwen3-4B latency may depend substantially on runtime
  configuration, hardware use, prompt processing or decoding speed.
- **Status:** closed on 17 October 2026 as a completed diagnostic
  investigation ([Investigations 1, 1A and 1B](inference-performance-diagnostics.md)).
  It is not a CPU-only or end-to-end performance validation. The
  slow B1 answers came from CPU-only execution caused by the launch
  environment; with the CUDA runtime available, all 37 layers were offloaded
  at the original configuration. Not measured: run-to-run variance, per-query
  retrieval latency, time to first useful result, the complete question-answer
  workflow and any CPU-only or consumer-hardware profile. GPU execution did not
  change answer reliability. The launch procedure is outside FREDL and is not a
  supported setup.
- **Current baseline:** the B1-2 and B1-3 latencies above, with the recorded
  llama.cpp server arguments and generation settings.
- **Controlled experiment:** diagnosis first, not optimization. Measure the
  current configuration with the same prompts before changing anything.
- **Measurements**, reported separately:
  - time to first useful result, such as the first verified source passage;
  - time to first generated token, where applicable;
  - time to the complete answer;
  - retrieval latency;
  - prompt processing (prefill) latency and throughput;
  - generation (decode) latency and throughput;
  - CPU and GPU utilization, offloaded layers, and CPU and GPU memory;
  - model loading and request overhead, and how latency grows with context.
- **Reference hardware:** first listed as a Dell Precision 7560 (Intel
  i7-11850H, RTX A3000). The investigation detected a Dell Precision 7550 with
  an Intel Core i7-10850H, 30 GiB usable RAM and an NVIDIA Quadro RTX 3000
  with 6 GB ([hardware and runtime](inference-performance-diagnostics.md#hardware-and-runtime)).
  The reference machine is a measurement environment, not the definition of a
  product target ([PD-006](product-decisions.md#pd-006-prefer-gpu-acceleration-for-experiments-without-making-cuda-a-product-dependency)).
  An ordinary consumer laptop without a dedicated GPU may later become a
  separate performance profile; it is not a current benchmarking
  requirement.
- **Historical provisional targets**, not product requirements or acceptance
  criteria: about 10–20 seconds for an ordinary grounded answer, and
  preferably no more than 30 seconds for a more complex multi-book answer,
  recorded on 16 October 2026. The GPU measurements cover
  model inference only, so nothing yet shows that the complete workflow meets
  them. Universal latency thresholds are not set at this stage
  ([PD-007](product-decisions.md#pd-007-evaluate-performance-as-one-users-sequential-question-workflow)).
- **Acceptance criteria:** not defined; they follow from reliable
  measurements.
- **Impact:** informs the context budget in Investigations 4 and 6.

### 2. Retrieval chunking strategy

- **Hypothesis:** current chunks may be too long for the embedding
  configuration and reduce semantic retrieval quality.
- **Current baseline:** reading-flow chunks of at most 1,600 characters, with
  section-interval context.
- **Controlled experiment:** compare smaller retrieval chunks, parent-child
  chunking, and separate retrieval units from context-delivery units, with the
  embedding model and fusion fixed.
- **Measurements:** evidence retrieval, delivered context size, latency and
  indexing cost; preservation of physical-page provenance, section boundaries
  and stable identifiers.
- **Evaluation:** a preliminary judged subset is acceptable (see
  [Evaluation scope](#evaluation-scope)). New records need new evidence
  mappings and judgments; identity-bound labels are not reused.
- **Acceptance criteria:** not defined. Smaller chunks are not assumed to be
  better: a technical explanation may need its surrounding text.
- **Impact:** new record identities, new pools and new human judgments; possible
  changes to the reader-facing passage and to
  [PD-001](product-decisions.md#pd-001-represent-a-passage-through-its-verified-section-and-exact-pdf-page).

### 3. Embedding model alternatives

- **Status:** next technical work block, *Embedding Comparison Readiness*,
  within the phase *Make a Technical PDF a Trustworthy Source*. Planned; no
  code, model download or run has started.
- **Hypothesis:** another local embedding model may accept longer input or
  offer a better retrieval behavior/cost trade-off on technical text. In B1,
  1346 of 2043 eligible records exceeded MiniLM's 256-word-piece limit.
- **Current baseline:** `sentence-transformers/all-MiniLM-L6-v2` at a pinned
  revision, CPU float32, mean pooling, L2 normalization, `max_length = 256`, no
  query instruction.
- **Candidates:** `BAAI/bge-small-en-v1.5` and `BAAI/bge-base-en-v1.5`, each
  pinned to a revision when approved. Their model-specific settings (pooling,
  query instruction, input limit) are taken from the pinned model card.
  BGE-M3 is a separate later experiment.
- **Readiness work, in dependency order:**
  1. embedding profiles that declare model, revision, pooling, normalization,
     query and passage instructions, input length, dtype and device; the
     MiniLM profile reproduces current behavior;
  2. separate query and passage encoding;
  3. tokenization diagnostics: untruncated length per record and the share of
     truncated records per profile;
  4. the complete profile in run configurations and persisted index
     identities, with existing MiniLM manifests still readable;
  5. persisted PDF embeddings per corpus and profile, loaded without
     re-encoding and rejected on any identity mismatch;
  6. profile selection in the multi-book retrieval runner;
  7. ranking comparison: overlap at 5 and 10, rank-biased overlap, rank
     displacement, records unique to one model and per-book distribution;
  8. paraphrase stability across alternative formulations of one question;
  9. a performance harness: model load, corpus encoding throughput, repeated
     query latency (median and 95th percentile after warm-up), peak RAM and,
     on GPU, peak VRAM;
  10. an optional offline HTML report of the largest ranking differences,
      without required annotation fields.

  Steps 1–6 and 7–10 are planned as two separate code changes.
- **Controlled experiment:** identical corpus, chunking, retrieval corpus
  policy and queries (the draft Golden Set v2 query variants and the curated
  120-question set); semantic-only and hybrid rankings for every profile; BGE
  with and without its query instruction as an ablation. Raw similarity scores
  are not compared across models.
- **Evidence class:** behavioral diagnostics and, where used, synthetic
  known-provenance checks or optional model judgments
  ([PD-009](product-decisions.md#pd-009-evaluate-without-mandatory-manual-annotation)).
  Overlap or stability does not show that a model retrieves more relevant
  evidence.
- **Acceptance criteria:** a model may be adopted for further experiments on
  stated diagnostic grounds (for example, removing truncation at an
  acceptable latency); that decision is recorded with its assumptions and is
  not a relevance claim.
- **Impact:** new semantic indexes and Hybrid rankings per profile. Existing
  B1 and October 18 runs stay unchanged.

### 4. Hybrid retrieval improvements

- **Hypothesis:** better chunking, embeddings or fusion may deliver sufficient
  evidence without enlarging the context unnecessarily.
- **Current baseline:** the current BM25 and equal-weight RRF Hybrid, kept as
  comparison baselines. BM25 parameters stay unchanged without evidence that
  they cause a weakness.
- **Controlled experiment:** compare ways to keep strong lexical matches,
  dense and sparse combinations, fusion weighting, candidate selection and
  optional reranking; a preliminary judged subset is acceptable (see
  [Evaluation scope](#evaluation-scope)).
- **Measurements:** evidence coverage against token budget, and retrieval
  quality against latency.
- **Acceptance criteria:** not defined.
- **Impact:** context size for answer generation (Investigation 6).

### 5. Qdrant feasibility

Qdrant is a candidate, not an approved replacement for the current
retrieval infrastructure. A storage backend is a decision that requires the
product owner.

- **Hypothesis:** a vector database may support dense, sparse and hybrid
  search with metadata filtering locally and offline, at acceptable
  operational cost.
- **Current baseline:** the current in-process BM25, semantic and Hybrid
  indexes and their persisted artifacts.
- **Controlled experiment:** reproduce current rankings or an agreed
  successor through a narrow integration boundary, without migrating existing
  code.
- **Measurements:** local development and offline deployment; dense, sparse and
  hybrid search; metadata filtering; document, section and page provenance;
  indexing and query speed; resource use and operational complexity;
  integration boundaries with the retrieval code; data isolation for possible
  multi-user use.
- **Acceptance criteria:** not defined. A database is not assumed to improve
  answer latency, relevance or grounding.
- **Impact:** storage and deployment boundaries. Database filtering restricts
  which records a query sees; it is not application-level authorization,
  which would remain a separate responsibility.

### 6. Grounded-answer reliability

- **Hypothesis:** some reliability checks are better handled by deterministic
  application components than by model instructions alone.
- **Current baseline:** the B1-2 and B1-3 prompt, which asked for citations,
  refusal and clarification and was not reliably followed.
- **Controlled experiment:** test each mechanism separately on fixed
  retrieved context. Mechanisms that do not depend on retrieval
  infrastructure, such as exact quotation verification, may be investigated
  independently of Investigations 2–5.
- **Mechanisms to examine:** evidence sufficiency before generation, citation
  presence and validity, claim-to-source support, exact quotation
  verification, refusal when evidence is missing, clarification of materially
  ambiguous questions and efficient context selection.
- **Acceptance criteria:** not defined.
- **Impact:** must preserve
  [PD-002](product-decisions.md#pd-002-require-direct-evidence-before-treating-retrieval-as-an-answer),
  [PD-004](product-decisions.md#pd-004-show-the-original-pdf-page-when-extracted-text-cannot-carry-the-evidence)
  and
  [PD-005](product-decisions.md#pd-005-ask-for-clarification-instead-of-guessing-the-users-meaning).
  Generated prose stays separate from book quotations, and exact provenance is
  guaranteed only for content FREDL retrieved and checked.

## LLM Model Comparison and Grounded Answer Reliability

> **Status: research direction approved; M1 research complete
> (17 October 2026).** PD-006, PD-007 and PD-008 clarifications are accepted.
> M1 covered candidate research and documentation only. Selection remains
> provisional. M2-A subsequently passed its approved one-request compatibility
> test. M2-B has now completed its approved 24 requests and awaits human
> review; no semantic quality or adoption decision follows. No alternative
> model was downloaded or loaded, and no inference was run during M1.

### Goal

Understand how different language models behave when they answer questions
from the same retrieved technical-document evidence. FREDL's aim is not fluent
answers but answers that help a reader find, understand and inspect source
material without relying on unsupported claims. This phase is a small,
interpretable comparison, not a general LLM benchmark.

It extends [Investigation 6](#6-grounded-answer-reliability): Investigation 6
examines reliability mechanisms with a fixed model, while this phase varies
the model with fixed evidence. Deterministic checks developed for either
apply to both.

### Research questions

1. How strongly does answer quality depend on the chosen LLM?
2. Which models follow source-grounding instructions reliably?
3. Which models provide valid and useful citations?
4. Which models reproduce direct quotations accurately?
5. Which models recognize insufficient evidence and refuse appropriately?
6. Which models recognize ambiguous questions and ask for clarification?
7. How do latency, memory use and context length affect practical usability?
8. Which reliability guarantees must be enforced outside the model?

A larger or newer model is not assumed to perform better.

### Starting point

- **Baseline:** Qwen3-4B Q4_K_M through llama.cpp with full GPU offload, as in
  [Investigation 1B](inference-performance-diagnostics.md#investigation-1b-gpu-baseline-for-all-twelve-b1-prompts-16-october-2026).
- **Known baseline behavior:** in six questions, missing citations, answers
  without sufficient evidence, no clarification and one quotation that occurs
  in no supplied passage
  ([B1-2](pdf-retrieval-experiments.md#b1-2-qwen3-4b-grounded-answer-smoke-test-16-october-2026),
  [B1-3](pdf-retrieval-experiments.md#b1-3-qwen3-4b-with-hybrid-top-5-versus-top-10-16-october-2026)).
  These are assistant checks still awaiting human verification.
- **Constraints from measurements:** a 6 GB GPU with about 860 MiB free after
  loading the baseline at a 16,384-token context with four slots; the largest
  B1 request needed about 9.2k tokens of context.
- **Governing decisions:** GPU-first experimentation without generalizing to
  CPU-only hardware ([PD-006](product-decisions.md#pd-006-prefer-gpu-acceleration-for-experiments-without-making-cuda-a-product-dependency)); sequential, single-user workflow
  measurement ([PD-007](product-decisions.md#pd-007-evaluate-performance-as-one-users-sequential-question-workflow)); local experimentation and replaceable models
  ([PD-008](product-decisions.md#pd-008-treat-qwen3-4b-as-a-baseline-and-keep-model-providers-replaceable)); direct evidence before answering
  ([PD-002](product-decisions.md#pd-002-require-direct-evidence-before-treating-retrieval-as-an-answer))
  and clarification instead of guessing
  ([PD-005](product-decisions.md#pd-005-ask-for-clarification-instead-of-guessing-the-users-meaning)).

### M1: select alternative local models

- **Research question:** which small set of local models from different
  families is worth comparing with the baseline?
- **Selection criteria:** compatibility with the available local inference
  runtime; ability to run on the current GPU; model size and quantization;
  context-window requirements; license and distribution restrictions;
  resource requirements; suitability for grounded question answering over
  technical documents.
- **Output:** a short, reasoned candidate list with the criteria each meets
  and the alternatives left out. The shortlist is an M1 research output;
  acquiring models and running M2 need separate approval.

### M1 findings (17 October 2026)

**Question:** which alternatives merit a controlled comparison with Qwen3-4B?
**Scope:** public primary-source research and the existing B1 reports only.
No private questions, passages, answers or artifacts were uploaded or copied
here. The baseline prompt is unchanged. **Verified** below means documented
in a cited source or previously measured in B1/Investigation 1B, not locally
tested during M1. **Estimates** are calculations, not allocation measurements.

#### Provisional comparison

Two alternative families suffice for the first comparison: different training,
tokenizers and attention designs near the baseline's size. No other candidate
was investigated. Larger models, reasoning variants and vision workflows add
resource/output-budget variables and are outside this shortlist, not measured
rejections. GB is decimal file size; GiB is binary. File size is a weight-memory
proxy, not an exact CUDA allocation.

| Family and version | Parameters | GGUF availability | Candidate quantization | Weight memory estimate | Context support | Expected hardware constraints | License | Main compatibility concerns | Why test |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-4B, existing baseline | 4.0B [Q] | Existing local GGUF, already measured | Q4_K_M unchanged | 2.50 GB file ≈ 2.33 GiB | B1 configured 16,384; local metadata reports trained 40,960; current card lists 32,768 native [Q] | Measured about 4,889 MiB GPU use at 16k; 37/37 layers offloaded | Apache-2.0 [Q] | Preserve exact template and disabled thinking; published/local context metadata differ | Reproducible reference with known preliminary grounding failures |
| Microsoft Phi-4-mini-instruct, not mini-reasoning | 3.8B [P] | Community GGUF by bartowski [PG], derived from Microsoft original | Q4_K_M first; Q5_K_M available, deferred | Q4: 2.49 GB ≈ 2.32 GiB; Q5: 2.85 GB ≈ 2.65 GiB [PG] | Maximum 131,072; LongRoPE from original 4,096 [PC] | Estimated 4.82–5.22 GiB total at 16k; plausible full offload, unverified | MIT [PL] | `phi3`, LongRoPE, partial rotary dimensions, tokenizer and stop metadata | Similar weight budget, technical/reasoning-oriented training, separate system role |
| Google Gemma 3 4B IT, QAT release | About 3.884B text + 417M vision [GT] | Official Google GGUF [GG] | QAT Q4_0; a post-training Q4_K_M is not an equivalent checkpoint | Text file 3.16 GB ≈ 2.94 GiB; separate 851 MB vision projector excluded [GF] | Advertised 128K; 1,024-token local attention interleaved with global attention [G, GT] | Conservative full-cache estimate 5.57–5.97 GiB total at 16k; fit depends on actual sliding-window allocation | Gemma Terms; gated access [GL, GF] | Role mapping, sliding-window KV allocation, exact GGUF metadata and CUDA path | Independent family/attention design; instruction-following and grounding evaluations motivate testing, not a quality prediction |

#### Compatibility and resource interpretation

Recorded runtime: llama.cpp b11574, commit `8bb3cf0d2`, CUDA 12.8.
Its [architecture registry][R] includes `qwen3`, `phi3` and `gemma3`.
Phi's official configuration identifies `Phi3ForCausalLM` [PC]; its GGUF
publisher records quantization with llama.cpp b4792 [PG]. Google documents
text-only llama.cpp use for its QAT GGUF [GG]. This establishes a documented
compatibility route, not proof of loading either file on this exact binary
and GPU. No load, tokenization or generation check occurred in M1. Do not
implicitly upgrade the runtime to make a candidate work.

The [hardware record](inference-performance-diagnostics.md#hardware-and-runtime)
reports 6,144 MiB physical VRAM, 5,738 MiB visible to CUDA and 5,648 MiB free
when idle. These are prior observations, not fresh memory readings. The usable
budget is smaller than nominal 6 GiB.

For one logical sequence, f16 K+V is approximately
`tokens × layers × KV heads × head dimension × 2 (K,V) × 2 bytes`.
Phi has 32 layers, 8 KV heads and head dimension 3072/24 = 128 [PC]:
2.00 GiB KV at 16,384 tokens, 1.50 GiB at 12,288. At 131,072 tokens KV alone
would be 16 GiB: the advertised window is not a GPU-resident budget here.

Google's implementation lists 34 text layers, 4 KV heads and dimension 256
[GC]. Full-sequence allocation for every layer would use about 2.125 GiB KV
at 16k, or 1.594 GiB at 12k. Keeping only local sliding windows could reduce
that substantially; the actual cache allocation in this llama.cpp build is
unverified. Do not count on that saving. Google's implementation supplies
these dimensions, not the llama.cpp context setting.

Table totals add an **assumed 0.5–0.9 GiB** of compute/runtime buffers to
file-size weights and full-length f16 KV. This allowance is not a measured
bound: batches, padding, GPU kernels and allocators may invalidate it. The
same arithmetic at 12k gives Phi 4.32–4.72 GiB and Gemma 5.04–5.44 GiB.
Host RAM, mapped weights and startup peaks are additional resource dimensions.
Q5 is deferred; f16/BF16 weights exceed a full-GPU weight budget for either
alternative. The runtime's actual slot/cache allocation must be logged.

B1 top-5 used 3,134–4,950 Qwen prompt tokens; top-10 used 5,052–8,538.
The largest needed 9,238 with 700 output tokens. **8k cannot replay every B1
input even with the baseline tokenizer.** M2 must count unchanged messages
using each candidate's tokenizer and include template/special tokens plus
700 output tokens. A 12k fallback is provisional until counts and allocation
are checked. Never truncate evidence or silently shift context to force fit.

#### Provisional order, grounding and license risks

**First: Phi-4-mini-instruct Q4_K_M**, against Qwen only. Its weight/cache
estimate offers more headroom and its separate system role reduces the
instruction-placement difference. This orders experiments, not grounding
quality. **Second: Gemma 3 4B QAT Q4_0**, for review after Phi; it is not
part of the first M2 run.

Phi's publisher describes instruction tuning, reasoning-oriented technical
and code training, but warns about factual errors, language disparities and
limited code coverage [P]. Gemma reports instruction-following and grounding
evaluations [G]; these do not measure FREDL's citation contract or this QAT
GGUF. Neither has verified FREDL citation adherence, quotation fidelity,
abstention or ambiguity handling. Quantization, prompt length and templates
may affect all of them. A community Phi conversion adds provenance/calibration
variables: pin its revision and file hash later.

Phi redistribution requires preserving copyright and permission notices [PL].
Gemma redistribution carries the agreement, downstream use restrictions,
modification notices and a Notice file where applicable; its prohibited-use
policy applies [GL]. Official download access requires accepting conditions
and sharing contact information [GF]. No terms were accepted and no model
redistribution is proposed. These are model-specific obligations, independent
of FREDL's repository license.

#### Minimal M2 proposal — not authorized or executed

1. **Freeze inputs privately.** Reuse the six B1 questions and twelve stored
   top-5/top-10 evidence payloads: twelve requests per model, 24 total for
   Qwen plus Phi, once each. Preserve ranks, passage order/text, `[S1]`-style
   labels, book/page metadata and record mappings. Hash the stored instructions,
   questions and evidence privately. Do not reretrieve or re-extract; do not
   give either model expected outcomes or labels. Maria should confirm which
   supplied contexts support an answer before comparison. This does not freeze
   the Golden Set or complete judged B1.
2. **Preflight after acquisition/run approval.** Pin revision, filename,
   SHA256, quantization, embedded template, stop tokens and runtime commit.
   Check tokenizer, LongRoPE and rotary metadata, and count all inputs.
   Record devices, actual offloaded layers, weights/KV/compute buffers and
   free/peak VRAM. Start with the largest input for fit, then the remaining
   requests. Stop for incompatible metadata, OOM, truncation, context shifting
   or unexpected CPU fallback. Do not repair/upgrade the runtime or quantize
   KV within this comparison.
3. **Keep configuration comparable.** Retain the working CUDA library
   environment, b11574, `-ngl 99 -c 16384`, the recorded four-slot configuration
   and f16 KV; send requests sequentially. Slots are a baseline property,
   not a concurrency target. Set `cache_prompt: false`; verify `cache_n = 0`.
   If needed, propose a common 12,288 context only when every input fits for
   both tokenizers; document the change and rerun both. No silent offload
   reduction or automatic fit adjustment.
4. **Keep B1 instruction content.** Temperature 0, top-p 1, seed 0, max output
   700; record all effective sampler, penalty, EOS and stop settings as well.
   Qwen keeps its baseline template and exact thinking-off controls. Phi uses
   its native system/user/assistant template [P, PG], with its own role/end
   tokens and no Qwen reasoning controls. Inspect rendered prompts to ensure
   instructions/evidence occur once. Do not tune sampling for individual
   failures. A later prompt revision needs a separate experiment and baseline.
5. **Expose template confounders.** If Gemma is approved later, its user/model
   format places unchanged system instructions before the unchanged question
   and evidence in the first user turn [GP]. Retain content hashes and a
   separate rendered-prompt hash; document role relocation and verify the
   chosen GGUF's actual mapping. Text-only testing excludes the vision
   projector. No template changes are made during M1.
6. **Measure and review separately.** Check decision-prefix compliance,
   citation presence/identifier validity, exact quotation matches, token counts,
   finish reasons and output truncation. Maria judges claim support, unsupported
   additions, refusal, clarification, language and usefulness against supplied
   evidence. Identifier/quotation matches alone do not prove support or
   success. Preserve failures and report per-case differences, without a
   superiority claim from six questions. Record load, context/validation time,
   prefill/decode, first token, wall time and resources separately. Retrieval
   is replayed and untimed; time to useful evidence and full workflow latency
   are unavailable, not zero.

**Proposed execution acceptance:** twelve uncached requests per model, preserved
payloads, no silent truncation or CPU fallback, complete configuration/resource
logs. These criteria accept an interpretable smoke test, not a model. There
is no numerical quality or product-latency threshold; adoption awaits review.

**Related decisions:** the resource gate applies
[PD-006](product-decisions.md#pd-006-prefer-gpu-acceleration-for-experiments-without-making-cuda-a-product-dependency)
without a CUDA product dependency. Sequential requests and separate stage
measurements follow
[PD-007](product-decisions.md#pd-007-evaluate-performance-as-one-users-sequential-question-workflow);
replay does not validate the complete workflow. Local handling, fixed evidence
and the retained Qwen baseline follow
[PD-008](product-decisions.md#pd-008-treat-qwen3-4b-as-a-baseline-and-keep-model-providers-replaceable).
M1 approves no provider abstraction or permanent model choice.

**Remaining approval questions:** confirm the first candidate and exact GGUF
source; authorize acquisition and M2 separately; approve the twelve-payload
sample, unchanged B1 instructions, settings and execution criteria; confirm
Maria's human review protocol and whether a common 12k fallback is allowed.
Gemma additionally needs terms/access and role-mapping review before acquisition.
No runtime upgrade, redistribution or M3/M4 execution is authorized.

#### M1 public sources

Checked on 17 October 2026. Cards/listings are mutable; M2 must pin artifacts.
A listing establishes availability, not local integrity.

- [Q: Qwen official card][Q]; local baseline measurements:
  [Investigation 1B](inference-performance-diagnostics.md#investigation-1b-gpu-baseline-for-all-twelve-b1-prompts-16-october-2026).
- [P: Microsoft model card][P], [PC: official configuration][PC],
  [PL: official MIT license][PL].
- [PG: Phi GGUF publisher's quantization record and sizes][PG]: primary
  documentation for the community artifact, not a Microsoft GGUF release.
- [G: Google model card][G], [GT: technical report][GT],
  [GC: Google's implementation dimensions][GC].
- [GG: official QAT GGUF card and text-only usage][GG], [GF: file listing][GF],
  [GP: Google role documentation][GP], [GL: Gemma Terms of Use][GL].
- [R: llama.cpp architecture registry at the recorded commit][R].
  Architecture registration is narrower evidence than an end-to-end load test.

[Q]: https://huggingface.co/Qwen/Qwen3-4B
[P]: https://huggingface.co/microsoft/Phi-4-mini-instruct
[PC]: https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/config.json
[PL]: https://huggingface.co/microsoft/Phi-4-mini-instruct/blob/main/LICENSE
[PG]: https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/blob/main/README.md
[G]: https://ai.google.dev/gemma/docs/core/model_card_3
[GT]: https://arxiv.org/html/2503.19786v1
[GC]: https://raw.githubusercontent.com/google/gemma.cpp/main/gemma/configs.cc
[GG]: https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf
[GF]: https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf/tree/main
[GP]: https://ai.google.dev/gemma/docs/core/prompt-structure
[GL]: https://ai.google.dev/gemma/terms
[R]: https://raw.githubusercontent.com/ggml-org/llama.cpp/8bb3cf0d2/src/llama-arch.cpp

### M2-A: compatibility and context-fit protocol (17 October 2026)

**Status: protocol approved and executed once on 17 October 2026; technical
compatibility gate passed, result awaiting review.** Phi-4-mini-instruct
Q4_K_M remains provisional. The preparation and command examples below are
preserved as the approved protocol; execution findings and one response-schema
correction are recorded in the [experiment journal](pdf-retrieval-experiments.md#m2-a-phi-4-mini-compatibility-smoke-17-october-2026).
No further generation is authorized by this record.
M2-A is one generation request after non-generating checks, not the twelve-case
quality comparison. It follows PD-006 (verify GPU use), PD-007 (sequential
requests and separate timings) and PD-008 (fixed evidence, retained baseline).
The M1 proposal above remains historical planning; this gate precedes it.

#### Read-only preparation findings

Inspected both B1 configuration records, all twelve CPU response metadata
records, the twelve Investigation 1B GPU records and summary, original B1
prompt builders, and the verbose GPU load log. Q1–Q6 below are the existing
public aliases; private question/record IDs, text, paths and fingerprints stay
outside this document. Only `llama-server --help` and `--version` were invoked:
no server or model was started. Installed binary confirms build 11574, commit
`8bb3cf0d2`, version 0.6.0-dev; proposed flags below occur in its help.

Both B1 configurations have identical system instructions, question order,
model hash and generation settings: temperature 0, top-p 1, seed 0, max tokens
700. Their saved Golden Set and Hybrid-run file hashes still match the current
private files. CPU B1 and uncached GPU 1B report identical prompt counts for
all twelve requests. CPU B1 records `finish_reason: stop` for every request;
GPU 1B recorded content and timings but did not persist final finish reasons.
Do not infer GPU EOS/completion validation from the CPU finish reasons.

**Storage correction to M1:** these are saved request metadata, not complete
saved request bodies. The B1 JSON stores sources (IDs, labels, pages, order),
question, response and `prompt_sha256`; it omits passage text and messages.
The original builder would re-extract PDFs. It must not be rerun unchanged.
For this preparation, existing local extraction snapshots were parsed as inert
pickle opcode data, without resolving globals or calling pickle constructors.
All twelve bodies were reconstructed in memory using the saved source order
and original formatting; all twelve matched their original body hashes.
No payload files were written and no extraction, retrieval or inference ran.
Thus **all twelve can preserve their exact source evidence**, subject to the
Phi context gate below. A later freeze must repeat hash checks and write new
private copies, never overwrite B1 artifacts. Do not use unrestricted pickle
loading for that step.

The legacy hash is `sha256(json.dumps(body, sort_keys=True).encode())` with
Python's default JSON options. `body` contains the original Qwen model alias,
original two messages and the four original generation settings; it excludes
later stream/cache controls. Verify this hash before changing the model alias.
The exact user-message reconstruction is:

```python
blocks = [
    f"[{source['label']}] book: {source['book_path']} | physical pages: "
    f"{source['pages'][0]}-{source['pages'][-1]}\n{original_record_text}"
    for source, original_record_text in ordered_saved_sources_with_snapshot_text
]
user = 'Source passages:\n\n' + '\n\n'.join(blocks) + '\n\nQuestion: ' + saved_question
```

Use the saved configuration's system string, exact settings and original model
alias to verify the body hash; no text normalization. Verify each source's book
and complete page list against the snapshot before building its block.
Freeze each verified `messages` array as `Q1-top5.messages.json`, etc., and keep
a private manifest mapping aliases to original files, source metadata and
hashes. A messages file has exactly the `system` and `user` objects, no labels
or expected outcomes. Store its content hash separately from the rendered
Phi prompt hash. This freeze is a prerequisite, not an existing exported file.

#### All twelve context requirements

Counts are **previously measured Qwen tokens**, including the effective Qwen
chat formatting, from saved `usage.prompt_tokens` and matching 1B `prompt_n`
with `cache_n = 0`. Adding 700 is a calculated maximum-output reservation,
not a newly measured count. Do not add chat overhead again to these Qwen
counts. Phi counts, template overhead and the longest Phi request are **not
measured**: no Phi tokenizer has been acquired. No fixed Qwen-to-Phi multiplier
is assumed.

| Request | Passages | Measured Qwen input | Input + 700 | Headroom at 16,384 | Fits 8,192 with 700? | Phi input |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Q1 top-5 | 9 | 3,134 | 3,834 | 12,550 | yes | unknown |
| Q2 top-5 | 11 | 3,664 | 4,364 | 12,020 | yes | unknown |
| Q3 top-5 | 9 | 3,191 | 3,891 | 12,493 | yes | unknown |
| Q4 top-5 | 11 | 3,849 | 4,549 | 11,835 | yes | unknown |
| Q5 top-5 | 11 | 4,000 | 4,700 | 11,684 | yes | unknown |
| Q6 top-5 | 13 | 4,950 | 5,650 | 10,734 | yes | unknown |
| Q1 top-10 | 15 | 5,052 | 5,752 | 10,632 | yes | unknown |
| Q2 top-10 | 18 | 5,709 | 6,409 | 9,975 | yes | unknown |
| Q3 top-10 | 22 | 7,657 | 8,357 | 8,027 | **no** | unknown |
| Q4 top-10 | 22 | 7,427 | 8,127 | 8,257 | yes, only 65 left | unknown |
| Q5 top-10 | 25 | 8,538 | 9,238 | 7,146 | **no** | unknown |
| Q6 top-10 | 22 | 8,027 | 8,727 | 7,657 | **no** | unknown |

All twelve fit 12,288 and 16,384 for Qwen. The four largest top-10 inputs
(Q3–Q6) deserve particular Phi preflight attention; Q4 is also marginal at 8k.
Any input may exceed a proposed Phi limit until actually tokenized.
At 16,384, the exact Phi input cap is 15,684 including every special token;
at 12,288 it is 11,588. At least 700 output positions must remain in the
**actual per-sequence capacity**, not merely in the requested command-line
context. Equality is the theoretical boundary; stop if runtime padding or
capacity rules make it unusable. No evidence truncation, reordering,
summarization, removal, regenerated retrieval or automatic smaller subset.
If one fails, report its alias/count and stop before generation.

Qwen's template is ChatML-like with Qwen-specific thinking controls; keep the
original instructions and thinking-off baseline unchanged. Phi's documented
format uses `<|system|>`, `<|user|>`, `<|assistant|>` and `<|end|>` [P, PG].
Use its embedded native template, not the Qwen rendered string. Verify the
assistant generation prefix, one copy of each message, BOS insertion policy,
EOS/EOG IDs and turn-end mapping from the actual GGUF. Word/character counts
or an upstream tokenizer different from the selected GGUF cannot certify fit.
LongRoPE, 0.75 partial rotary factor, vocabulary 200,064, 32 layers and 8 KV
heads must agree with the official Phi configuration [PC]. No metadata or
RoPE overrides to repair a mismatch within this test.

#### Exact candidate artifact and resource gate

Proposed artifact: repository `bartowski/microsoft_Phi-4-mini-instruct-GGUF`,
file **`microsoft_Phi-4-mini-instruct-Q4_K_M.gguf`**. The actual listing includes
`microsoft_`; the shorter README table label must not be used as a filename.
Pin the file-upload commit **`915429cb42fe8eba71bd1d3117a7d63070892268`**.
Its [publisher commit and LFS pointer](https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/commit/915429cb42fe8eba71bd1d3117a7d63070892268)
record **2,491,874,688 bytes** and SHA256
`01999f17c39cc3074afae5e9c539bc82d45f2dd7faa3917c66cbef76fce8c0c2`.
The [file metadata page](https://huggingface.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF/blob/main/microsoft_Phi-4-mini-instruct-Q4_K_M.gguf)
reports the same SHA256. This is the publisher's expected digest, **not a
locally verified file**. After separately approved acquisition, compare actual
byte length and `sha256sum` to these values before loading; record source
revision and actual digest privately. An Xet chunk hash is not the file SHA256.
No `-hf` or other automatic download flag appears in the smoke commands.

At 16k, weight-size proxy is about 2.321 GiB; single shared-pool f16 KV estimate
is 2.000 GiB, plus the M1 assumed 0.5–0.9 GiB buffers: about **4.82–5.22 GiB**.
These are estimates. Prior CUDA-visible/free budgets were 5,738/5,648 MiB;
new free memory must be measured. Log actual model, host, KV and compute
allocations plus sampled device memory. Require at least **256 MiB sampled
free VRAM** during the smoke as a proposed operational margin, not a product
threshold or proof that short peaks were captured.

The previous Qwen log reports `n_ctx = n_ctx_seq = 16384`, `n_seq_max = 4`,
`n_slots = 4`, `kv_unified = true`, one 16,384-cell KV pool of 2,304 MiB,
6 generation/batch threads, batch 2,048, microbatch 512 and Flash Attention
auto. Do **not** divide this recorded context by four or multiply its measured
KV allocation by four. Explicit `-np 4 --kv-unified` is intended to reproduce
that layout; verify Phi's actual layout instead of assuming the same semantics.
No parallel requests. If memory/capacity fails, do not retry with 12k, fewer
slots, CPU layers, smaller batches or quantized KV without a revised protocol.

#### Proposed commands — review only, not executed

Commands assume separately approved acquisition and the private payload freeze
above. Set the three private paths locally; no real private paths or content
belong in public documentation. `M2A_RUNTIME_ROOT` contains the two already
installed runtime directories; `M2A_INPUTS` contains the twelve verified
messages files. Use a new output directory. A missing prerequisite is a stop,
not permission to reconstruct different evidence. These are manual stages:
**do not paste the generation stage before reviewing preflight artifacts**.

```bash
set -euo pipefail
: "${M2A_RUNTIME_ROOT:?set existing runtime root}"
: "${M2A_MODEL:?set acquired pinned GGUF file}"
: "${M2A_INPUTS:?set private verified messages directory}"
: "${M2A_OUT:?set new private output directory}"
umask 077
mkdir "$M2A_OUT"
export M2A_INPUTS M2A_OUT
M2A_SERVER="$M2A_RUNTIME_ROOT/llama-b11574/llama-server"
export LD_LIBRARY_PATH="$M2A_RUNTIME_ROOT/llama-b11574:$M2A_RUNTIME_ROOT/cudart-llama-b11574-bin-ubuntu-cuda-12.8-x64"
export M2A_MODEL
python3 - <<'PY'
import hashlib, os
from pathlib import Path
p = Path(os.environ['M2A_MODEL'])
assert p.stat().st_size == 2491874688
h = hashlib.sha256()
with p.open('rb') as f:
    while chunk := f.read(1 << 20):
        h.update(chunk)
assert h.hexdigest() == '01999f17c39cc3074afae5e9c539bc82d45f2dd7faa3917c66cbef76fce8c0c2'
PY
sha256sum "$M2A_MODEL" "$M2A_SERVER" > "$M2A_OUT/artifact-sha256.txt"
"$M2A_SERVER" --version > "$M2A_OUT/runtime-version.txt" 2>&1
"$M2A_SERVER" --list-devices > "$M2A_OUT/devices.txt" 2>&1
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv > "$M2A_OUT/gpu-before.csv"
```

Stop here unless the digest/version/device checks pass, the intended GPU is
visible and the approved loopback port is free. No other inference server may
be resident. In terminal A, preserving the variables from setup:

```bash
"$M2A_SERVER" -m "$M2A_MODEL" --alias phi4-mini-m2a \
  --host 127.0.0.1 --port 18080 -ngl 99 --fit off \
  -c 16384 -np 4 --kv-unified -ctk f16 -ctv f16 \
  -t 6 -tb 6 -b 2048 -ub 512 -fa auto --jinja \
  --no-context-shift --no-warmup -lv 4 \
  > "$M2A_OUT/server.log" 2>&1
```

`--no-warmup` avoids an extra empty inference. `--fit off` forbids automatic
memory adaptation; explicit settings expose deviations from the old implicit
defaults. Phi receives no Qwen-specific reasoning-budget controls. Record the
resolved sampler and template settings. Review `server.log` before continuing:
all model layers expected by this architecture and KV must be on CUDA0; do not
require Qwen's model-specific 37-layer number. Log `n_ctx_seq`, KV cells/types,
all allocation sizes, tokenizer metadata and the chosen template. Keep logs
private since verbosity may expose prompt/token content. Do not proceed on a
fallback template or an unexplained special-token mismatch.

In terminal B, set/export `M2A_INPUTS` and `M2A_OUT` to the same private values.
Capture properties and start sampling before preflight/generation:

```bash
set -euo pipefail
: "${M2A_INPUTS:?set private verified messages directory}"
: "${M2A_OUT:?set the same output directory as terminal A}"
export M2A_INPUTS M2A_OUT
curl --fail --silent --show-error --max-time 10 \
  http://127.0.0.1:18080/health > "$M2A_OUT/health.json"
curl --fail --silent --show-error --max-time 10 \
  http://127.0.0.1:18080/props > "$M2A_OUT/props.json"
nvidia-smi --query-gpu=timestamp,memory.used,memory.free,utilization.gpu \
  --format=csv -lms 200 > "$M2A_OUT/gpu-during.csv" &
M2A_MONITOR_PID=$!
trap 'kill "$M2A_MONITOR_PID" 2>/dev/null || true' EXIT
```

After metadata review, perform twelve template/tokenization checks and prepare
one request. These endpoints do not generate; they still require the loaded
model and are therefore proposed only. `add_special: true` follows the model's
BOS policy; inspect the resulting IDs/pieces against the template and metadata
for missing/double BOS, literal role tokens and unexpected EOS. The native
completion uses the resulting integer IDs, avoiding a second template or BOS
application. Its endpoint differs from B1 chat completion; this is a
compatibility probe, not a measured quality comparison.

```bash
python3 - <<'PY'
import hashlib, json, os, urllib.request
from pathlib import Path
root, out = Path(os.environ['M2A_INPUTS']), Path(os.environ['M2A_OUT'])
props = json.loads((out / 'props.json').read_text())
capacity = props['default_generation_settings']['n_ctx']
assert capacity == 16384  # also verify n_ctx_seq in the load log

def post(route, body):
    req = urllib.request.Request('http://127.0.0.1:18080/' + route,
        data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)

rows = []
for cutoff in (5, 10):
    for q in range(1, 7):
        alias = f'Q{q}-top{cutoff}'
        raw = (root / f'{alias}.messages.json').read_bytes()
        messages = json.loads(raw)
        assert [m['role'] for m in messages] == ['system', 'user']
        rendered = post('apply-template', {'messages': messages})
        prompt = rendered['prompt']
        assert all(prompt.count(m['content']) == 1 for m in messages)
        tokenized = post('tokenize', {'content': prompt, 'add_special': True,
                                     'parse_special': True, 'with_pieces': True})
        ids = [x['id'] for x in tokenized['tokens']]
        assert ids and all(type(x) is int for x in ids)
        (out / f'{alias}.template.json').write_text(json.dumps(rendered))
        (out / f'{alias}.tokens.json').write_text(json.dumps(tokenized))
        rows.append({'alias': alias, 'input_tokens': len(ids),
                     'required': len(ids) + 700, 'ids': ids,
                     'messages_sha256': hashlib.sha256(raw).hexdigest(),
                     'rendered_sha256': hashlib.sha256(prompt.encode()).hexdigest()})
(out / 'context-counts.json').write_text(json.dumps(
    [{k: v for k, v in r.items() if k != 'ids'} for r in rows], indent=2))
assert all(r['required'] <= capacity for r in rows), 'STOP: context does not fit'
chosen = max(rows, key=lambda r: r['input_tokens'])  # first alias wins ties
(out / 'selected.json').write_text(json.dumps(
    {k: v for k, v in chosen.items() if k != 'ids'}, indent=2))
request = {'prompt': chosen['ids'], 'n_predict': 700, 'temperature': 0,
           'top_p': 1, 'seed': 0, 'cache_prompt': False, 'stream': False,
           'ignore_eos': False, 'stop': [], 'return_tokens': True}
(out / 'smoke-request.json').write_text(json.dumps(request))
PY
```

**Manual gate:** verify frozen-message hashes against the private B1 manifest,
all twelve counts, actual capacity, rendered role boundaries and token pieces,
BOS/EOS/EOG metadata, layer/KV offload and VRAM margin. The count script checks
whole-message inclusion but cannot certify every template boundary. Do not
advance on a missing property or unsupported endpoint; record it and stop.
Q5 top-10 is longest for Qwen; the script chooses the actual longest for Phi
rather than assuming it is the same request. Then send exactly one generation:

```bash
curl --fail --silent --show-error --max-time 300 \
  -H 'Content-Type: application/json' \
  --data-binary "@$M2A_OUT/smoke-request.json" \
  http://127.0.0.1:18080/completion > "$M2A_OUT/smoke-response.json"
```

Review completion and then end the monitor and server (Ctrl-C in terminal A):

```bash
python3 - <<'PY'
import json, os
from pathlib import Path
out = Path(os.environ['M2A_OUT'])
r = json.loads((out / 'smoke-response.json').read_text())
s = json.loads((out / 'selected.json').read_text())
assert r['truncated'] is False
assert r['stop_type'] == 'eos'  # no custom stop strings were requested
assert r['timings']['cache_n'] == 0
assert r['timings']['prompt_n'] == s['input_tokens']
assert r['tokens_evaluated'] == s['input_tokens']
assert r['generation_settings']['n_ctx'] == 16384
assert 0 < r['timings']['predicted_n'] < 700
assert r['content'].strip()
PY
kill "$M2A_MONITOR_PID"
wait "$M2A_MONITOR_PID" || true
trap - EXIT
```

If any stage fails, stop the monitor and this test's server without retrying
or editing input/settings. Shell cleanup is required even after an assertion
or curl failure. The numeric/flag assertions do not prove semantic completion:
Maria must confirm the answer did not end mid-sentence, emit a new user/system
turn, loop role tokens or produce an empty/garbled answer. Record the generated
IDs and terminal EOG diagnostics where available; an unexplained EOS mapping
is inconclusive, not a pass. A valid refusal may be complete. Citation and
grounding defects are recorded for later quality analysis, not repaired here.

#### Acceptance, stop conditions and subsequent Qwen baseline

Accept M2-A only when all gates pass: artifact integrity, correct native
architecture/tokenizer/template, all twelve unmodified payloads counted and
fitting, actual 16k shared-pool layout verified, full GPU/KV offload, allocation
logs and sampled VRAM margin recorded, exactly one uncached generation with
all input tokens evaluated and a natural, complete EOG-terminated response.
Report loading/tokenization/generation separately; no quality or product
latency claim follows. A one-request test cannot prove all twelve outputs
complete or validate multilingual/citation behavior in general.

**Stop immediately** for checksum/size drift, missing data or hash mismatch,
unknown/fallback template, duplicated/missing messages or BOS, incorrect role
or EOG IDs, unsupported LongRoPE/rotary metadata, device/library errors, OOM,
unexpected CPU/KV fallback, auto-fit/configuration drift, insufficient sampled
memory margin, any input plus 700 exceeding actual capacity, prompt truncation
or context shift, cached prompt tokens, timeout/HTTP error, missing diagnostic
fields, `limit`/unexpected stop, an output limit hit or visibly incomplete
response. Preserve failure logs; do not silently shorten evidence, lower
context, switch quantization, add stop strings or raise output budget.

No fresh Qwen run is needed to write or inspect this protocol, or to establish
Phi's isolated loading compatibility. Before an actual model comparison,
rerun Qwen and Phi on the same frozen inputs and agreed settings. A fresh Qwen
baseline is specifically required if context/slots/KV/batches/FA/threading,
runtime/CUDA environment, effective sampler, prompt content, template controls,
endpoint, cache policy or timing instrumentation changes or is unverifiable.
The proposed explicit fit/warmup flags and native-token completion endpoint
make this smoke unsuitable as a direct latency/quality comparison with old
B1. For the later comparison, choose one common endpoint and instrumentation,
verify Qwen's original rendered input/token count and thinking-off behavior,
and rerun it; preserve historical B1/1B records unchanged in either case.

**Approval requested next:** pinned artifact acquisition and this one-request
M2-A only; private payload freeze; native-token completion endpoint; explicit
four-slot shared-KV configuration and 256 MiB sampled margin. Confirm whether
a later common 12k fallback should be separately considered after a failed
16k gate (none is authorized here). Exact Phi counts, GPU allocations and EOG
handling remain execution-time unknowns. Passing M2-A does not authorize the
full comparison or model adoption.

Endpoint/flag semantics were checked against the installed binary's help and
[llama.cpp server documentation at the recorded commit](https://raw.githubusercontent.com/ggml-org/llama.cpp/8bb3cf0d2/tools/server/README.md).
Runtime-returned fields and logs must confirm them; missing fields stop the
protocol rather than inviting assumptions.

#### M2-A execution outcome (17 October 2026)

- Pinned GGUF size and SHA256 matched the approved values. All twelve B1 body
  hashes matched before freezing new private messages; original artifacts
  remained unchanged after execution.
- Native Phi templates, special-token positions and detokenization matched
  every unchanged input. Measured Phi inputs ranged from 3,077 to 8,267 tokens;
  the largest plus the 700-token reserve needed 8,967. All twelve fit.
- Effective capacity was 16,384 per request, confirmed by load logs, `/props`,
  all four `/slots` entries and processing an 8,267-token prompt without
  truncation. It was not inferred from `-c` alone or divided by four.
- Full 33/33 GPU layer offload; CUDA model/KV/compute buffers were
  2,368.57 / 2,048.00 / 88.01 MiB. Minimum sampled free VRAM was 1,121 MiB;
  maximum sampled GPU use was 4,619 MiB. No CPU-only fallback or OOM occurred.
- Exactly one uncached native-token `/completion` request: 8,267 input and
  73 generated tokens, HTTP 200, natural EOG `<|end|>`, no truncation;
  prefill 13.176 s, decode 1.304 s, client wall 14.493 s. No retry.
- Native response settings are flat and omit `n_ctx`; use the independently
  verified logs/properties/slots for capacity rather than the preparation
  snippet's response-field assertion. No configuration change was needed.
- The loader's disabled-Phi-SWA warning and generic RoPE log label are recorded
  in the journal with their scope. Long/short factor tensors were present;
  this is not validation of the advertised 128k window.
- Server and sampler stopped. M2-B is technically feasible with the same
  frozen evidence and 16k configuration, but requires separate approval, an
  agreed common endpoint and fresh Qwen baseline. No grounding-quality
  acceptance, permanent model choice or full comparison follows.

Full token counts, effective settings, limitations and measurements:
[M2-A experiment record](pdf-retrieval-experiments.md#m2-a-phi-4-mini-compatibility-smoke-17-october-2026).

### M2: controlled answer comparison

M2-A is the completed compatibility gate above. M2-B denotes the controlled
baseline-versus-alternative comparison below: its approved 24 requests have
completed and await human review. No model winner or adoption is established.

- **Research question:** with identical evidence, how does one alternative
  model's behavior differ from the baseline's?
- **Design:** the baseline and one alternative at first.
- **Fixed:** the same questions, retrieved passages and citation identifiers;
  equivalent answer instructions where technically possible; comparable
  generation settings; sequential requests; recorded hardware and runtime.
- **Not changed:** retrieval, chunking and ranking.
- **Experimental variables:** unavoidable model-specific differences, such as
  chat templates or the way a model's reasoning mode is disabled, are recorded
  as variables rather than hidden.
- **Measurements:** answer behavior for M3, and latency stages and resource
  use following [PD-007](product-decisions.md#pd-007-evaluate-performance-as-one-users-sequential-question-workflow).

#### M2-B controlled comparison protocol (17 October 2026)

**Status: executed, 24/24 technically complete; awaiting human review.**
(Since 1 November 2026 that review is optional; see [PD-009](product-decisions.md#pd-009-evaluate-without-mandatory-manual-annotation).)
The approved scope was exactly 24 fresh requests: two pinned models × six
existing questions × two saved contexts. The design below is preserved as
protocol; [execution observations](pdf-retrieval-experiments.md#m2-b-controlled-comparison-execution-17-october-2026)
are recorded separately. The M2-A answer and historical B1/1B answers are not reused as M2-B
responses. The earlier preparation itself performed no new model, inference, retrieval
or production work. See the [preparation record](pdf-retrieval-experiments.md#m2-b-controlled-comparison-preparation-17-october-2026).

**Inputs and targets.** The twelve original request-body hashes and frozen
messages were reverified. A separate private snapshot holds exact questions,
instructions, ordered passage text, source IDs/record mappings and generation
configuration, with per-component and package hashes. The original B1/M2-A
files remain unchanged. The inference runner must read messages only, never
expectations, source judgments or review notes. Six private draft expectations
have twelve context-specific entries: answer with citations, partial answer
with explicit limits, insufficient-evidence refusal or clarification. A partial
answer uses the existing `DECISION: ANSWER` format, not a new prompt category.
Maria must review and approve these targets before generation; no answer is
scored now. Anchor absence alone cannot establish evidence insufficiency.

**Proposed common inference mechanism.** Use the existing llama.cpp b11574 /
`8bb3cf0d2` and CUDA environment, loopback only. One model resident at a time,
requests sequential. Native `/apply-template`, then `/tokenize`, then integer
token IDs to `/completion` for both models; no raw untemplated prompt or second
rendering. Preserve identical system/user contents and all evidence. Before
any generation, verify exact message inclusion, native role/BOS/EOG handling,
roundtrip, all twelve token budgets and effective capacity for the loaded model.
Qwen retains `--reasoning off --reasoning-budget 0`; confirm that its native
rendering disables thinking. Phi uses its verified native template without
Qwen controls. No template override, extra instructions or response rescue.

| Control | Proposed value for both models |
| --- | --- |
| Context/layout | `-c 16384 -np 4 --kv-unified`; verify per-request capacity from logs, properties and slots |
| GPU/memory | `-ngl 99 --fit off -ctk f16 -ctv f16`; full model-layer and KV offload; at least 256 MiB sampled free VRAM |
| Threads/batching | `-t 6 -tb 6 -b 2048 -ub 512 -fa auto` |
| Template and safety | `--jinja --no-context-shift --no-warmup`; no automatic downloads, fit changes or retries |
| Generation | 700 output tokens, temperature 0, top-p 1, seed 0 |
| Explicit effective sampler controls | top-k 40, min-p 0.05, typical-p 1, repeat penalty 1 / last 64, presence/frequency penalties 0, DRY multiplier 0, XTC probability 0, mirostat 0 |
| Request behavior | `cache_prompt: false`, `ignore_eos: false`, no custom stop strings, non-streaming, raw generated IDs retained |
| Instrumentation | private raw response/settings/logs; 200 ms GPU sampling; 300 s request timeout as an operational stop, not a product latency target |

The private inference plan pins both existing artifact hashes and binary hash.
The complete effective sampler order, defaults and any inactive controls are
also captured from the runtime and each response. Historical B1 saved only a
subset of sampler settings: explicitly recording common values creates a new
comparison baseline, not proof of bit-identical historical generation.
Different tokenizers/templates, Qwen's thinking-off wrapper and model-specific
EOG sets remain declared variables. Equal token budgets do not imply equal
word counts. Qwen's current native rendering has not been executed/tokenized
in this preparation: verify its counts against B1 and stop on unexplained
drift before generation. A historical rendered-prompt hash is unavailable;
unchanged message hashes plus token counts cannot prove identical historical
template bytes. No comparison claim hides this limitation.

**Planned order: four blocks of six, 24 total.** Define set A as top-5 for
odd question positions and top-10 for even positions; set B contains the other
context for each question. Retain question order within each set. Run:

1. Fresh Qwen, set A; stop server.
2. Fresh Phi, set A.
3. Same Phi server, set B; stop server.
4. Fresh Qwen, set B; stop server.

Each block contains three inputs of each cutoff, each input is run once per
model, and each model runs first for three top-5 and three top-10 pairs. This
uses three server loads and avoids simultaneous model residency. It reduces
simple order bias but does not remove thermal/session/order effects. Record
load boundaries and first requests; no extra warmup generation or repeats.
The exact 24-row schedule is frozen privately.

**Measurements per request.** Record model and GGUF SHA256, binary/version,
server arguments/effective sampler, model load/session, input/rendered/token
hashes, effective sequence capacity, prompt/generated tokens, prefill/decode,
client wall time, finish reason/EOG token, truncation, prefix reuse, sampled
GPU peak use/minimum free memory, offload/buffer sizes and technical errors.
Separate load and template/tokenization time from request wall time. Retrieval
is replayed and untimed; TTFT is not measured for non-streaming requests.
`tokens_cached` after completion is not prefix reuse; use `timings.cache_n`.
Report raw lengths, phase times and phase throughput together. A shorter
refusal can be faster without being better. No global speed ranking from raw
wall time or statistically significant superiority from six related questions.

**Rubric before generation.** The private rubric and 24 blank review cards
cover correctness against supplied evidence, claim-level support, citation
identifier validity, citation relevance, quotation fidelity, unsupported
claims, refusal, clarification, usefulness, completeness and language
consistency. Keep these dimensions separate; no composite quality score.
Identifier lookup and exact quotation matching are deterministic observations;
claim entailment, relevance, correctness and usefulness require Maria's review.
An uncited claim may be supported elsewhere but still lack a required citation;
a valid citation ID or verbatim quote alone proves neither relevance nor
support. No direct quotation is required. Preserve uncertain/not-assessable
and not-applicable states separately from unreviewed nulls.

**Human review.** Approve all twelve context targets first. After the separately
authorized run, prepare an opaque-ID packet hiding model/runtime/performance
metadata, with unmodified response text and its exact evidence. Review the
24 individual responses in four sessions of six, then twelve same-input pairs
with balanced randomized left/right positions. Commit individual judgments
before pair comparison. Model identity may leak through style or prior
familiarity: record it; blinding is partial and the preparer knows the key.
Review claim/source spans and quotation candidates; record reasons and
confidence. Preserve first/revised judgments and unresolved disagreements;
with one reviewer, do not claim inter-rater agreement. Lock judgments before
unblinding and reveal performance last. Compare models within each cutoff;
compare contexts within each model. Link evidence by record/text identity,
not the local citation label across different payloads.

**Validity and stops.** Accept a completed technical dataset only with 24
fresh, uncached, hash-preserved requests, verified templates/capacity/offload,
adequate sampled memory and complete naturally terminated responses. Missing
citations, an incorrect decision prefix or unsupported content are quality
findings, not grounds for secretly rerunning or discarding an otherwise valid
response. Resource use, grounding and Maria's usefulness judgments are reported
separately. Stop the batch on integrity/input drift, unsupported formatting,
unexplained token/count/configuration differences, insufficient capacity,
CPU/KV fallback, OOM or memory margin failure, context shift/truncation,
prefix reuse, timeout/HTTP error, output-limit termination, missing completion
diagnostics or an incomplete response. Save the partial run and stop its
server/monitor; no automatic retry, evidence edits or parameter changes.

**Planning decision (subsequently approved for this execution).** The proposal
was to approve the private expectation/rubric snapshot, including
its flagged source-reading reassessment, common native-token endpoint, settings,
24-row order and Maria's review workflow, then separately authorize execution.
Maria approved these before generation; the former draft was preserved and
an approved private expectation version recorded. Human judgments remain pending.
If expectations need revision, create a new private version before generation.
Any later discovered target error must preserve the old target and be applied
symmetrically after review, never tuned to favor an output. No model adoption,
provider implementation, new Golden Set labels or M3/M4 execution is authorized
by this preparation. M1/M2-A results and original baseline instructions remain
unchanged. This follows PD-006, PD-007 and PD-008, using the roadmap's
Observation → Interpretation → Decision → Implementation process.

### M3: grounding and reliability analysis

- **Research question:** how reliable are the M2 answers, and which checks can
  be automated?
- **Evaluated:** citation presence and validity; whether citations support
  the claims; exact quotation fidelity; unsupported statements; appropriate
  refusals; appropriate clarification requests; language consistency; answer
  usefulness.
- **Method:** deterministic checks, such as citation identifiers that exist
  and quotations that occur verbatim in the supplied passages, are reported
  apart from human judgments. A syntactically valid citation does not show
  that a claim is supported, and automatic checks do not replace human
  evaluation.

### M4: interpret results and decide next steps

Possible outcomes, none of them approved in advance: continue with the
baseline model; investigate another local model; test a different prompt or
context strategy; introduce deterministic output validation; investigate
model-independent grounding controls. No provider abstraction or model switch
follows unless the evidence supports it and the product owner approves it.

### Open questions for this phase

M2-B execution is complete; its 24 responses await Maria's human review.
Since 1 November 2026 this review is optional ([PD-009](product-decisions.md#pd-009-evaluate-without-mandatory-manual-annotation)); the questions below stay
open without blocking further work.

1. What do the blinded individual and matched-pair reviews show about claim
   support, citations, refusal, clarification, completeness and usefulness?
2. Which judgments remain uncertain or need a recorded revision before the
   model mapping is revealed?
3. After review, does the evidence justify another experiment or keeping the
   current baseline? No model or implementation decision is automatic.

### Future provider choice

A user may later choose the model provider in the application: a locally
hosted model, a local inference service or a remote API-based model, as
described in [Product Direction](product-direction.md#model-choice). This is a
future capability, not part of this phase. No provider settings interface,
authentication or API key management, universal inference adapter, remote API
integration, multi-user infrastructure or billing is planned.

Open questions:

- **Compatibility:** providers differ in chat templates, context limits,
  structured output, reasoning controls and tokenization; which differences
  matter for grounded answers?
- **Privacy:** a remote provider would receive selected passages and
  questions; how does the user see and approve what leaves the device?
- **Costs:** how are per-request costs or local resource needs shown to the
  user?
- **Capabilities:** how does FREDL handle a provider that cannot follow the
  citation, refusal or clarification contract?
- **Configuration:** what is the minimum a user must configure, and how are a
  provider's settings recorded for reproducible comparison?

## Possible multi-user direction

Shared and organization-managed libraries are a possible future direction,
reconsidered only after the personal learning loop has evidence. They are not
a milestone, and FREDL remains local-first.

Requirements that such a direction might bring:

- personal and shared libraries;
- team-level document ownership;
- per-user access to documents and retrieval results;
- isolation of private information;
- retrieval and inference services that scale independently.

No authentication, authorization, organizations, tenancy, billing,
distributed services, orchestration or concurrency optimization is planned;
FREDL initially targets one user asking one question at a time
([PD-007](product-decisions.md#pd-007-evaluate-performance-as-one-users-sequential-question-workflow)). Current design choices
should avoid assumptions that would permanently prevent this direction, for
example by keeping document identity and provenance explicit, without adding
infrastructure for it now.

## Provisional decisions

Recorded on 16 October 2026 by the product owner. They guide planning only and
do not show that any technology or configuration is suitable.

1. **Human judgments:** a small, purposefully selected judged subset is
   acceptable for preliminary comparisons; the complete judged B1 baseline
   remains unfinished. (Superseded as a requirement on 1 November 2026 by
   [PD-009](product-decisions.md#pd-009-evaluate-without-mandatory-manual-annotation); a judged subset remains optional.)
2. **Latency:** about 10–20 seconds for ordinary answers and preferably no
   more than 30 seconds for complex multi-book answers are aspirations, not
   acceptance thresholds. (Since 17 October 2026 these figures are kept only
   as historical provisional targets, not product requirements; see
   [PD-007](product-decisions.md#pd-007-evaluate-performance-as-one-users-sequential-question-workflow).)
3. **Reference hardware:** start with the Dell Precision 7560, verify its
   actual hardware and runtime during Investigation 1, and consider
   consumer-grade hardware later. Product performance targets are not tied to
   this one machine. (Investigation 1 identified the machine as a Dell
   Precision 7550; see Investigation 1 above.)
4. **Order:** start with inference performance diagnostics. Grounded-answer
   reliability remains important and may be investigated independently where
   appropriate.

## Open questions

1. Which questions form the preliminary judged subset, and which judgment
   protocol applies to it? (No longer required since 1 November 2026;
   see [PD-009](product-decisions.md#pd-009-evaluate-without-mandatory-manual-annotation).)
2. When should a CPU-only or consumer-grade hardware profile be measured?
3. Which reference question is used for routine performance comparisons? Its
   selection criteria are recorded in
   [PD-007](product-decisions.md#pd-007-evaluate-performance-as-one-users-sequential-question-workflow);
   the question itself is not chosen.

The question of which hardware and runtime the reference laptop provides was
answered by Investigation 1.
