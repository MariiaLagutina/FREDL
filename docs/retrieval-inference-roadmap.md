# Retrieval and Inference Roadmap

> **Status: Proposed / Not Yet Approved for Implementation.** This document
> lists investigations, not commitments. Investigation 1 has been completed as
> a diagnosis; Investigations 2–6 have not started. Each needs its own
> approval, its acceptance criteria are defined before it runs, and its result
> may be to keep the current solution.

This roadmap belongs to the current PDF evaluation work described in the
[README](../README.md#1-establish-the-next-pdf-evaluation). It does not change
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
| Controlled experiment | What changes and what stays fixed |
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

Investigation 1 is the first planned technical investigation. Investigations
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

- **Hypothesis:** another local embedding model may retrieve technical text
  better, accept longer input, or offer a better quality/performance trade-off.
- **Current baseline:** pinned `all-MiniLM-L6-v2` on CPU, 256 word pieces.
- **Controlled experiment:** compare candidates with the same corpus, chunking
  and queries; a preliminary judged subset is acceptable (see
  [Evaluation scope](#evaluation-scope)).
- **Measurements:** supported input length, technical-text retrieval quality,
  CPU and GPU speed, memory, model size, indexing cost, offline operation and
  compatibility with the chunking strategy.
- **Acceptance criteria:** not defined.
- **Impact:** semantic index rebuild and new Hybrid rankings. No model is
  selected or installed by this roadmap.

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

## Next research phase: model comparison

The next phase compares a small number of alternative local LLMs against the
current Qwen3-4B baseline, as agreed in
[PD-008](product-decisions.md#pd-008-treat-qwen3-4b-as-a-baseline-and-keep-model-providers-replaceable).
Comparisons keep equivalent retrieved evidence and evaluation conditions and
report grounding, citations, quotation fidelity, abstention, clarification,
latency and resource use. Candidate models, the evaluation set and acceptance
criteria are not yet chosen, and the phase needs its own approval. Performance
comparisons follow
[PD-007](product-decisions.md#pd-007-evaluate-performance-as-one-users-sequential-question-workflow).

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
   remains unfinished.
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
   protocol applies to it?
2. When should a CPU-only or consumer-grade hardware profile be measured?
3. Which reference question is used for routine performance comparisons? Its
   selection criteria are recorded in
   [PD-007](product-decisions.md#pd-007-evaluate-performance-as-one-users-sequential-question-workflow);
   the question itself is not chosen.

The question of which hardware and runtime the reference laptop provides was
answered by Investigation 1.
