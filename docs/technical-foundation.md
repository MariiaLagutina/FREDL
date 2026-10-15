# Technical Foundation

## What exists

The current Python code provides reusable building blocks for ingestion,
retrieval, grounded generation, persistence, validation, and a local API.
It supports Python, Markdown, plain text, and text-based PDFs. It does not yet
provide the production learning interface.

Reusable components include:

- structure-aware text chunking;
- AST-aware Python chunking;
- a local PDF ingestion pipeline with exact physical-page provenance;
- a custom two-field BM25 index;
- optional semantic and hybrid retrieval;
- exact source spans and source-path provenance;
- persisted indexes with fingerprints and checksums;
- bounded context construction;
- replaceable model configuration;
- citation validation and answer caching;
- PDF retrieval and answerability evaluation utilities;
- a CLI and local FastAPI boundary.

The repository keeps evaluation utilities because retrieval quality must be
measured independently from answer style. The old course datasets, evaluator
scripts, submission instructions, and reports are not part of FREDL.

## PDF ingestion

### Implemented

Technical validation covers four text-based books: Beyond Vibe Coding,
AI-Powered Business Intelligence (Early Release), AI Engineering, and
AI and Machine Learning for Coders. They include print and EPUB-reflow layouts;
this is bounded validation, not general PDF support. The pipeline records:

- the source document identity and neutral book-card metadata;
- one-based physical page numbers and positioned text fragments with font
  metadata;
- raw extracted text, kept alongside the prepared reading text;
- an observable classification of every page as text, text with images,
  image-only, possible vector graphics, or empty; non-text pages are routed
  separately instead of entering the reading text;
- repeated page margins, removed while preserving raw line separators and
  remaining auditable;
- words and sentences reconstructed across page boundaries;
- embedded-outline headings confirmed against extracted text and layout, and
  mapped to verified section intervals;
- verified section paths, separate unverified section hints, and explicit
  resolution status and reasons for unresolved boundaries;
- section-aware bounded chunks with stable identifiers;
- for every chunk, its reading-flow range and the exact source range on each
  contributing physical page;
- index-neutral records that keep text separate from structural metadata.

Heading matching handles different heading conventions. For existing outlines
without destinations, a fallback searches complete title lines with display-font
evidence and checks their order against located headings. Repeated titles need
an unambiguous position between located headings. Outline order alone cannot
create a match. Unresolved entries limit verified attribution instead of
silently extending the preceding section.

How a passage is presented, including section paths, physical pages, and
non-content outline entries, is recorded in
[PD-001](product-decisions.md#pd-001-represent-a-passage-through-its-verified-section-and-exact-pdf-page).

### Reliability and support limits

Reliability reports expose heading outcomes and layout findings, including code
overflow, code split by chunks, tables, formulas, footnote markers, and visual
pages. Findings are diagnostic: they neither repair content nor affect retrieval
eligibility or ranking. Detection is heuristic and incomplete.

Prose and heading attribution were checked across the four books. Print-layout
code preserves line text, but chunk splits can degrade it; EPUB-reflow code may
already be clipped in the source. Tables lose column structure, footnotes lose
marker relationships, and formulas can lose subscript and superscript meaning.
Figures and image-only or vector-only content are unsupported without OCR or
visual interpretation. Multi-column reading order and complex structures are
not generally supported. Books with no embedded outline remain unverified.

The mapping from retrieval chunks to passages in a production reading interface
also remains unimplemented. Successful text extraction does not establish
faithful support for every structure on a page.

## Retrieval corpus and context

The corpus classifier identifies content, index, table of contents, and
unsectioned front matter from observable structure. It also locates confirmed
repeated boilerplate. Three reversible policies are available:

- P0 (`all_records`): search all original records;
- P1 (`exclude_navigation`): exclude confidently identified navigation;
- P2 (`exclude_navigation_and_boilerplate`): apply P1 and mask confirmed
  boilerplate in searchable text, preserving offsets and line breaks.

P2 is the current experimental default, as recorded in
[PD-003](product-decisions.md#pd-003-keep-navigation-and-repeated-boilerplate-out-of-experimental-search).
Original records remain unchanged. Uncertain content and unsectioned front
matter stay eligible. Classification heuristics were checked on four books;
they are not a universal content classifier.

Neighbor context requires the same known `section_interval_id` and touching
reading-flow ranges. Equal section paths or hints are insufficient because
different intervals can share them. A record without an interval identity
receives no neighbor context.

## Retrieval evaluation for FREDL

Earlier benchmark results from an unrelated corpus do not establish quality for
books. FREDL uses a small evaluation set made from real questions asked while
using the founder's own library. Questions, labels, and book text stay in the
private library, outside the repository.

### Evaluation data and metrics

Each golden case keeps:

- the original question;
- why it was asked;
- an expected answer;
- whether the book answers it completely, partially, or not at all;
- evidence physical pages with their role and modality;
- descriptive tags such as `hard_negative` or `unsupported_comparison`.

Golden sets are validated strictly and versioned. A label is fixed before any
model run and is never changed to match a prediction; a labelling error is
corrected openly, with its reason recorded. Passage-level relevance judgments
mark index records as required, supporting, or irrelevant and are versioned
separately.

Judgment schema v2 records migration provenance. Snapshots and migration tools
preserve historical inputs; automatic transfer requires a unique byte-identical
or whitespace-only text match, unchanged physical pages and section path,
unchanged golden case, and evidence-page overlap. Other mappings require review.
Migration to golden v2 is complete: 56 labels were preserved, 55 automatically
and one through Maria's manual review. This transfers existing judgments rather
than expanding their coverage: they remain limited to one book, and the added
hard negative has no passage-level judgments.

The evaluation measures:

- required-record recall and reciprocal rank for BM25, MiniLM, and hybrid
  retrieval;
- evidence-unit coverage, which accepts one required record per labelled
  evidence item and counts a record only for units where it is labelled required;
- ranked top-k coverage separately from coverage expanded with eligible
  adjacent context;
- judged-irrelevant and unjudged records in top-k.

Page-overlap probes are diagnostics, not relevance judgments: sharing an
evidence page does not prove that a record contains the required evidence.
Answerability is evaluated separately, with false refusals and false acceptances
reported independently of retrieval metrics.

The first evaluation covers one book and 17 questions, five of them hard
negatives. Its results and limits are recorded in
[PD-002](product-decisions.md#pd-002-require-direct-evidence-before-treating-retrieval-as-an-answer).

### Evaluation contract and reproducibility

PDF records use canonical order by book, reading flow, offsets, and identifier,
so input order does not determine ranking ties or corpus classification.
Versioned canonical-JSON fingerprints distinguish:

- record sets (v2), including behavior-relevant text, structure, and provenance;
- retrieval corpora (v2), including policy, eligible records, and searchable text;
- retrieval configurations, including retriever, top-k, context rule,
  tokenization, encoder revision, and ranking parameters;
- answerability configurations, including model-file hash, runtime, server
  arguments, prompt, guard, and generation settings.

The legacy record fingerprint is retained for historical verification. Run
manifests bind the golden dataset hash, record set, corpus, and configuration;
incompatible inputs are rejected rather than silently rescored. Code revision,
dirty state, and package versions are recorded as reproducibility warnings,
not run identity. Matching fingerprints alone do not prove identical execution.

The `python -m src pdf_*` commands support record snapshots, judgment migration,
retrieval and answerability runs, evaluation, and retrieval-run comparison.
These are local evaluation utilities, not the product learning flow.

### Not yet measured

- new BM25, MiniLM, hybrid, evidence-coverage, and answerability baselines after
  PDF Tasks 1–5; historical runs use an earlier corpus and protocol and cannot
  be compared directly with the pending baselines;
- retrieval and answerability across several books and document classes;
- whether a returned passage was useful in the product flow.

Recall and ranking metrics can diagnose retrieval. User feedback must remain a
separate signal: a technically relevant passage may still be poorly explained,
come from a disliked source, or arrive at the wrong time.

## Answerability

Retrieval always returns the closest passages, even when a book does not answer
the question. An implemented evaluation runner reads a compatible stored
retrieval run and tests whether its passages directly support an answer.
Integration of this stage before answer generation in a product reading flow
remains future work. Model output is limited to `ANSWERABLE` or `NOT_FOUND`.

The versioned passage policy selects ranked top-k records followed by
deduplicated context and supplies original record text, including any boilerplate
masked for search. Its manifest binds the context rule and searchable-text
policy to the retrieval run and answerability configuration. Model-only results
are reported separately from the pipeline's subject guard and model comparison
verifier results.

The current experimental model is Qwen3-4B Q4_K_M with thinking disabled,
served locally by llama.cpp outside FREDL's dependencies. The evaluation client
accepts only loopback server addresses. Historical results do not establish a
new baseline for the current contract or a production guarantee.

## Privacy and model boundary

PDFs are supplied by the user and should remain under the user's control. The
PDF pipeline and the answerability experiment run locally; book text,
retrieved passages, and evaluation cases are not sent to an external service.
A model integration must be replaceable and explicit about which text it
receives. The learning-memory system should store the user's question, reason,
actions, and concise reflections. It should not duplicate whole books as
memory.

## Next implementation slice

Four-book technical validation and the evaluation-contract utilities are
implemented, with the limits above. New baselines remain pending; expanding the
golden set across books is separate future work requiring reviewed questions
and labels. Neither the diagnostic probes nor the completed migration establish
improved retrieval quality or answerability.

The rest of the milestone remains intentionally small:

1. show the exact supporting passage;
2. save the question and why it mattered;
3. return to that record later;
4. record whether the source or explanation helped.

This slice is sufficient to test the central product loop before adaptive
timing, provider integrations, richer explanations, or a mobile application.
