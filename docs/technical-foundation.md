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

The PDF pipeline has been verified on one text-based technical book. It
records:

- the source document identity and neutral book-card metadata;
- one-based physical page numbers and positioned text fragments with font
  metadata;
- raw extracted text, kept alongside the prepared reading text;
- an observable classification of every page as text, text with images,
  image-only, possible vector graphics, or empty; non-text pages are routed
  separately instead of entering the reading text;
- repeated page margins, removed from reading text while remaining auditable;
- words and sentences reconstructed across page boundaries;
- embedded-outline headings confirmed against extracted text and layout, and
  mapped to verified section intervals;
- section-aware bounded chunks with stable identifiers;
- for every chunk, its reading-flow range and the exact source range on each
  contributing physical page;
- index-neutral records that keep text separate from structural metadata.

How a passage is presented, including section paths, physical pages, and
non-content outline entries, is recorded in
[PD-001](product-decisions.md#pd-001-represent-a-passage-through-its-verified-section-and-exact-pdf-page).

### Not yet verified or implemented

- books without a usable embedded outline;
- multi-column layouts, footnotes, tables, formulas, captions, and code-heavy
  pages as distinct structures;
- scanned or image-only books, which would require OCR;
- the relationship between a retrieval chunk and the passage a reader sees in
  a production interface.

These cases must be tested on representative books rather than treated as
ordinary paragraphs.

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

The evaluation measures:

- required-record recall and reciprocal rank for BM25, MiniLM, and hybrid
  retrieval;
- evidence-unit coverage, which accepts one required record per labelled
  evidence item;
- coverage after adding adjacent records from the same section;
- answerability, with false refusals and false acceptances reported
  separately.

The first evaluation covers one book and 17 questions, five of them hard
negatives. Its results and limits are recorded in
[PD-002](product-decisions.md#pd-002-require-direct-evidence-before-treating-retrieval-as-an-answer).

### Not yet measured

- retrieval and answerability across several books and document classes;
- whether a returned passage was useful in the product flow.

Recall and ranking metrics can diagnose retrieval. User feedback must remain a
separate signal: a technically relevant passage may still be poorly explained,
come from a disliked source, or arrive at the wrong time.

## Answerability

Retrieval always returns the closest passages, even when a book does not answer
the question. The planned answerability stage sits after retrieval and decides
whether the passages directly support an answer before any answer is
generated. In the current experiment its output is limited to `ANSWERABLE` or
`NOT_FOUND`, and its prompt is versioned.

The current experimental baseline is a local Qwen3-4B model in Q4
quantization, served by llama.cpp outside FREDL's dependencies. It is an
evaluated experiment, not part of a reading flow and not a production
guarantee.

## Privacy and model boundary

PDFs are supplied by the user and should remain under the user's control. The
PDF pipeline and the answerability experiment run locally; book text,
retrieved passages, and evaluation cases are not sent to an external service.
A model integration must be replaceable and explicit about which text it
receives. The learning-memory system should store the user's question, reason,
actions, and concise reflections. It should not duplicate whole books as
memory.

## Next implementation slice

Ingesting a technical PDF, preserving page-level provenance, and searching it
with real questions are verified on one book. Before building on them, the
pipeline is tested on several more books with different layouts to expose the
failures a single book cannot reveal, and the next golden set is built from
questions across those books.

The rest of the milestone remains intentionally small:

1. show the exact supporting passage;
2. save the question and why it mattered;
3. return to that record later;
4. record whether the source or explanation helped.

This slice is sufficient to test the central product loop before adaptive
timing, provider integrations, richer explanations, or a mobile application.
