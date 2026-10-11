# Technical Foundation

## What exists

The current Python code provides reusable building blocks for ingestion,
retrieval, grounded generation, persistence, validation, and a local API.
It supports Python, Markdown, and plain text. It does not yet provide PDF
ingestion or the production learning interface.

Reusable components include:

- structure-aware text chunking;
- AST-aware Python chunking;
- a custom two-field BM25 index;
- optional semantic and hybrid retrieval;
- exact source spans and source-path provenance;
- persisted indexes with fingerprints and checksums;
- bounded context construction;
- replaceable model configuration;
- citation validation and answer caching;
- a CLI and local FastAPI boundary.

The repository keeps evaluation utilities because retrieval quality must be
measured independently from answer style. The old course datasets, evaluator
scripts, submission instructions, and reports are not part of FREDL.

## PDF ingestion requirements

PDF support must preserve enough structure for a reader to trust and revisit a
result. The first implementation should record:

- the source document identity;
- page number and stable page coordinates;
- extracted text before retrieval normalization;
- headings, paragraphs, lists, captions, tables, and code where recoverable;
- the relationship between a retrieval chunk and the passage shown to a reader;
- extraction warnings for scanned, malformed, or ambiguous pages.

Scanned documents may require OCR. Multi-column layouts, footnotes, tables,
formulas, and repeated headers must be tested explicitly rather than treated as
ordinary paragraphs.

## Retrieval evaluation for FREDL

Earlier benchmark results from an unrelated corpus do not establish quality for
books. FREDL needs a small evaluation set made from real questions asked while
using the founder's own library.

Each evaluation case should keep:

- the original question;
- why it was asked;
- one or more relevant passages and pages;
- acceptable alternative passages;
- the book type and extraction conditions;
- whether lexical, semantic, or hybrid retrieval found the evidence;
- whether the returned passage was useful in the product flow.

Recall and ranking metrics can diagnose retrieval. User feedback must remain a
separate signal: a technically relevant passage may still be poorly explained,
come from a disliked source, or arrive at the wrong time.

## Privacy and model boundary

PDFs are supplied by the user and should remain under the user's control. A
model integration must be replaceable and explicit about which text it receives.
The learning-memory system should store the user's question, reason, actions,
and concise reflections. It should not duplicate whole books as memory.

## Next implementation slice

The next technical milestone is intentionally small:

1. ingest a technical PDF;
2. preserve page-level provenance;
3. search it with a real question;
4. show the exact supporting passage;
5. save the question and why it mattered;
6. return to that record later;
7. record whether the source or explanation helped.

This slice is sufficient to test the central product loop before adaptive
timing, provider integrations, richer explanations, or a mobile application.

