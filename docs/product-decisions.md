# Product Decision Log

This log records product decisions that change what FREDL presents to the
reader or treats as learning content. Each entry preserves the reason for the
decision, the evidence available at the time, and the conditions that may
justify revisiting it.

Implementation history belongs in Git. This document records product meaning.

## PD-001: Represent a passage through its verified section and exact PDF page

**Date:** 12 October 2026
**Status:** accepted

### Context

A PDF may expose several page identities. Printed numbering can begin after
the cover and front matter, while a PDF viewer addresses physical pages from
the beginning of the file. An embedded outline may also include navigation
labels such as `Cover` and `Copyright`. These labels help navigate the file but
do not represent knowledge the reader wants to retrieve.

A chapter and section path is more meaningful in the interface than a page
number alone. FREDL still needs the physical PDF page because every retrieved
passage must lead back to an exact source location.

### Decision

- Present the verified chapter and section path as the primary structural
  context for a retrieved passage.
- Preserve the one-based physical PDF page as the exact navigation and
  provenance coordinate.
- Show a printed book page only when its relationship to the physical PDF page
  has been verified.
- Preserve embedded outline entries for extraction audit, but classify
  `Cover` and `Copyright` as non-content. They do not become section metadata
  and do not enter RAG retrieval content.
- Treat an outline destination as a page hint. Confirm its heading against
  extracted text and layout before assigning the section to a passage.
- Keep unresolved or ambiguous headings explicit instead of inventing a
  section boundary.

### Evidence

The first active technical book contains 195 embedded outline entries.
Layout-aware matching confirmed all 193 content headings. The remaining two
entries, `Cover` and `Copyright`, describe file navigation rather than learning
content.

The experiment also exposed several PDF-specific cases: headings split across
visual lines, omitted whitespace in extracted text, ligature characters,
chapter markers stored separately from titles, and a heading repeated as the
first word of the following paragraph. Text, font size, and fragment position
were all needed to resolve these cases without guessing.

### Consequences

- PDF chunks need a verified section path alongside physical-page provenance.
- Reader links use physical PDF pages even when the interface also displays a
  printed page number.
- Outline labels alone are insufficient evidence for retrieval metadata.
- Front matter may remain available for document audit without polluting the
  reader's knowledge search.

### Revisit when

Revisit this decision when representative books lack usable outlines, use page
labels that cannot be mapped reliably, or contain meaningful learning content
under a label currently classified as non-content.

## PD-002: Require direct evidence before treating retrieval as an answer

**Date:** 12 October 2026
**Status:** accepted

### Context

Lexical and semantic retrieval always return the closest passages, even when a
book does not answer the question. Similarity scores, score margins, and
agreement between BM25 and MiniLM did not separate answerable questions from
hard negatives in the first private PDF evaluation set.

The local Qwen3-0.6B model correctly classified 13 of 16 answerability cases.
It produced two false refusals and one false acceptance. Qwen3-1.7B, tested
with the same retrieved context and binary prompt, correctly classified 15 of
16 cases after generation was limited to the exact decision label. It accepted
all 12 answerable or partially answerable questions and rejected three of four
unsupported questions.

The remaining failure asked for advantages of React over Vue. The retrieved
passages mentioned React but did not discuss Vue or provide a comparison. The
model treated topic overlap as evidence for a relationship the book did not
state.

### Decision

- Keep retrieval ranking separate from answerability. A high similarity score
  does not prove that a passage answers the question.
- Use Qwen3-1.7B as the current local experimental answerability model. It is
  an evaluated baseline, not a production guarantee.
- Treat both complete and partial source support as answerable. A partial
  answer must preserve its limitation instead of becoming a full claim.
- Return `not found` when passages contain only related terms without the
  requested fact, relationship, comparison, or implementation detail.
- For a comparison, require material evidence about every compared subject and
  an explicit basis for comparison. Mentioning only one subject is not enough.
- Keep answerability output machine-checkable and limited to `ANSWERABLE` or
  `NOT_FOUND` before answer generation begins.
- Keep books, retrieved passages, questions, and evaluation cases local during
  this experiment.

### Evidence

The private evaluation contains 16 questions: 10 completely answerable, two
partially answerable, and four hard negatives. Hybrid retrieval plus adjacent
same-section context covers every labelled evidence unit for the 12 answerable
questions.

With Qwen3-1.7B, the binary answerability experiment reached 93.75% accuracy,
100% answerable acceptance, 75% not-found rejection, and 87.5% balanced
accuracy. An additional relation-specific comparison verifier and an abstract
negative example did not change the remaining React/Vue error. The model still
treated two parallel framework examples as evidence for a comparative claim.
Further prompt tuning on this single case was rejected as golden-set overfit.

The model remains insufficiently validated for automatic production refusal
because the dataset contains only four negative cases from one book.

### Consequences

- FREDL needs an answerability stage after retrieval and before grounded answer
  generation.
- False acceptance and false refusal remain separate reported metrics.
- Model size and prompt changes must be compared on the same versioned golden
  set without changing expected labels after seeing a prediction.
- A generated answer cannot mark itself useful or correct. Source support and
  later user feedback remain separate concerns.

### Revisit when

Revisit the model choice or decision rule after the evaluation includes more
books, more unsupported relationships, tables and figures, or when a different
local model materially improves not-found rejection without increasing false
refusals. A larger model should be tested only against the same frozen cases
and reported with both error types.
