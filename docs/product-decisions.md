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
  or become retrieval content merely through the outline. This classification
  does not automatically exclude the corresponding page text from retrieval;
  text eligibility follows the corpus policy in PD-003.
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

**Validation update, 15 October 2026:** technical PDF validation now covers
four books. An existing outline without destinations can supply verified
section context when its headings are independently located in the page text
and layout. Unresolved headings remain explicit. This does not establish
support for books without an outline or for all PDF layouts.

### Consequences

- PDF chunks need a verified section path alongside physical-page provenance.
- Reader links use physical PDF pages even when the interface also displays a
  printed page number.
- Outline labels alone are insufficient evidence for retrieval metadata.
- Unsectioned front matter remains searchable under the current experimental
  corpus policy; an outline label alone is not grounds for excluding it.

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

Qwen3-4B, quantized to Q4_K_M and served locally through llama.cpp, was then
tested against the same frozen cases, retrieved context, and prompt. Its own
binary decision classified all 16 cases correctly, including the React/Vue
hard negative. It also classified all 17 cases of a second golden-set version
correctly, including a new hard negative that Qwen3-1.7B accepted.

### Decision

- Keep retrieval ranking separate from answerability. A high similarity score
  does not prove that a passage answers the question.
- Use Qwen3-4B Q4_K_M as the current local experimental answerability model.
  It is an evaluated baseline, not a production guarantee. Qwen3-1.7B remains
  the previous baseline for comparison.
- Keep the llama.cpp runtime outside FREDL dependencies during this
  experiment. Adding a model runtime or adapter remains a separate decision.
- Run the answerability baseline with thinking disabled. Use thinking mode
  only as a diagnostic to inspect why the model reached a decision on a
  doubtful case, not as the evaluated configuration.
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

**Historical scope, clarified 15 October 2026:** the results below were
obtained under the earlier single-book corpus and evaluation protocol, before
PDF Tasks 1–5. The original numbers remain historical evidence. They cannot
be directly compared with the pending new baselines, which have not yet been
measured under the current corpus and evaluation contract.

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

With Qwen3-4B Q4_K_M, prompt `pdf-answerability-v1`, temperature 0, thinking
disabled, and a 16K context, the experiment reached 100% accuracy, 100%
answerable acceptance, and 100% not-found rejection, with no false refusals or
false acceptances. The decision was recorded separately for the model, the
deterministic comparison-subject guard, and the comparison verifier. The
model's own decision was already correct for every case; the guard and the
verifier did not change any result. Two consecutive runs produced identical
context and decisions. All 16 cases took about 42 seconds on a 6 GB GPU.

A second golden-set version keeps the 16 original cases unchanged and adds one
hard negative: it asks what a Dockerfile for the book's Express backend should
contain. Docker, containers, Express, and production deployment all occur in
the book, but only for agent sandboxes, backend scaffolding, and general
deployment practice. The label was fixed before any model run. On this
17-case version with five unsupported questions, Qwen3-4B Q4_K_M again reached
100% accuracy with no false refusals or false acceptances, reproduced in two
runs. Qwen3-1.7B, given the identical context, accepted the new hard negative
in two of two runs.

The first formulation of the new case was withdrawn because of a labelling
error. It asked how much AI speeds up senior, mid-level, and junior
developers and was labelled unsupported. A diagnostic run with thinking
enabled exposed a passage the pre-run review had missed: a junior example in
which an endpoint taking a day drops to about an hour. The thinking run itself
still answered `NOT_FOUND` because it misread that example as not tied to a
level, so the reasoning was useful for review, not as a decision. The case
would have been partially answerable, so the model without thinking had
accepted it correctly. The case was replaced before it was used as evidence,
and the withdrawal is recorded with the private evaluation data. Labels are
now reviewed by reading the relevant sections in full rather than by
searching for figure patterns alone.

On the replacement case, thinking mode also returned `NOT_FOUND`, and its
reasoning found no Dockerfile content in any retrieved passage. Thinking
generated about 340 tokens for this decision instead of a single label, and
its reasoning length varied between runs at temperature 0 in an earlier
diagnostic. The model reached the same decision without thinking, which
supports keeping thinking disabled in the baseline.

The 1.7B and 4B results differ in both model size and runtime: the 1.7B model
ran through Transformers, and the 4B model ran quantized through llama.cpp.
The comparison therefore shows a better configuration, not an isolated
effect of model size.

Neither model is validated for automatic production refusal because the
dataset contains only five negative cases from one book. A perfect score on
this set is not evidence of general reliability.

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
and reported with both error types. Revisit the runtime boundary before any
production answerability stage depends on llama.cpp.

## PD-003: Keep navigation and repeated boilerplate out of experimental search

**Date:** 15 October 2026
**Status:** accepted for the current experiment

### Context

Navigation can repeat the terms a reader searches for without explaining them.
Repeated boilerplate can similarly compete with substantive passages. Removing
uncertain material, however, risks hiding useful or unique evidence.

### Decision

- Use P2 (`exclude_navigation_and_boilerplate`) as the current experimental
  policy: exclude confidently identified navigation and mask confirmed repeated
  boilerplate in searchable text while preserving original source text.
- Keep uncertain content and unsectioned front matter searchable. A title or
  an extraction warning alone does not justify exclusion.
- Keep this choice reversible. P0 (`all_records`) and P1
  (`exclude_navigation`) remain available for comparison.

### Evidence

Four-book structural diagnostics identified navigation and repeated boilerplate
that affect search inputs. These diagnostics and page-overlap probes are not
human relevance judgments and do not establish improved retrieval quality or
answerability. New baselines remain pending.

### Consequences

Search may omit confidently identified navigation or ignore repeated boilerplate
without deleting the source. Reliability findings remain diagnostic and do not
by themselves suppress content or change its rank.

### Revisit when

Revisit the policy after measured baselines, reviewed relevance judgments, or
new document types expose lost evidence or unreliable classification.

## PD-004: Show the original PDF page when extracted text cannot carry the evidence

**Date:** 15 October 2026
**Status:** accepted requirement; not implemented

### Context

Text extraction can flatten formulas and tables, drop the content of figures
and diagrams, or lose other structure. Retrieval may still locate the right
record or physical page while its extracted text lacks what the reader needs.
Filling that gap with reconstructed content would look like a source quotation
without being one.

### Decision

- FREDL never reconstructs missing source content and presents it as a
  verified quotation. When extracted text is insufficient for a reliable
  answer, FREDL says so and helps the reader inspect the original source.
- **Phase 1, Open in PDF:** offer an action that opens the original local PDF
  at the exact physical page, preserving book identity and physical-page
  provenance. State clearly when the extracted text is insufficient, and never
  claim the original content was verified when only extracted text was read.
- **Phase 2, original source preview:** show a rendered image of the original
  PDF page inside FREDL, later a focused crop around a formula, table or figure
  once reliable coordinates exist. Use the original PDF rendering, never an
  AI-reconstructed image, and let the reader open the full PDF from the
  preview.
- Rendering stays local, like the PDFs themselves.

### Evidence

The multi-book candidate review contains a reviewed formula case: the original
page shows the complete equation, while the extracted text keeps its symbols
but loses exponents, indices and the fraction structure. Its expected outcome
is `extraction_limited`. Retrieval can locate that page, but the extracted text
alone cannot support an exact answer.

### Consequences

Evaluation keeps three questions apart:

1. whether retrieval located the relevant record or physical page;
2. whether the extracted text supplies sufficient evidence for an exact answer;
3. whether the reader can inspect the original page through source navigation.

Source navigation does not change human relevance judgments, does not make
extracted evidence sufficient, and does not change an `extraction_limited`
outcome. This is a future reading-flow feature, not part of the current
multi-book evaluation, and no viewer, renderer or navigation is implemented.

### Revisit when

Revisit the preview design when layout-aware extraction recovers formulas,
tables or figures reliably enough to verify, when crop coordinates prove
unreliable, or when a real reading session shows that page navigation alone
does not let the reader resolve the evidence.

## PD-005: Ask for clarification instead of guessing the user's meaning

**Date:** 15 October 2026
**Status:** accepted for evaluation; product behavior not implemented

### Context

A technical library can give one short question several valid meanings. The
same word can name a software library and an unrelated technique, and a metric
can have several definitions in different books. Choosing one meaning silently
would answer a question the reader may not have asked, and evidence for one
meaning could look like support for another.

Ambiguous intent is a different problem from insufficient evidence. A clear
question can still be only partially supported, unsupported, limited by
extraction, or incomplete in the source itself. Treating either problem as the
other would hide what actually went wrong.

### Decision

- Record what a question means separately from how well the books support it.
  Intent is `answer` or `clarification_required`; evidence sufficiency remains
  the expected outcome.
- Require clarification only when two or more materially different valid
  meanings remain open within the question and its scope. A question that names
  its book, method, figure, table or measurement is treated as scoped.
- When clarification is required, FREDL should ask which meaning the reader
  intends instead of choosing one, and must not merge evidence from different
  meanings into one answer.
- Evaluate each meaning as its own interpretation, with a clarified question,
  expected outcome and evidence.
- Never use clarification to cover missing evidence, and never present
  insufficient evidence as ambiguity.
- Do not infer why the reader asked or what they intend to learn.

### Evidence

An intent-clarity review of the 22 human-approved multi-book candidates
identified two questions with materially different valid interpretations. This
demonstrated the need to evaluate question intent separately from evidence
sufficiency.

The multi-book golden schema version 2 encodes the decision for evaluation.
Each question records its expected behavior, each meaning of an ambiguous
question is an interpretation with its own evidence, interpretations never
share evidence units, and freezing requires a clarified question for every
interpretation. The draft is not frozen, and no retrieval or model run has
measured it.

### Consequences

- Multi-book retrieval metrics are planned per interpretation for ambiguous
  questions and are reported apart from clear questions. They show whether
  evidence for each meaning is found, not whether FREDL asks.
- The planned multi-book answerability scoring keeps ambiguous questions out of
  its primary binary accuracy.
- No implemented FREDL component detects ambiguity or asks a clarifying
  question. Designing and evaluating that behavior is separate future work.

### Revisit when

Revisit this decision when real reading sessions show that clarification
prompts interrupt more than they help, when ambiguous questions become common
enough to need an ordering of readings, or when a component that asks for
clarification is designed and needs its own evaluation contract.
