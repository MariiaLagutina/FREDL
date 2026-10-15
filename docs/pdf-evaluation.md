# PDF Evaluation and CLI

## Implemented boundary

These are local experimental utilities, not the production learning flow.
Each executable golden dataset, record snapshot and retrieval run belongs to
one book. BM25, pinned CPU MiniLM and hybrid retrieval can be evaluated with
human record judgments; answerability has separate runs and metrics.

A separate [multi-book golden contract](#multi-book-golden-contract) with human
relevance judgments is implemented and tested. It has no retrieval runner or
metrics yet: no command searches the combined library or scores a multi-book
run.

The general `index` and API indexing path use text/code ingestion and do not
dispatch to the PDF pipeline. For PDFs, use the `pdf_*` commands below. They
preserve physical-page provenance and verified structural context; they do not
provide OCR, formula interpretation, a PDF reader or personal learning memory.
See [Technical Foundation](technical-foundation.md) for extraction limits,
P0/P1/P2 corpus policies, section-interval context and fingerprint layers.

## Commands

Run from the repository with the locked environment installed (`make install`).
Python Fire accepts positional arguments or their named flags. Inspect current
help before using private inputs:

```bash
uv run python -m src --help
uv run python -m src pdf_run_retrieval --help
```

| Command | Required arguments, in order | Purpose |
| --- | --- | --- |
| `pdf_snapshot_records` | `library_root book_path output` | Extract one book and save canonical records with legacy and v2 fingerprints. |
| `pdf_migrate_judgments` | `source_judgments source_snapshot source_dataset target_dataset library_root output_dir` | Check historical labels, map records, and produce a migration report and review queue. |
| `pdf_run_retrieval` | `library_root dataset retriever output` | Save rankings, neighbor context and a compatibility manifest. |
| `pdf_evaluate_retrieval` | `run dataset judgments library_root output` | Validate current inputs and corpus eligibility, then score the stored run. |
| `pdf_compare_runs` | `first second` | Compare stored retrieval manifests and report identity differences and environment warnings. |
| `pdf_run_answerability` | `retrieval_run dataset library_root model_file output` | Validate a stored retrieval run and classify its passages through an already-running loopback server. |
| `pdf_evaluate_answerability` | `run dataset output` | Score stored model-only and guard/verifier pipeline decisions separately. |
| `pdf_import_multibook_golden` | `package_dir library_root decisions output_dir` | Verify a reviewed candidate package and explicit decisions, then write an unfrozen multi-book draft and an import report. `--migrated_from` names a version 1 draft to preserve. |
| `pdf_freeze_multibook_golden` | `draft output approved_by approval_note` | Freeze a validated draft under a named approval without changing its content. |
| `pdf_prepare_judgment_pilot` | `golden library_root cases output_dir` | Build an anchor-based pool for selected questions and a blind labelling worksheet. |
| `pdf_collect_judgments` | `golden library_root pool worksheet reviewer output` | Validate a filled worksheet against its pool and write human judgments; `--status` defaults to `pilot`. |
| `pdf_migrate_judgment_pilot` | `v1_golden v2_golden library_root old_pool labels labels_sha256 reviewer output_dir` | Regenerate a pilot pool for a version 2 draft and move its human labels unchanged. |

`book_path` is a canonical library-relative POSIX path such as
`books/example.pdf`; `library_root` points to the library outside the checkout.
Source books, golden sets, judgments, model files and outputs stay outside Git.
PDF commands do not add files to Git or publish results.

## Retrieval and migration

`pdf_run_retrieval` accepts `bm25`, `minilm` or `hybrid`; its default `top_k` is
5 and its default `policy` is P2, `exclude_navigation_and_boilerplate`.
P0 `all_records` and P1 `exclude_navigation` remain explicit comparison options.
MiniLM and hybrid require the configured encoder weights/runtime; unit tests
use deterministic fake encoders rather than measuring model quality.

Interface example with placeholder paths and an already-reviewed single-book
dataset (not the Task 6 candidate package):

```bash
uv run python -m src pdf_snapshot_records \
  --library_root=/path/to/private-library \
  --book_path=books/example.pdf \
  --output=/path/to/private-evaluation/records.json

uv run python -m src pdf_run_retrieval \
  --library_root=/path/to/private-library \
  --dataset=/path/to/private-evaluation/golden.json \
  --retriever=bm25 \
  --policy=exclude_navigation_and_boilerplate \
  --top_k=5 \
  --output=/path/to/private-evaluation/retrieval-run.json
```

Evaluation requires judgments compatible with that dataset and current record
set. Migration accepts a historical snapshot and source/target golden datasets.
It transfers only unique byte-identical or whitespace-only matches with checked
pages, section path and unchanged case/evidence. Other mappings require review.
Use `--review_decisions=/path/to/decisions.json` after reviewing the queue;
judgments are not written while unresolved labels lack decisions. Retaining
historical records is explicit, not guessing a new relevance label.

Snapshot and run writers require new output files. Report and migration writes
accept an existing file only when its contents are identical; different contents
are not overwritten. Prepare parent output directories first (migration creates
its output directory). Use separate output names for distinct runs.

## Compatibility and metrics

Manifests bind the exact golden-file SHA256, record-set fingerprint, corpus
policy/searchable text and retrieval settings. Even changing golden JSON
whitespace changes its file hash. Input drift is rejected, not silently scored.
Code revision, dirty state and package versions produce reproducibility warnings
outside run identity; matching fingerprints do not prove equal execution.
Comparison reports differences; it does not make different corpora or retrievers
directly comparable merely by printing their scores.

Required-record recall and MRR describe reviewed records. Evidence-unit coverage
counts a record only for the evidence unit where it is required; ranked top-k
coverage and coverage with eligible, touching same-interval context are distinct.
Unjudged records are not irrelevant. Page overlap is a diagnostic proxy, not
proof of required evidence or a substitute for relevance judgments.

## Answerability

The experimental configuration is Qwen3-4B Q4_K_M with thinking disabled,
served by llama.cpp outside repository dependencies. The default server URL is
`http://127.0.0.1:18080/v1/chat/completions`. `--server_url`, `--model_name`,
`--runtime` and `--server_args` describe the operator's existing local server;
the command does not launch it or verify its loaded weights. `model_file` is
hashed for the run manifest. Default token limits are 3 for the binary decision
and 1 for the comparison verifier; temperature is 0.

Passage policy v1 supplies original record text in ranked-top-k-then-context
order, deduplicated. Boilerplate masked for search can therefore still appear in
model-visible passages. The manifest records this policy, the supported context
rule, retrieval identity, model-file hash, prompt, guard and runtime settings.

Model output is `ANSWERABLE` or `NOT_FOUND`; complete and partial source support
both count as answerable. The subject guard and model comparison verifier are
reported separately from the model's own decision. False acceptance and false
refusal are separate errors. These metrics do not measure generation correctness
or user usefulness, and the binary protocol does not independently handle the
multi-book review's ambiguity or extraction-limitation classes.

## Multi-book golden contract

A separate schema, `fredl.pdf.multibook-golden` version 2, describes a private
multi-book evaluation set; the single-book format above is unchanged. It pins
the library by PDF hash and record-set fingerprint and keeps original-source
support apart from extracted-text support. Expected outcomes are `complete`,
`partial`, `ambiguous`, `unsupported`, `extraction_limited` and
`source_incomplete`; the last three stay distinct, with an explicit limitation
reason where the text or the source itself falls short. Evidence units point
at exact anchors on physical pages. Sufficient evidence is OR across sets and
AND within a set: a question is supported when every unit of at least one set
is supported. Sets must be minimal, so a set that contains another set is
rejected. Negative questions may list related units that are near the topic
but are not sufficient evidence; related units are context, not judged
targets. Reference notes are kept for reviewers and excluded from metrics.

Each question also records `expected_behavior`, kept separate from its
outcome. `answer` means the wording is clear enough to answer from evidence,
which may still be partial or unsupported. `clarification_required` means
several materially different meanings remain open: FREDL should ask which one
the user means instead of choosing. Each meaning is an interpretation with its
own label, clarified question, outcome, support and evidence sets; evidence is
never shared between interpretations. FREDL does not guess what the user wants
to learn. This is an evaluation reference, recorded in
[PD-005](product-decisions.md#pd-005-ask-for-clarification-instead-of-guessing-the-users-meaning);
no FREDL component asks for clarification yet.

The library corpus applies one corpus policy to each book separately and
requires record identities to be unique across books. Import verifies the
package checksums and explicit review decisions and always writes a draft.
Freezing needs a named approval and a clarified question for every
interpretation, and the frozen file must keep the draft's content fingerprint.
A frozen set is changed only through a new version.

### Relevance judgments and pooling

A human reviewer labels one record against one evidence unit of one question:
`required`, `supporting` or `irrelevant`. The same record can be required for
one unit and irrelevant for another. For a required or supporting record,
`supplies_evidence` is `yes` when the extracted text itself provides the
content and `no` when the record only points at content that its text does not
carry, such as a figure caption or a flattened formula. Irrelevant records take
no such value.

A judgment pool lists every record offered for review and why: overlap with an
evidence anchor, the anchor's physical page, or, once retrieval pooling runs,
the BM25, MiniLM or hybrid top-k. Pool order follows questions, units and
reading order, never a ranking. The worksheet hides these sources, so the
reviewer cannot tell which retriever found a record, and it names the
interpretation above each unit of an ambiguous question. A judgment set binds
the golden content, library and pool fingerprints; its status is `pilot`,
`draft` or `frozen`, and a frozen set needs a judgment for every pooled entry.
An unjudged record is never treated as irrelevant. A relevant record from a
book other than its unit's produces a warning, not an error.

These judgments are the basis for multi-book evidence-unit coverage. A unit is
located when a required record for it is retrieved, and supplied only when that
record also supplies the evidence; a question's evidence is complete when every
unit of one sufficient set is supplied. These multi-book metrics are planned
and not yet implemented.

### Schema migration

Version 1 drafts stay readable through a separate read-only reader, so drafts
and pools bound to their content fingerprints remain verifiable. Importing with
`--migrated_from` records the version 1 dataset and content fingerprint and
checks that every version 1 question, outcome, support value, limitation,
evidence set, evidence unit, library identity and package provenance is
unchanged; only the new fields may differ.

A schema change alters the golden content fingerprint, so a pool must be
regenerated. `pdf_migrate_judgment_pilot` checks the labels file hash and the
library fingerprint, regenerates the pool, and requires the same entries in the
same order with the same sources. It then moves the labels unchanged by
question, unit and record. If any check fails, nothing is written. None of the
multi-book commands runs retrieval or a model, and none assigns a relevance
label.

## Current review state and unmeasured work

PDF Tasks 1–5 are implemented. The multi-book contract, import, freeze rules,
judgment pooling and label migration above are implemented and covered by
synthetic tests.

Completed private work:

- 22 human-approved Task 6 candidates have been imported as an unfrozen version
  2 draft, with two questions requiring clarification;
- a judgment pilot covers four questions with different evidence patterns,
  using anchor-based pools only, and its labels moved unchanged from the
  version 1 draft to version 2.

Planned and not yet implemented: a runner that searches the combined library,
a command that pools its top-10 results (the pooling function itself exists and
is tested), multi-book retrieval metrics with per-book and cross-book slices,
an answerability context policy that labels each passage with its book and
physical page, and public aggregate reporting.

B0 Slice 1 adds typed query plans and separate version 1 multi-book run and
manifest contracts, with synthetic compatibility tests. Query identities keep
the original question separate from each approved clarified interpretation.
Configuration/comparison identities are distinct from results fingerprints;
validation checks pinned inputs, ranking rules and direct neighbor provenance.
These contracts do not execute retrieval, pooling or metrics. The three planned
multi-book retrieval commands are not implemented yet.

The approved future primary metric is Best Evidence Recall (`BER@5`), with
`BER@10` diagnostic: the greatest supplied-unit coverage of any sufficient set.
Localization, sufficient-set success and `MRR@10_localization` /
`MRR@10_supply` remain separate. These metrics are not implemented in Slice 1.

Not yet performed: retrieval runs over the library, full human labelling,
freezing the Golden Set or its judgments, and any multi-book retrieval or
answerability measurement.

No new private retrieval or answerability baselines have been established after
the corpus and evaluation-contract changes. Historical single-book results in
[PD-002](product-decisions.md#pd-002-require-direct-evidence-before-treating-retrieval-as-an-answer)
remain historical and are not directly comparable with pending baselines.
Retrieval runs, Golden Set freezing and model evaluation each require separate
approval.
