# PDF Evaluation and CLI

## Implemented boundary

These are local experimental utilities, not the production learning flow.
The historical golden datasets, record snapshots and evaluation commands each
belong to one book. BM25, pinned CPU MiniLM and hybrid retrieval can be evaluated
with human record judgments; answerability has separate runs and metrics.

A separate [multi-book golden contract](#multi-book-golden-contract) with human
relevance judgments is implemented and tested. A global multi-book retrieval
runner now searches the pinned eligible library. Deterministic retrieval
pooling produces blind human-review worksheets. An interpretation-aware
collector and retrieval metrics evaluator are implemented with synthetic tests;
no private multi-book baseline has been measured.

The general `index` and API indexing path use text/code ingestion and do not
dispatch to the PDF pipeline. For PDFs, use the `pdf_*` commands below. They
preserve physical-page provenance and verified structural context; they do not
provide OCR, formula interpretation, a PDF reader or personal learning memory.
See [Technical Foundation](technical-foundation.md) for extraction limits,
P0/P1/P2 corpus policies, section-interval context and fingerprint layers.

## Code layout

Historical single-book evaluation modules stay directly under
`src/evaluation/pdf/`. Multi-book modules are organized by responsibility under
`src/evaluation/pdf/multibook/`:

- `library.py`: the library corpus that pins several books and applies the
  corpus policy to each one;
- `curation/`: the golden-set schema, its read-only version 1, candidate
  import, the judgment pilot with its pooling, and their commands;
- `retrieval/`: query plans, run manifests and global retrieval over the
  pinned library;
- `pooling/`: pooling of saved runs into blind review packages;
- `scoring/`: human judgment collection and multi-book retrieval metrics.

Tests mirror these responsibilities under `tests/evaluation/pdf/multibook/`, in
`curation/`, `retrieval/`, `pooling/` and `scoring/`. Tests that run several
responsibilities together, such as retrieval, pooling and scoring through the
CLI, live in `tests/evaluation/pdf/multibook/integration/`. Moving a module
changed its import path only; command names, arguments, artifacts and
fingerprints stay the same.

## Architecture Refactoring — October 2026

### Motivation

As the multi-book evaluation subsystem grew, 23 multi-book modules and 19 of
their test modules sat in the same flat directories as the single-book code.
Each name was descriptive, but finding one responsibility meant scanning many
files that shared the `multibook_` prefix. The refactoring addressed that
navigation and maintenance cost. It was not a response to a known defect in
retrieval, pooling or scoring.

### Decision

Modules are grouped by the responsibility they carry, in the order data flows
through an evaluation: `library → curation → retrieval → pooling → scoring`.
The library pins the books and applies a corpus policy to each; curation
defines and imports the golden set and runs the judgment pilot; retrieval
searches the library and records runs with their provenance; pooling turns
saved runs into blind review tasks; scoring collects the human labels and
computes metrics. The package list is in [Code layout](#code-layout).

Imports point the other way: a package may import from packages before it in
this order, never from packages after it. The historical pilot pooling lives in
`curation/` because it serves the golden set and its pilot labels, not the
retrieval pooling that follows a run. Historical single-book modules stay
directly under `src/evaluation/pdf/` and import nothing from `multibook/`.

### Mirrored tests

Tests under `tests/evaluation/pdf/multibook/` use the same packages, so the
tests for a module are found by the same path. Tests that run several
responsibilities together live in `integration/`. Shared helpers sit in the
package whose objects they build and are imported only from the same or a later
package; no test module imports another test module. Each test module binds the
fixtures it uses explicitly, and there is no global `conftest.py`, so
restrictive fixtures such as the ones that forbid retrieval or model loading
apply only where a module opts in. The contract tests for `library.py`
currently live in `curation/test_golden_contracts.py`.

### Migration

The move took five commits. Each left the whole test suite passing.

1. `d49cd5368b11805fcd4b991bf63080c84f7ef852`: moved shared helpers out of test
   modules, so that no test module imports another.
2. `6e82c7778570fb0fcbdf4336c0d010a204f96dc2`: scoring.
3. `94b2ef7a3148315a8d8f970705d55b1714efb7ff`: pooling.
4. `61cc26f35b2194ad9d321e6d26a42aab496f90eb`: retrieval.
5. `095e667351e7e0afe02b8264e63df10eee7f2831`: curation and the library.

Packages moved from the last layer to the first, so each step changed imports
only in the moved package and in packages already moved. No compatibility
modules were left at the old paths: a patch applied to such a module would not
reach the module that actually runs the code.

### Preserved behavior and verification

The refactoring was intended to change module paths only. Against the layout
before the move, these checks passed:

- 1213 tests pass, including 442 PDF and 332 multi-book tests, with none
  skipped; test IDs, including parameters, are the same once paths are mapped,
  and every test uses the same fixtures;
- the 16 PDF commands are unchanged, and their `--help` output is
  byte-identical;
- all 61 string monkeypatch targets in the tests resolve, and those that patch
  module functions point at the module that calls them;
- production modules have the same syntax tree once import statements are
  excluded; five test modules changed in structure when helpers were extracted
  in the first step;
- fingerprint regression tests pass, and the version 1 and version 2 golden-set
  and pilot judgment contracts are unchanged;
- the import graph has no cycles and no import against the layer order, and no
  single-book module changed;
- strict lint and privacy checks pass, and a full test run with external
  network connections blocked made no connection attempts.

These checks show that behavior is preserved as far as the existing tests
cover it. They do not measure retrieval quality or performance, and no private
evaluation was run.

### Known limitations

The move did not create these issues; it made some of them easier to see.

- Five imports cross module boundaries to reach private functions, such as
  book-path resolution in PDF ingestion.
- Retrieval pooling still uses shared pieces kept in curation modules: locator
  verification in the importer, and the judged-unit list and labelling guide in
  the pilot modules.
- The scoring command imports a run loader from the pooling command module.
- Importing any multi-book module loads the historical `src/evaluation/pdf/`
  package initializer, which imports the single-book baselines and their
  dependencies.
- Some curation modules are large, notably the importer, but each remains
  mostly about one task.

GitHub CI has not yet run on these commits, because the branch is not yet
published.

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
| `pdf_collect_multibook_judgments` | `golden library_root runs inputs output` | Validate pinned human CSV labels and write a new interpretation-aware draft artifact. |
| `pdf_evaluate_multibook_retrieval` | `golden library_root runs inputs output` | Revalidate the same human inputs and score complete ranked/context bindings at 5 and 10. |
| `pdf_pool_multibook_judgments` | `golden library_root runs output_dir` | Pool three compatible global runs, verified anchors and direct context into blank review tasks; `--depth=10`. |
| `pdf_run_multibook_retrieval` | `library_root golden retriever output` | Search the entire pinned library globally; save full rankings, top-5/top-10 context provenance and a versioned manifest. |
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

## Global multi-book retrieval

`pdf_run_multibook_retrieval library_root golden retriever output` accepts
`bm25`, `minilm` or `hybrid`, `--query_mode=original|clarified|all` (default
`all`), and `--policy` (default P2, `exclude_navigation_and_boilerplate`).
It uses exactly the books pinned in Golden Set v2, checking PDF hashes and
record-set fingerprints before encoding. It neither discovers other books nor
uses evidence anchors, outcomes or reference notes as query text.

Each baseline builds one global index from all eligible records and aligned
searchable text. BM25 statistics and semantic rankings cover the combined
corpus; Hybrid fuses the global rankings with the existing RRF parameters.
There are no book quotas. MiniLM uses its pinned CPU configuration with
`local_files_only=True`; a missing local model fails without a download.

The version 1 run stores complete available rankings and scores, Hybrid source
ranks, and direct context with ranked-parent identities separately at 5 and 10.
Context stays within a book, flow and known section interval, requires touching
eligible records, and does not expand recursively. BM25's empty ranking records
zero lexical matches; it is not an unsupported-answer decision. Original
ambiguous questions and clarified variants remain separate queries.

Outputs require an existing directory and a new filename. The command validates
the complete run, writes a same-directory temporary file, flushes it, and
publishes it with an exclusive hard link. Existing files and symlinks are not
replaced; failed writes/publication clean up the temporary file. The filesystem
must support hard links. Configuration identity, shared comparison identity and
results fingerprint are distinct from the SHA256 of the exact UTF-8 output
bytes, returned in the command summary.

This runner is covered by synthetic tests with fake encoders and generated
PDFs. It does not compute multi-book relevance metrics, collect judgments or
establish cross-book retrieval quality. Private runs require separate approval.

## Global retrieval judgment pooling

`pdf_pool_multibook_judgments golden library_root runs output_dir --depth=10`
reads stored runs without searching or loading models. `runs` is a JSON
manifest with `schema_name: "fredl.pdf.multibook-pooling-inputs"`, `version: 1`
and a `runs` array. Each entry has `retriever` (`bm25`, `minilm` or `hybrid`),
`path` and the exact file `sha256`. Relative paths resolve beside the manifest.
Exactly one run of each kind is required; duplicate references, changed bytes,
invalid fingerprints or incompatible inputs fail. Golden file bytes, current
PDF hashes, records, corpus, query plans, context protocols and evidence page
hashes/anchor offsets are verified. Environment diagnostics do not affect
comparison identity. These checks cannot prove scores without rerunning an
engine; pooling never reruns it.

Depth is 5 or 10, matching the run's independently validated context cutoffs.
The candidate union contains each retriever's top-depth records, its direct
context and required/supporting anchor overlaps. Anchors can remain reviewable
when a corpus policy excludes them from search; no other excluded neighbors
are added. No book quotas or automatic relevance labels apply. All origins
remain in `judgment-pool.json`: ranked origins retain per-retriever rank
and score; context origins retain ranked parents and cutoff. Context is never
given an artificial rank or score. Input run order does not affect output: queries follow the query
plan, units their case/interpretation reference order, records canonical
source order, and run provenance BM25/MiniLM/Hybrid order followed by anchors.

The separate version 1 `fredl.pdf.multibook-retrieval-pool` contract keeps
candidates by `(case_id, interpretation_id | null, record_id)` and review tasks
by `(case_id, interpretation_id | null, unit_id, record_id)`. Original
ambiguous questions retain separate conditional-unit tasks with their meaning
shown; clarified variants receive only their own sufficient-set units. Evidence
is never combined across meanings. Related-unit anchors contribute candidates
for their own query variant, but related units are not judgment targets.
Queries without target units retain a candidate inventory without labels.

The new output directory requires an existing parent. It contains
`judgment-pool.json`, blind `worksheet.md`, blank `worksheet.csv`,
`pool-summary.md` and a completion `package-manifest.json` with exact output
file hashes and the semantic pool fingerprint. Within each query/unit group,
worksheet order uses the versioned `blind-review-order-v1` identity hash
(query, unit, book and record), independent of rank, provenance, pool
fingerprint and source order. Worksheets omit retrievers, scores, anchor status
and expected outcomes; each excerpt keeps its record ID and original text,
using a code fence longer than any source backtick run.
Review the worksheet separately from provenance. Blank labels remain unjudged.
Pooling computes no metrics. The historical pilot format, collector and labels
are unchanged; the old `pdf_collect_judgments` command does not accept this
query-aware sheet. Use the separate interpretation-aware collector below.

All files are written into a private same-parent staging directory and their
exact bytes are checked before one atomic no-replace directory rename publishes
the complete package. Existing paths, including empty directories and symlinks,
are refused; concurrent callers cannot replace or mix packages. This requires
Linux libc/kernel `renameat2(RENAME_NOREPLACE)` and filesystem support. An
unsupported platform/filesystem fails without a check-then-rename fallback.
Ordinary failures clean staging; abrupt process termination may leave a hidden
staging directory. No fsync is performed: atomic visibility does not guarantee
crash durability. Consumers must check the completion manifest's hashes.
Synthetic verification does not establish private retrieval quality.

## Human collection and multi-book retrieval metrics

Both new commands take `golden library_root runs inputs output`. `runs` is the
Slice 3 run manifest. `inputs` is read-only JSON with
`schema_name: "fredl.pdf.multibook-judgment-inputs"`, `version: 1`,
`pool: {"path": "judgment-pool.json", "sha256": "<exact hash>"}`,
`pool_fingerprint`, and a `worksheets` array of `{path, sha256, reviewer}`.
Relative pool/worksheet paths resolve beside this manifest. Every exact file
hash is checked; pool/provenance is regenerated against the pinned Golden Set,
current corpus, run identities, rankings and both context cutoffs. No retrieval
or model execution occurs. Source runs' exact hashes and fingerprints are kept.

`pdf_collect_multibook_judgments` writes a new version 1
`fredl.pdf.multibook-collected-judgments` draft. CSV files may partition the
worksheet rows; together they must contain every exact item and binding once.
Missing/unknown rows, malformed cells, duplicated or conflicting bindings and
hash drift are errors. Original and clarified bindings never share labels.
Accepted relevance cells are `required`, `supporting`, `irrelevant` or blank;
supply cells are exactly `yes`, `no` or blank. Blank relevance requires blank
supply. Irrelevant requires blank supply. A relevant label with blank supply
is retained as incomplete. Notes and source file bytes are not rewritten.
There is no automatic pilot migration, label carry-over or freeze path.

`pdf_evaluate_multibook_retrieval` independently recollects those pinned CSVs
in memory, then writes a version 1 `fredl.pdf.multibook-retrieval-evaluation`.
It requires complete labels for every unit/record binding contributing to
any run's ranked top-5/top-10 or either direct-context cutoff, including the
original ambiguous diagnostic track. Missing bindings are reported in full;
no final artifact is written. A depth-5 pool that omits top-10 bindings cannot
silently pass. Blank anchor-only rows outside the evaluated retrieval scope
remain unjudged and are retained in the collected draft.

Metric policy `declared-sets-query-macro-v1` uses the Golden Set's declared
sufficient sets without modality heuristics. Within each query variant:

- `BER@5` is primary; `BER@10` is diagnostic. For each sufficient set, count
  distinct units supplied by at least one retrieved record, divide by set
  size, then take the best set's coverage. Supply policy
  `required-and-explicit-yes-v1` requires both `required` relevance and explicit
  human `yes` for that exact unit binding. `supporting` never closes a unit,
  even with `yes`; its human label remains preserved unchanged.
- Sufficient-set success is true only when at least one complete AND-set is
  supplied. Alternative sets are OR, never a combined denominator.
- Localization coverage uses the same best-set calculation but requires
  `required` relevance, independently of supply. A located caption/formula
  with supply `no` cannot count toward BER.
- `MRR@10_localization` and `MRR@10_supply` use the reciprocal of the first
  ranked top-10 record meeting the respective criterion for any eligible unit.
  No hit gives zero; no sufficient set gives null. Context never gets a rank.

`ranked_only` and `with_context` are separate tracks at both cutoffs;
`context_only_gain` is their difference. Exact context parents remain in the
report. MRR is recorded once per query/engine and is unaffected by expansion.
Original and clarified results are separate artifact lists. Original ambiguous
queries retain conditional-unit diagnostics, but have no case-level sufficient
set and therefore null coverage/success/MRR. Unsupported and source-incomplete
queries similarly contribute null, not failed-retrieval zeros.

Every aggregate records its metric denominator, query/interpretation/case
counts and excluded identities. Means weight query variants equally. Headline
averages include only complete original/clarified queries. Partial and
extraction-limited queries use their declared sets in separate outcome slices;
their scores concern those sets, not full-question answerability. Track and
outcome slices expose other denominators explicitly.

Artifacts bind Golden/corpus/library/query/comparison identities, pool and
judgment fingerprints, run hashes, pool file hash and human-input hashes.
Evaluation environment/code diagnostics are excluded from its semantic
fingerprint; exact input hashes still identify the consumed bytes. Reordering
valid CSV rows preserves decisions and metric values, but changes byte hashes
and therefore artifact provenance/fingerprints. Outputs
require new filenames and existing parent directories, with exclusive hard-link
publication as in Slice 2. Existing outputs are refused; no crash durability is
promised. These synthetic checks establish implementation behavior only, not
private retrieval quality, answerability or learning-loop usefulness.

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
question, unit and record. If any check fails, nothing is written. The import,
freeze and judgment preparation/collection/migration commands do not run
retrieval or models and do not assign relevance labels automatically.

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

Planned and not yet implemented: multi-book answerability evaluation,
an answerability context policy that labels each passage with its book and
physical page, and public aggregate reporting.

B0 Slice 1 adds typed query plans and separate version 1 multi-book run and
manifest contracts, with synthetic compatibility tests. Query identities keep
the original question separate from each approved clarified interpretation.
Configuration/comparison identities are distinct from results fingerprints;
validation checks pinned inputs, ranking rules and direct neighbor provenance.
These contracts do not execute retrieval, pooling or metrics. B0 Slice 2 adds
the [global retrieval runner](#global-multi-book-retrieval); Slice 3 adds
[retrieval judgment pooling](#global-retrieval-judgment-pooling). Slice 4 adds
[human collection and retrieval metrics](#human-collection-and-multi-book-retrieval-metrics),
verified synthetically only.

The primary retrieval metric is Best Evidence Recall (`BER@5`), with
`BER@10` diagnostic: the greatest supplied-unit coverage of any sufficient set.
Localization, sufficient-set success and `MRR@10_localization` /
`MRR@10_supply` remain separate. Their implementation has synthetic coverage;
private baseline measurement remains pending.

Not yet performed on the private library: retrieval runs, full human labelling,
freezing the Golden Set or its judgments, and any multi-book retrieval or
answerability measurement.

No new private retrieval or answerability baselines have been established after
the corpus and evaluation-contract changes. Historical single-book results in
[PD-002](product-decisions.md#pd-002-require-direct-evidence-before-treating-retrieval-as-an-answer)
remain historical and are not directly comparable with pending baselines.
Retrieval runs, Golden Set freezing and model evaluation each require separate
approval.
