# PDF Evaluation and CLI

## Implemented boundary

These are local experimental utilities, not the production learning flow.
Each executable golden dataset, record snapshot and retrieval run belongs to
one book. BM25, pinned CPU MiniLM and hybrid retrieval can be evaluated with
human record judgments; answerability has separate runs and metrics.

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

## Current review state and unmeasured work

PDF Tasks 1–5 are implemented. A private Task 6 package contains 22 human-approved
candidates with evidence and review history. It is not a frozen Golden Set,
record-level judgments, or an executable multi-book benchmark. The current
single-book schema cannot represent its OR-of-AND source alternatives and
separate source/text limitations unchanged.

No new private retrieval or answerability baselines have been established after
the corpus and evaluation-contract changes. Historical single-book results in
[PD-002](product-decisions.md#pd-002-require-direct-evidence-before-treating-retrieval-as-an-answer)
remain historical and are not directly comparable with pending baselines.
Multi-book contract design, record relevance review, Golden Set freezing and
evaluation execution require separate approval.
