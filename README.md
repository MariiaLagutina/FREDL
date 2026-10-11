# FREDL

**FREDL helps you return not only to what you read, but to why it mattered.**

FREDL is a personal learning system for people whose PDF libraries grow faster
than they can read them. It is built around the questions that bring someone
to a book, the reason a question mattered at that moment, and what changed
after the idea was used in practice.

> [!IMPORTANT]
> FREDL is at an early research and development stage. A manual prototype under
> founder-led testing and a verified retrieval foundation exist today. PDF
> ingestion, the production frontend, and the learning-memory system described
> below are active plans, not finished product features.

## Why FREDL

Technical and scientific books are rarely read in a straight line. Sometimes
someone reads an author from cover to cover. More often, a real question leads
to one chapter, one explanation, or one piece of code because a project needs
an answer now.

Months later, the important context is easy to lose:

- Where did this idea come from?
- Why was I looking for it?
- Which project needed it?
- What did I implement?
- Did it work?
- What would I do differently now?

Most reading tools preserve documents, highlights, or completion. FREDL is
designed to preserve the reader's path from a question to an experience.

Its central memory is not an isolated quote. It is the relationship between a
question, **why the question mattered**, the supporting source, and what the
reader later learned from applying it.

## The Product Idea

FREDL is intended to connect two searchable spaces:

1. **The library** — user-provided PDFs, their topics, exact passages, and the
   relationships between explanations in different books.
2. **The learning memory** — the user's questions, reasons, notes, returns,
   decisions, and later reflections.

The long-term experience is designed around a simple loop:

```mermaid
flowchart LR
    Add["Add personal PDFs"] --> Explore["Explore a book and its topics"]
    Explore --> Ask["Ask a real question"]
    Ask --> Compare["Compare grounded passages"]
    Compare --> Save["Save the question and why it matters"]
    Save --> Apply["Use the idea in practice"]
    Apply --> Return["Return to the earlier understanding"]
    Return --> Reflect["Keep, revise, branch, or close the question"]
    Reflect --> Ask
```

The user remains in control. FREDL may retrieve, compare, suggest, and ask, but
only the user decides whether a question is resolved or their understanding has
changed.

## What Returning Means

A return is not a quiz and it is not a generic spaced-repetition card. FREDL
brings back the user's own context:

- the original question;
- why it arose at that moment;
- the book and exact supporting passage;
- what the user previously thought;
- what happened after the idea was applied;
- any new question that grew from it.

The first prototype tests six lightweight reactions:

- still think the same;
- understand it better;
- think differently now;
- want to reread the source;
- found the answer;
- no longer interested.

There are no reading percentages, mandatory streaks, or penalties for not
returning. Different books and subjects need different rhythms. The first
experiment lets the user choose when to return; adaptive timing remains a
hypothesis to test later.

## Manual Prototype

The current manual prototype is version 0.8. It explores the product behavior
before a production frontend is built. It includes a small personal library,
saved thought cards, `Why`, `Born from`, branching questions, return history,
statistics, and export.

![Manual prototype library with books and questions to return to](docs/assets/readme/manual-prototype-library.png)

*The library view keeps books and open questions together without treating
completion as the goal.*

![Manual prototype return queue](docs/assets/readme/manual-prototype-return-later.png)

*The return queue shows the original question and why it mattered.*

![Manual prototype thought history](docs/assets/readme/manual-prototype-thought-history.png)

*A thought keeps its history and can branch into a new question.*

The prototype does not yet include topic directions, side-by-side passages from
different books, source preference feedback, or the complete “then and now”
view. It must not be confused with the future production application.

### Founder-led validation

The prototype is currently tested only by FREDL's founder, using her own books,
questions, and engineering work. The purpose is to learn whether returning to
the original question and its context is genuinely useful before investing in
the complete application.

A positive result will be a signal to continue, not proof that the approach
works for everyone. A negative result will be equally valuable because it can
change the product before the prototype becomes infrastructure. Wider testing
will be considered only after the founder experiment produces evidence worth
carrying forward.

## Current Project State

| Area | Current state |
| --- | --- |
| Product concept | Defined and being refined through use |
| Manual prototype | Version 0.8 in founder-led self-testing |
| Retrieval foundation | Implemented and measured on a supplied vLLM corpus |
| Code and Markdown ingestion | Implemented |
| PDF ingestion and page provenance | Planned next; not implemented |
| Production frontend | Not implemented |
| Learning-memory storage | Designed conceptually; not implemented |
| Adaptive resurfacing | Research hypothesis; not implemented |
| External model integrations | Future direction; not implemented |

FREDL is also part of its founder's own engineering development. She is using
the prototype to study from technical books while building the retrieval,
memory, and product systems that the project requires. Its portfolio value
comes from that documented work: real decisions, implementation, measurement,
and revision as the evidence changes.

## Retrieval Foundation Available Today

The existing Python system began as a 42 retrieval project and now provides a
tested technical foundation for FREDL. It currently works with a supplied vLLM
code and documentation corpus, not with a user's PDF library.

| Capability | Current implementation |
| --- | --- |
| Lexical retrieval | Custom two-field BM25 with bounded reranking |
| Exact evidence | Project-relative paths and character ranges |
| Code ingestion | AST-aware Python chunking |
| Text ingestion | Structure-aware Markdown and plain-text chunking |
| Optional semantic search | Pinned CPU MiniLM embeddings |
| Optional hybrid search | Weighted Reciprocal Rank Fusion for documentation |
| Answer generation | Local `Qwen/Qwen3-0.6B` with citation validation |
| Interfaces | Python Fire CLI and local FastAPI server |
| Persistence | Corpus, pipeline, schema, and checksum validation |

Retrieval is evaluated separately from generation. The measured foundation
keeps exact source text separate from ranking metadata, rejects stale or
incompatible indexes, and treats an invalid generated citation as an error
rather than presenting an unsupported answer.

### Verified retrieval results

| Measure | Result |
| --- | ---: |
| Full index, 20,096 documents | 29.4 s |
| Retrieval normalized to 200 questions | 19.51–19.88 s |
| Documentation Recall@5 / Recall@10 | 0.850000 / 0.900000 |
| Code Recall@5 / Recall@10 | 0.787879 / 0.848485 |
| Optional documentation hybrid Recall@5 / Recall@10 | 0.870000 / 0.910000 |
| Incremental reindex after one changed file | 7.1 s |

These results describe the existing vLLM evaluation corpus. They do not predict
PDF retrieval quality. PDF extraction, page-level provenance, and evaluation on
real book questions require their own evidence.

## Product Principles

- **The question belongs to the user.** A model may suggest that something is
  understood, but it cannot close or rewrite the question without confirmation.
- **Context matters more than collection.** FREDL preserves why a question was
  asked and what happened next, not only what was highlighted.
- **Sources stay visible.** Retrieved evidence must lead back to the exact book
  location. Generated explanations must be visually separate from source text.
- **Feedback is intentional.** A written or spoken reflection is stronger than
  a quick marker; a quick marker is stronger than passive behavior. Time on a
  page never proves attention or understanding.
- **Adaptation stays transparent.** A preference or suggestion should explain
  the events behind it and remain correctable by the user.
- **Learning stays non-linear.** FREDL preserves unchanged ideas, dead ends,
  abandoned questions, and changes of mind without forcing a universal path.
- **Models remain replaceable.** Future external model integrations should be
  selected and connected by the user, with an explicit boundary around the
  context they receive.

## Development Direction

The next work is deliberately narrower than the complete product vision.

### 1. PDF ingestion and provenance

- extract useful text from technical PDFs;
- preserve page identity and stable source coordinates;
- separate retrieval chunks from reader-facing passages;
- build an evaluation set from real questions about the founder's books;
- measure lexical, semantic, and hybrid retrieval instead of assuming a winner.

### 2. First production user flow

- add a PDF;
- see an overview of the book and its topics;
- ask a question;
- compare exact passages;
- save the question, its reason, and its source;
- return to it and record what changed.

### 3. Learning memory

- store questions, intents, topics, sources, and events as separate entities;
- keep explicit notes, quick markers, and passive observations distinguishable;
- derive source preferences from events rather than storing an unexplained
  score;
- make every inferred preference visible and correctable.

### 4. Later experiments

- compare explanations from different books;
- test calm, adaptive resurfacing without streaks or engagement pressure;
- explore alternative explanations through examples, diagrams, images, and
  step-by-step reasoning;
- connect user-selected model providers;
- expose selected library and memory capabilities through MCP;
- generate practical exercises from verified context.

These are hypotheses and planned experiments, not release promises.

## Running the Current Retrieval Foundation

The commands below run the existing vLLM retrieval system. They do not start a
FREDL PDF application.

Requirements: Python 3.10 or later, `uv`, the supplied vLLM corpus, and the
question datasets.

Install the locked environment:

```bash
make install
```

Prepare the expected local data directories:

```bash
scripts/prepare_evaluation_inputs.sh
```

Place evaluator-provided files under `data/`:

```text
data/
├── raw/
│   └── vllm-*/
└── datasets/
    ├── AnsweredQuestions/
    │   ├── dataset_docs_public.json
    │   └── dataset_code_public.json
    └── UnansweredQuestions/
        ├── dataset_docs_public.json
        └── dataset_code_public.json
```

Build the index and search one question:

```bash
uv run python -m src index --max_chunk_size 2000
uv run python -m src search "How does prefix caching work?" --k 5
```

Generate a grounded answer with the local model:

```bash
uv run python -m src answer "How does prefix caching work?" --k 5 --device auto
```

Run the local HTTP API:

```bash
uv run uvicorn 'src.api:create_app' --factory --host 127.0.0.1 --port 8000
```

Datasets, indexes, models, reports, and generated outputs remain outside Git.

### Development commands

```bash
make run          # show the CLI
make debug        # run the CLI through pdb
make test         # run pytest
make lint         # run assignment-compatible lint checks
make lint-strict  # run flake8 and strict mypy
make clean        # remove local Python check caches
```

## Documentation

The current technical documents describe the verified retrieval foundation.
They will be reviewed and reframed as FREDL architecture develops.

- [`docs/architecture.md`](docs/architecture.md) — current retrieval component
  boundaries and data flows.
- [`docs/decision-log.md`](docs/decision-log.md) — reconsidered technical
  decisions and measured trade-offs.
- [`docs/end-to-end-run-log.md`](docs/end-to-end-run-log.md) — reproduced
  commands, timings, memory use, and retrieval scores.
- [`docs/answer-quality-checklist.md`](docs/answer-quality-checklist.md) —
  grounded-answer review criteria.
- [`docs/bm25-tuning-log.md`](docs/bm25-tuning-log.md) — lexical, semantic, and
  hybrid retrieval experiments.

## Verification

The retrieval foundation was last verified with:

```text
pytest: 579 passed
flake8: passed
mypy with assignment flags: passed for 239 source files
mypy --strict: passed for 239 source files
git diff --check: passed
```

Expected third-party deprecation warnings currently come from Starlette's
AnyIO compatibility alias and Python Fire's coroutine inspection.

## License

FREDL is available under the [MIT License](LICENSE).
