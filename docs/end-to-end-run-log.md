# End-to-End Run Log

This log records clean, assignment-compatible pipeline runs. Generated
datasets, indexes, model files, and answer outputs remain local and are not
committed to Git.

## 2026-09-13 - Phase 29 clean public-dataset run

### Environment

- Linux development machine
- Python 3.13.15 selected by `uv`
- `uv` 0.12.5
- NVIDIA Quadro RTX 3000 with 6 GiB VRAM for answer generation
- mandatory model: `Qwen/Qwen3-0.6B`
- clean local clone of commit `267ad94`
- corpus, public datasets, Moulinette, and model checkpoint supplied as local
  evaluation inputs outside Git

The clean clone used a new virtual environment and an empty `uv` package
cache. `uv sync --frozen` prepared 63 locked packages in 1 minute 37 seconds.

### Commands

```bash
uv sync --frozen

uv run python -m src index --max_chunk_size 2000

uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions/docs

uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions/code

./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/docs/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000

./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/code/dataset_code_public.json \
  data/datasets/AnsweredQuestions/dataset_code_public.json \
  --k 10 --max_context_length 2000

HF_HOME=/path/to/local/huggingface HF_HUB_OFFLINE=1 \
  uv run python -m src answer_dataset \
  --student_search_results_path \
  data/output/search_results/UnansweredQuestions/docs/dataset_docs_public.json \
  --save_directory \
  data/output/search_results_and_answer/UnansweredQuestions/docs \
  --device cuda --offline

HF_HOME=/path/to/local/huggingface HF_HUB_OFFLINE=1 \
  uv run python -m src answer_dataset \
  --student_search_results_path \
  data/output/search_results/UnansweredQuestions/code/dataset_code_public.json \
  --save_directory \
  data/output/search_results_and_answer/UnansweredQuestions/code \
  --device cuda --offline
```

### Index and retrieval results

The index contained 20,096 documents. Its corpus and pipeline fingerprints
matched the established production baseline.

| Operation | Elapsed | Peak RSS |
| --- | ---: | ---: |
| Index | 21.10 s | 1,246,800 KiB |
| Docs retrieval, 100 questions | 12.12 s | 872,312 KiB |
| Code retrieval, 99 questions | 11.45 s | 872,464 KiB |

Both search-result files passed Moulinette validation.

| Dataset | R@1 | R@3 | R@5 | R@10 |
| --- | ---: | ---: | ---: | ---: |
| Docs | 0.560 | 0.770 | 0.850 | 0.900 |
| Code | 0.535 | 0.707 | 0.788 | 0.848 |

The mandatory thresholds are R@5 `0.80` for Docs and `0.50` for Code. The
sequential retrieval time for all 199 questions was 23.57 seconds, below the
90-second limit for 200 questions.

### Answer-generation results

| Dataset | Completed | Elapsed | Peak RSS |
| --- | ---: | ---: | ---: |
| Docs | 100/100 | 334.84 s | 3,716,452 KiB |
| Code | 99/99 | 287.57 s | 3,716,912 KiB |

Both output files preserved input order, question IDs, question text, and all
ten retrieved sources per question. No answer was empty. The complete batch
generation took 622.41 seconds on this machine.

The run also exposed a model-quality limitation that structural completion
must not hide. Qwen started 95 of 100 Docs answers and 94 of 99 Code answers
with `The sources conflict:`. Only five Docs answers and two Code answers used
the requested `[Source N]` syntax. These observations do not invalidate the
assignment JSON schema, but they show that successful orchestration is not the
same as high answer quality.

### Verification

- focused generation and CLI tests: 54 passed
- complete test suite: 479 passed
- strict flake8 and mypy: passed for 199 source files
- `git diff --check`: passed

### Future comparison

`Qwen/Qwen3-0.6B` remains the mandatory default and every submission workflow
must continue to work with it. After the mandatory release is stable, the same
persisted retrieval results and fixed question set can be used to compare
other free local models. A useful comparison should record answer quality,
citation compliance, conflict frequency, generation time, and peak memory
without changing retrieval at the same time.
