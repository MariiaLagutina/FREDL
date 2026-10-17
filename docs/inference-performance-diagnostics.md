# Inference Performance Diagnostics

> **Research record.** This report contains configuration, hardware and
> aggregate timings only: no question text, model answers, book passages,
> private paths or private fingerprints. Its timings describe one development
> machine and set no product-wide performance requirement.

## Investigation 1: local inference performance (16 October 2026)

### Purpose

Diagnose why the local Qwen3-4B answers in
[B1-2](pdf-retrieval-experiments.md#b1-2-qwen3-4b-grounded-answer-smoke-test-16-october-2026)
and
[B1-3](pdf-retrieval-experiments.md#b1-3-qwen3-4b-with-hybrid-top-5-versus-top-10-16-october-2026)
took 81–331 seconds, as planned in
[Investigation 1](retrieval-inference-roadmap.md#1-local-inference-performance)
of the roadmap. This is diagnosis only: nothing was optimized, rebuilt or
reconfigured.

### Hardware and runtime

Detected on the machine that ran B1-2 and B1-3. This is an experiment
environment, not a product requirement.

| Item | Detected value |
| --- | --- |
| Machine | Dell Precision 7550 |
| Operating system | Ubuntu 26.04.1 LTS, Linux kernel 7.0.0 |
| CPU | Intel Core i7-10850H, 6 cores, 12 threads, up to 5.1 GHz |
| RAM | 30 GiB usable |
| GPU | NVIDIA Quadro RTX 3000, 6144 MiB (5738 MiB visible to CUDA, 5648 MiB free when idle) |
| Driver | NVIDIA 595.91.07, CUDA 13.2 driver API |
| Idle GPU use | 0%, 4 MiB |

The roadmap originally listed the reference machine as a Dell Precision 7560
with an i7-11850H and an RTX A3000. The detected machine, CPU and GPU all
differ; the roadmap now records the detected hardware.

| Item | Value |
| --- | --- |
| Runtime | llama.cpp `llama-server` build 11574 (commit `8bb3cf0d2`, version 0.6.0-dev), prebuilt Ubuntu x64 package with a CUDA 12.8 backend, outside FREDL |
| Backends shipped | CPU variants and `libggml-cuda.so`; CUDA runtime libraries (`libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`) in a separate companion directory |
| Model | Qwen3-4B Q4_K_M GGUF (file type 15), 2.50 GB, SHA256 prefix `7485fe6f` |
| Architecture | `qwen3`, 36 blocks, embedding 2560, 32 attention heads, 8 KV heads, head size 128, trained context 40960 |
| Server arguments | `-ngl 99 -c 16384 --host 127.0.0.1 --port 18080 --reasoning off --reasoning-budget 0` |
| Reported by server | 6 CPU threads, 4 slots, 16384 context per slot, unified KV cache |
| Generation | temperature 0, top-p 1, seed 0, max tokens 700, thinking disabled |

### Finding: the GPU was not used

In the environment used for B1-2 and B1-3, the library search path contained
only the runtime directory, not the companion CUDA runtime directory.

| Check | B1 environment | With the CUDA runtime directory added |
| --- | --- | --- |
| `ldd libggml-cuda.so` | `libcudart.so.12` and `libcublas.so.12` not found | all resolved |
| `llama-server --list-devices` | `(none)` | `CUDA0: Quadro RTX 3000` |

Without a CUDA device, `-ngl 99` had no effect: zero layers were offloaded and
the model ran on the CPU. The server printed no warning. The replay below
confirms it: GPU utilization stayed at 0% and GPU memory at 4 MiB while all
six CPU threads were busy. The cause is the launch environment of these
experiments, not FREDL code and not the model.

### Diagnostic procedure

1. Read the hardware with `lscpu`, `free`, `nvidia-smi` and the DMI product
   name.
2. Inspected the runtime with `llama-server --version`, `--list-devices` and
   `ldd`, with and without the companion directory on `LD_LIBRARY_PATH`. No
   model was loaded for these checks.
3. Read the per-request `timings` that `llama-server` returned in every saved
   B1-2 and B1-3 response (12 requests).
4. Replayed one request, the B1-2 Q1 top-5 prompt, once in the exact B1
   environment. The prompt hash and the reply text matched B1-2. The request
   was streamed to time the first token, while a sampler read the server's
   CPU time and resident memory from `/proc` and GPU use from `nvidia-smi`
   once per second. Raw records are kept privately.
5. Stopped the diagnostic server. No other inference process was running.

### Measurements

Values in the `llama.cpp` columns come from the server's `timings`. Wall time
is the client's request time. Shares are calculated afterwards.

| Run | Question | Prompt tokens | Prefill s | Prefill tok/s | Output tokens | Decode s | Decode tok/s | Wall s | Prefill share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top-5 | Q1 | 3134 | 70.5 | 44.5 | 59 | 10.4 | 5.57 | 80.9 | 87% |
| top-5 | Q2 | 3664 | 82.9 | 44.2 | 327 | 78.3 | 4.16 | 161.5 | 51% |
| top-5 | Q3 | 3191 | 76.1 | 41.9 | 87 | 16.2 | 5.29 | 92.7 | 82% |
| top-5 | Q4 | 3849 | 91.9 | 41.9 | 186 | 42.5 | 4.36 | 134.6 | 68% |
| top-5 | Q5 | 4000 | 95.2 | 42.0 | 73 | 17.1 | 4.20 | 112.6 | 85% |
| top-5 | Q6 | 4950 | 128.0 | 38.7 | 10 | 2.8 | 3.23 | 131.1 | 98% |
| top-10 | Q1 | 5052 | 137.8 | 36.7 | 79 | 22.9 | 3.41 | 160.7 | 86% |
| top-10 | Q2 | 5709 | 141.7 | 40.3 | 321 | 106.1 | 3.02 | 248.0 | 57% |
| top-10 | Q3 | 7657 | 218.7 | 35.0 | 55 | 23.2 | 2.33 | 242.2 | 90% |
| top-10 | Q4 | 7427 | 214.6 | 34.6 | 168 | 68.1 | 2.45 | 283.3 | 76% |
| top-10 | Q5 | 8538 | 251.0 | 34.0 | 175 | 79.5 | 2.19 | 331.0 | 76% |
| top-10 | Q6 | 8027 | 231.6 | 34.7 | 10 | 4.1 | 2.19 | 236.3 | 98% |

Replay of top-5 Q1 in the B1 environment:

| Measurement | Value |
| --- | --- |
| Prefill (llama.cpp) | 3134 tokens in 70.3 s, 44.6 tok/s |
| Decode (llama.cpp) | 59 tokens in 10.4 s, 5.60 tok/s |
| Time to first generated token (client) | 70.3 s |
| Complete answer (client) | 80.6 s |
| Server CPU use (sampled) | mean 600%, maximum 603%: six threads fully busy |
| Server resident memory (sampled) | up to 6.8 GB |
| GPU (sampled) | 0% utilization, 4 MiB throughout |
| Reproducibility | same prompt hash and identical reply as B1-2; timings within 0.5% |

Other timings:

- Server-side overhead outside prefill and decode was below 0.5 s per request
  (wall time minus the two llama.cpp phases).
- Model loading took about 3.4 s at server start; the server stays loaded
  between requests, so loading is not paid per request.
- Prompt construction took under 0.1 s. Loading and extracting the PDF corpus
  inside the one-off experiment script took 35 s; it happens once per script
  run, before any request, and is not part of the request timings above.
- Retrieval latency per query was not measured. The B1-1 runs (38–81 s each)
  include extraction and indexing of the whole library for 27 queries.
- Time to first useful result, such as the first verified source passage, is
  not measurable: no product flow shows passages before the answer.

### Top-5 versus top-10

Calculated from the table: the top-10 prompts were 1.6–2.4 times longer.
Prefill throughput fell from 38.7–44.5 to 34.0–40.3 tokens per second, so
prefill time grew faster than the prompt. Decode throughput fell from
3.2–5.6 to 2.2–3.4 tokens per second, so answers of similar length took
longer to generate. Both effects match CPU attention cost growing with
context, although this cause was not measured separately.

### Bottlenecks

Ranked by estimated impact, with confidence:

1. **No GPU execution (high confidence, largest impact).** The CUDA backend
   could not load its runtime libraries, so no device existed and every layer
   ran on the CPU. Evidence: device list, `ldd`, 0% GPU and 600% CPU during
   the replay.
2. **CPU prefill (high confidence).** At 34–45 tokens per second, prefill took
   51–98% of each answer, with a median near 85%; time to first token equals
   prefill time.
3. **CPU decode (high confidence).** At 2.2–5.6 tokens per second, long
   answers spend up to 106 s decoding.
4. **Prompt size (medium confidence).** Top-10 prompts of up to 8.5k tokens
   magnify both phases. Their size is set by retrieval and context policy,
   which belong to Investigations 2, 4 and 6.
5. **Application overhead (low impact).** Under 0.5 s in the server. PDF corpus
   preparation is significant in the experiment script but outside the
   request; a product flow would need to reuse prepared records.

Not supported by evidence: CPU thread count (all six physical cores were
busy), model loading per request, or VRAM limits (the GPU was never used).

### Recommendations, not implemented

1. **Make the CUDA runtime available to the existing server.**
   - Problem and evidence: no CUDA device; the companion libraries are missing
     from the search path.
   - Change: launch the same build with the companion directory on
     `LD_LIBRARY_PATH`; no rebuild, installation or model change.
   - Expected mechanism: prefill and decode run on the GPU. No speed-up is
     claimed before measurement.
   - Quality effect: same weights and settings, but GPU arithmetic may change
     a temperature-0 reply; compare replies.
   - Resources and risk: estimated, not measured, f16 KV cache for one full
     16384-token context of about 2.25 GiB plus 2.33 GiB of weights, against
     5.6 GiB free, before compute buffers. The fit is uncertain at
     `-c 16384`.
   - Validation and approval: the next experiment below; it changes the
     runtime environment and needs approval.
2. **Size the context to the prompts, if VRAM is short.** The largest B1
   request needed about 9.2k tokens (8538 prompt plus 700 output), so a
   smaller context such as 12288 would still fit it. This reduces KV memory
   without changing answers for prompts that fit. Separate experiment.
3. **Check the slot count.** The server ran four slots for one user. Whether
   `-np 1` changes memory or speed with the unified KV cache is unmeasured.
   Separate experiment.
4. **Quantize the KV cache only if needed.** A q8_0 KV cache would reduce
   memory but may change answers. Only after recommendations 1–2 and with a
   reply comparison.
5. **Reduce prompt size through retrieval and context policy.** Fewer or more
   focused passages lower both phases, but they affect evidence delivery. This
   belongs to Investigations 2, 4 and 6 and should be measured after GPU
   execution works.

Changing the model or quantization is not recommended on this evidence: the
measured slowness follows from CPU execution, not from the model choice.

### Proposed next controlled experiment

Replay the same Q1 prompt at top-5 and top-10 (two requests) with the same
build, model and arguments, adding only the CUDA runtime directory to the
library path. Record:

- the device list;
- offloaded layers and buffer sizes, from load logs at higher verbosity;
- VRAM after loading and during the request;
- prefill and decode timings;
- time to first token;
- CPU and GPU utilization;
- whether the reply text matches B1-2 and B1-3.

Stop if the model does not fit at `-c 16384` and report the failure instead
of changing other settings.

### Limitations

- One replayed request and twelve historical requests, on one machine, one
  model and one build. Per-request variance was not measured beyond one
  repetition.
- GPU behavior, VRAM fit and GPU throughput are unmeasured because the GPU
  was never active.
- KV-cache placement and size were not reported by the server at the default
  log level; the memory figures above are calculations.
- Retrieval latency per query and time to first useful result remain
  unmeasured.
- The B1-2 and B1-3 answer-quality observations were produced with CPU-only
  inference; GPU execution may change temperature-0 replies.

## Investigation 1A: GPU inference validation (16 October 2026)

### Change under test

The same llama.cpp build, model, server arguments
(`-ngl 99 -c 16384 --reasoning off --reasoning-budget 0`) and generation
settings, with one temporary difference: the companion CUDA runtime
directory was added to `LD_LIBRARY_PATH` for this launch only. No shell
configuration, system library, runtime file or FREDL code was changed.

### Loading on the GPU

Before loading, `llama-server --list-devices` reported
`CUDA0: Quadro RTX 3000 (5738 MiB, 5648 MiB free)`, and no other process used
the GPU. The model loaded in about 1.4 s. A separate load-only start at
verbosity level 4, with no requests, recorded the placement:

| Item | Reported by llama.cpp |
| --- | --- |
| Offloaded layers | 37/37: 35 repeating layers and the output layer (log wording) |
| Model buffers | 2375.91 MiB on CUDA0; 304.28 MiB host-mapped |
| KV cache | 2304.00 MiB on CUDA0, f16 K and V, 16384 cells, 36 layers, unified across 4 sequences |
| Compute buffers | 93.01 MiB on CUDA0; 36.02 MiB host |
| Batch | `n_batch` 2048, `n_ubatch` 512; flash attention `auto`; 2 graph splits |
| GPU memory after load | 4772 MiB used by the context, 858 MiB free (`nvidia-smi`: 4874 MiB for the server process) |

The model fits in VRAM at the original `-c 16384` with four slots. The
warning `common_fit_params: failed to fit params ... n_gpu_layers already
set by user to 99, abort` means that automatic parameter fitting was skipped
because the layer count was set explicitly; loading itself succeeded.

### Requests

Two private B1 prompts were replayed sequentially: Q1 Hybrid top-5 and then
Q1 Hybrid top-10. Both prompt hashes matched the B1 records. The server kept
the prompt cache enabled, its default: the top-10 request reused 1682 cached
prompt tokens shared with the preceding top-5 prompt and processed the other
3370. The B1-3 CPU request had no cached tokens, so the top-10 totals are not
like-for-like.

### CPU-only baseline versus GPU

All values are measured unless marked as calculated or estimated. CPU values
are the B1-2 replay for top-5 and the B1-3 record for top-10.

| Measurement | Top-5 CPU | Top-5 GPU | Top-10 CPU | Top-10 GPU |
| --- | ---: | ---: | ---: | ---: |
| Prompt tokens in the request | 3134 | 3134 | 5052 | 5052 |
| Prompt tokens processed (not cached) | 3134 | 3134 | 5052 | 3370 |
| Output tokens | 59 | 59 | 79 | 59 |
| Prefill time | 70.3 s | 2.00 s | 137.8 s | 2.33 s |
| Prefill throughput | 44.6 tok/s | 1567 tok/s | 36.7 tok/s | 1444 tok/s |
| Decode time | 10.4 s | 0.90 s | 22.9 s | 0.95 s |
| Decode throughput | 5.60 tok/s | 64.2 tok/s | 3.41 tok/s | 61.2 tok/s |
| Time to first token (client) | 70.3 s | 2.0 s | not measured | 2.3 s |
| Complete answer (client) | 80.6 s | 2.9 s | 160.7 s | 3.3 s |
| GPU layers offloaded | 0 | 37/37 | 0 | 37/37 |
| GPU memory (sampled) | 4 MiB | 4887 MiB | not sampled | 4887 MiB |
| GPU utilization (sampled, 1 s) | 0% | 18–100% | not sampled | 44–100% |
| Server CPU use (sampled) | about 600% | about 100% | not sampled | about 100% |
| Server resident memory | up to 6.8 GB | 0.81 GB | not sampled | 0.81 GB |

Calculated speedups for top-5 (like-for-like): prefill 35×, decode 11.5×,
time to first token 35×, complete answer 28×. For top-10, the throughput
ratios are 39× for prefill and 18× for decode; the total-time ratio (49×) is
not like-for-like because of the cached prefix and the shorter answer.
Estimate, not measured: an uncached top-10 prefill at the measured 1444 tok/s
would take about 3.5 s, for a complete answer of about 4.5 s.

### Answers

- Top-5: the GPU reply is identical to the CPU reply in B1-2.
- Top-10: the GPU reply has the same decision, the same claims and the same
  single citation as B1-3, but omits B1-3's final uncited sentence. GPU
  arithmetic and the cached prefix are both possible causes; one request
  cannot separate them. Neither version is assumed more correct.

### Remaining bottlenecks

- Inference latency is no longer dominant for these prompts: about 3 s of
  model time per answer, against 81–331 s on the CPU.
- Preparing the PDF corpus took 33–35 s per run of the experiment script. A
  product flow would need prepared, persisted records rather than extraction
  per question. This is application work, not inference.
- Retrieval latency per query, time to first useful result and run-to-run
  variance remain unmeasured.
- VRAM headroom is 858 MiB at `-c 16384` with four slots, which limits larger
  contexts or concurrent requests on this GPU.
- The answer-behavior problems in B1-2 and B1-3 (missing citations,
  unsupported answers, no clarification, an invented quotation) were observed
  with CPU inference and are not addressed by this change.

### Recommended next experiment

Rerun the six B1-2 and six B1-3 prompts on the GPU on a freshly started
server with prompt caching disabled per request, so that each prefill is
uncached. This would measure GPU timings for all twelve prompts and show
whether the B1-2 and B1-3 behavior observations still hold with GPU
inference. It needs separate approval.

### Limitations

Two requests on one machine, one model and one build; no repetition. The
top-10 request benefited from the prompt cache. GPU utilization and memory
were sampled once per second, so short peaks may be missed. These results
describe this machine only and set no product-wide performance requirement.

## Investigation 1B: GPU baseline for all twelve B1 prompts (16 October 2026)

### Configuration

Identical to Investigation 1A: the same build, model, arguments
(`-ngl 99 -c 16384`, four slots), generation settings and temporary
`LD_LIBRARY_PATH` with the companion CUDA runtime directory. A load-only start
at verbosity level 4 again reported 37/37 layers, the KV cache (2304 MiB) and
the model buffers on CUDA0, with 858 MiB free. The timed server was then
started fresh at the default log level.

The twelve private B1 prompts (Q1–Q6 at Hybrid top-5, then at top-10) were
replayed sequentially; every prompt hash matched its B1 record. Each request
set the server's per-request `cache_prompt: false` option, and the server
reported `cache_n = 0` for all twelve, so no prompt prefix was reused. The
script would have stopped on any reuse. No errors or memory pressure
occurred; GPU memory stayed at 4887–4889 MiB.

### Measurements

Prefill and decode come from llama.cpp `timings`; time to first token and
totals are client measurements; speedups are calculated against the B1 CPU
wall times.

| Run | Prompt tokens | Output CPU / GPU | Prefill GPU s (tok/s) | Decode GPU s (tok/s) | First token GPU s | Total CPU s | Total GPU s | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top-5 Q1 | 3134 | 59 / 59 | 1.98 (1582) | 0.91 (63.7) | 1.99 | 80.9 | 2.90 | 27.9× |
| top-5 Q2 | 3664 | 327 / 321 | 2.18 (1679) | 5.11 (62.6) | 2.40 | 161.5 | 7.51 | 21.5× |
| top-5 Q3 | 3191 | 87 / 78 | 1.89 (1689) | 1.22 (63.2) | 2.15 | 92.7 | 3.37 | 27.5× |
| top-5 Q4 | 3849 | 186 / 136 | 2.33 (1652) | 2.19 (61.6) | 2.55 | 134.6 | 4.74 | 28.4× |
| top-5 Q5 | 4000 | 73 / 95 | 2.44 (1639) | 1.53 (61.6) | 2.71 | 112.6 | 4.23 | 26.6× |
| top-5 Q6 | 4950 | 10 / 10 | 3.11 (1592) | 0.15 (58.5) | 3.38 | 131.1 | 3.53 | 37.1× |
| top-10 Q1 | 5052 | 79 / 79 | 3.18 (1587) | 1.31 (59.5) | 3.57 | 160.7 | 4.88 | 32.9× |
| top-10 Q2 | 5709 | 321 / 285 | 3.67 (1556) | 4.85 (58.5) | 4.10 | 248.0 | 8.96 | 27.7× |
| top-10 Q3 | 7657 | 55 / 110 | 5.21 (1470) | 1.96 (55.6) | 5.67 | 242.2 | 7.63 | 31.7× |
| top-10 Q4 | 7427 | 168 / 174 | 5.02 (1479) | 3.08 (56.1) | 5.62 | 283.3 | 8.70 | 32.6× |
| top-10 Q5 | 8538 | 175 / 118 | 5.95 (1434) | 2.14 (54.7) | 6.61 | 331.0 | 8.75 | 37.8× |
| top-10 Q6 | 8027 | 10 / 10 | 5.52 (1453) | 0.17 (53.8) | 6.28 | 236.3 | 6.44 | 36.7× |

| Summary (median, range) | Top-5 | Top-10 |
| --- | --- | --- |
| Total GPU | 3.9 s (2.9–7.5) | 8.2 s (4.9–9.0) |
| First token GPU | 2.5 s (2.0–3.4) | 5.6 s (3.6–6.6) |
| Prefill throughput | 1646 tok/s (1582–1689) | 1474 tok/s (1434–1587) |
| Decode throughput | 62.1 tok/s (58.5–63.7) | 55.9 tok/s (53.8–59.5) |
| Total CPU (B1) | 121.9 s (80.9–161.5) | 245.1 s (160.7–331.0) |
| Speedup | 27.7× (21.5–37.1) | 32.8× (27.7–37.8) |

During every request the sampled GPU utilization reached 99–100% and the
server used about one CPU core. These figures are model inference only. They
exclude PDF preparation, which took 32.7 s once per script run (extraction of
the four books before any request), and retrieval, which used the stored B1
Hybrid run and was not timed. Prompt construction took under 0.1 s.

Without prompt-cache reuse, the top-10 Q1 reply was identical to its B1-3 CPU
reply, so the difference noted in Investigation 1A came from the cached
prefix, not from GPU arithmetic.

### Answer behavior, GPU versus CPU

Assistant observations from citation, quotation and term checks against the
supplied passages; they are not human judgments.

| Question | Identical to CPU | Citations | Exact quotations | Refusal or clarification | Change in substance |
| --- | --- | --- | --- | --- | --- |
| Q1 | top-5 and top-10 | 1, valid | none | none needed | none |
| Q2 | no | 0 at both cutoffs, as on CPU | none | answered without citations, as on CPU | none; wording differs |
| Q3 | no | 1–2, valid | none | none needed | top-10 adds code-level details that occur in the cited passage |
| Q4 | no | 1–2, valid; bracket form throughout | none | answered without the expected evidence, as on CPU | none; the top-5 reply contains a stray non-Latin token inside an English word |
| Q5 | no | 2–3, valid | top-10: 1 of 3 quotations not in any passage, as on CPU | no clarification at either cutoff, as on CPU | none |
| Q6 | top-5 and top-10 | none | none | correct abstention, without detail, as on CPU | none |

Every cited identifier occurred in the supplied context. GPU inference did
not resolve any behavior problem from B1-2 and B1-3: missing citations in
Q2, answering without evidence in Q4, no clarification in Q5 and the
non-verbatim quotation in Q5 at top-10 all remain. It added one new defect,
the stray token in Q4 at top-5.

### Remaining bottlenecks

- Model inference is now 2.9–9.0 s per answer on this machine; prefill grows
  with prompt length, from about 2 s at 3.1k tokens to about 6 s at 8.5k.
- PDF preparation (about 33 s per script run) and retrieval time are outside
  these numbers. A product flow would need persisted records and measured
  per-query retrieval.
- VRAM headroom is about 860 MiB at `-c 16384` with four slots.
- Answer reliability, not speed, is now the dominant problem in the smoke
  test.

### Status of Investigation 1

The diagnostic questions are answered with direct evidence: the cause of the
slow B1 answers (no CUDA runtime on the library path), full GPU offload at
the original configuration, and a twelve-request GPU baseline without cache
reuse. Investigation 1 was closed on 17 October 2026 as a completed
diagnostic investigation, not as a CPU-only or end-to-end performance
validation. Still unmeasured: run-to-run variance (one run per
prompt), per-query retrieval latency, time to first useful result, a
consumer-hardware profile, and whether the provisional 10–20 s and 30 s
aspirations hold once retrieval and preparation are included. The launch
procedure itself is outside FREDL and is not yet documented as a supported
setup.

### Recommended next investigation

Investigation 6, grounded-answer reliability, starting with deterministic
checks that need no new model runs: citation presence and validity, and exact
quotation verification against the supplied passages, evaluated on the 24
stored CPU and GPU replies. It needs separate approval.

### Limitations

One run per prompt on one machine, one model and one build. GPU utilization
and memory were sampled every 0.5 s. The answer comparison uses six questions
and assistant checks, not human judgment. These results describe this
machine only and set no product-wide performance requirement.
