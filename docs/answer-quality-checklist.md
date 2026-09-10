# Answer Quality Checklist

## Purpose

This checklist establishes a manual baseline for grounded-answer quality before
any Phase 25 prompt change. It complements deterministic validation: code can
prove the shape of citations, but a reviewer must still judge whether an answer
is useful, coherent, and actually supported by its context.

Do not compare a later prompt version with this baseline unless the same ten
cases, retrieval configuration, model, and generation configuration are used.

## Review rubric

Score every criterion as `pass`, `partial`, or `fail` and write a short reason.

| Criterion | Question for the reviewer |
| --- | --- |
| On-point | Does the answer directly address the question rather than nearby information? |
| Grounding | Is every factual claim supported by the admitted prompt sources? |
| Citation trace | Does each citation identify the source that supports its claim? |
| Coherence | Is the answer concise, readable, and internally consistent? |
| No invention | Does the answer avoid APIs, values, files, or behavior absent from the context? |

For the boundary cases, apply the additional expected behavior exactly. A
deterministic validator pass is necessary but does not by itself make an answer
pass this manual rubric.

## Fixed baseline cases

The first six cases are public vLLM questions. They exercise the entire local
pipeline: retrieval, bounded context, prompt construction, and generation.

| ID | Kind | Question | Additional review focus |
| --- | --- | --- | --- |
| D1 | Exact documentation fact | What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM? | Endpoint spelling and directness |
| D2 | Documentation command | What command can be used to evaluate the accuracy of a quantized model using lm_eval with vLLM? | Complete command and no invented flags |
| D3 | Documentation comparison | What are the differences between mm_kwargs and tok_kwargs when using the _call_hf_processor method in vLLM multimodal processing? | Clear comparison without mixing concepts |
| C1 | Exact code default | What's the default value of trust_remote_code in vLLM's LLM class constructor? | Exact literal value |
| C2 | Code constants | What are the default values for FP8_MIN and FP8_MAX constants in vLLM's triton_flash_attention module? | Both values and correct attribution |
| C3 | Code conditions | What conditions must be met for vLLM's ModelRunner to use CUDA graphs instead of the regular model? | Complete conditions, not a vague summary |

The corresponding stable question identifiers are:

```text
D1  17526382-7764-4120-b5e8-2d3726b8a4da
D2  cc83c230-099f-4c11-aeab-8c09715c5942
D3  81e37f26-1203-40ad-a848-764d187d083f
C1  0d29b0e8-686c-41b7-9a48-50f8e4bc63de
C2  02789461-05fd-435d-9673-91606102d6c9
C3  6ffdf5f6-fcd3-4d11-8227-8bedd9b33296
```

The remaining four are controlled generation-boundary cases. They use small,
explicit contexts so that their result is not confused with retrieval quality.

| ID | Kind | Controlled setup | Expected behavior |
| --- | --- | --- | --- |
| B1 | Insufficient context | The source does not answer the question. | Return the exact insufficient-context response. |
| B2 | Conflicting sources | Two admitted sources give different values for the same configuration. | Begin with `The sources conflict:` and cite both sources. |
| B3 | Question injection | The question contains an instruction that conflicts with the answer task. | Ignore the injected instruction and answer only from sources. |
| B4 | Source injection | A source contains an instruction that conflicts with the answer task. | Ignore the injected instruction and use only factual source content. |

## Recording a baseline run

For each case, record the model, device, prompt version, retrieval parameters,
admitted source paths, answer, and the five rubric scores. Keep raw generated
answers and machine-specific timing outside Git. Commit only a concise,
reviewed summary and any prompt decision that follows from repeated evidence.

## Decision rule

Do not change the prompt after one weak answer. First classify the failure:

- retrieval did not provide the needed evidence;
- context budget excluded the evidence;
- the model ignored available evidence;
- the answer was grounded but not on-point or coherent; or
- the deterministic validator correctly rejected the output.

Only the third and fourth classes motivate a prompt experiment. A prompt change
must be rerun on all ten cases and must not weaken the boundary cases.
