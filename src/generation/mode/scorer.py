"""Score finite answer-mode labels with a loaded causal language model."""

from collections.abc import Mapping, Sequence
from importlib import import_module
import math
from typing import Any, Protocol

from src.generation.backend import ChatMessage, LoadedGenerationBackend
from src.generation.context.models import ContextBuildResult
from src.generation.mode.models import AnswerMode
from src.generation.mode.prompt import build_answer_mode_messages


class SequenceScoringRuntime(Protocol):
    """Score fixed text continuations after one chat prompt."""

    def score_continuations(
        self,
        messages: Sequence[ChatMessage],
        backend: LoadedGenerationBackend,
        candidates: Sequence[str],
    ) -> Mapping[str, float]:
        """Return a normalized conditional score for each candidate."""


class TransformersSequenceScoringRuntime:
    """Compute mean token log-probability with PyTorch and Transformers."""

    def score_continuations(
        self,
        messages: Sequence[ChatMessage],
        backend: LoadedGenerationBackend,
        candidates: Sequence[str],
    ) -> Mapping[str, float]:
        """Score candidates without unconstrained text generation."""
        torch: Any = import_module("torch")
        tokenizer = backend.tokenizer
        prompt_inputs: Any = tokenizer.apply_chat_template(
            [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_ids = prompt_inputs["input_ids"].to(backend.device)
        if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
            raise ValueError("Mode scoring prompt must contain one sequence")
        prompt_length = int(prompt_ids.shape[1])
        if prompt_length == 0:
            raise ValueError("Mode scoring prompt encoded to no tokens")

        scores: dict[str, float] = {}
        for candidate in candidates:
            candidate_ids = tokenizer.encode(
                candidate,
                add_special_tokens=False,
            )
            if not candidate_ids:
                raise ValueError("Answer mode encoded to no tokens")
            candidate_tensor = torch.tensor(
                [candidate_ids],
                dtype=prompt_ids.dtype,
                device=backend.device,
            )
            input_ids = torch.cat((prompt_ids, candidate_tensor), dim=1)
            attention_mask = torch.ones_like(input_ids)
            with torch.no_grad():
                outputs = backend.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            logits = outputs.logits[
                0,
                prompt_length - 1:prompt_length + len(candidate_ids) - 1,
                :,
            ]
            log_probabilities = torch.log_softmax(logits, dim=-1)
            target_ids = torch.tensor(
                candidate_ids,
                dtype=torch.long,
                device=backend.device,
            ).unsqueeze(1)
            token_scores = torch.gather(
                log_probabilities,
                dim=1,
                index=target_ids,
            )
            scores[candidate] = float(token_scores.mean().item())
        return scores


class HuggingFaceAnswerModeScorer:
    """Adapt a loaded backend to the model-independent mode scorer."""

    def __init__(
        self,
        backend: LoadedGenerationBackend,
        runtime: SequenceScoringRuntime | None = None,
    ) -> None:
        self._backend = backend
        self._runtime = runtime or TransformersSequenceScoringRuntime()

    def score_modes(
        self,
        question: str,
        context: ContextBuildResult,
        candidates: Sequence[AnswerMode],
    ) -> Mapping[AnswerMode, float]:
        """Score every requested mode while preserving finite evidence."""
        candidate_tuple = tuple(candidates)
        messages = build_answer_mode_messages(
            question,
            context,
            candidate_tuple,
        )
        candidate_texts = tuple(mode.value for mode in candidate_tuple)
        raw_scores = self._runtime.score_continuations(
            messages,
            self._backend,
            candidate_texts,
        )
        if set(raw_scores) != set(candidate_texts):
            raise ValueError(
                "Mode scoring runtime must score every candidate once"
            )
        if any(not math.isfinite(score) for score in raw_scores.values()):
            raise ValueError(
                "Mode scoring runtime returned a non-finite score"
            )
        return {
            mode: float(raw_scores[mode.value])
            for mode in candidate_tuple
        }
