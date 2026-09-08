"""Generate one deterministic answer with a loaded chat model."""

from dataclasses import dataclass
from typing import Any, Sequence

from src.generation.backend.config import GenerationConfig
from src.generation.backend.runtime import LoadedGenerationBackend


SUPPORTED_CHAT_ROLES = frozenset({"system", "user", "assistant"})


class GenerationError(RuntimeError):
    """Report a generation result that cannot be used as an answer."""


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """Represent one validated message passed to the model chat template."""

    role: str
    content: str

    def __post_init__(self) -> None:
        """Reject roles and content unsupported by the local chat workflow."""
        if self.role not in SUPPORTED_CHAT_ROLES:
            raise ValueError(f"Unsupported chat role: {self.role!r}")
        if not self.content.strip():
            raise ValueError("Chat message content must not be empty")


def generate_answer(
    messages: Sequence[ChatMessage],
    backend: LoadedGenerationBackend,
    config: GenerationConfig,
) -> str:
    """Generate and decode only the answer tokens appended after the prompt."""
    if not messages:
        raise ValueError("At least one chat message is required")

    tokenizer = backend.tokenizer
    model_inputs: Any = tokenizer.apply_chat_template(
        [
            {"role": message.role, "content": message.content}
            for message in messages
        ],
        add_generation_prompt=True,
        enable_thinking=config.enable_thinking,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = model_inputs.to(backend.device)
    output_ids = backend.model.generate(
        **model_inputs,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
    )

    prompt_length = model_inputs["input_ids"].shape[-1]
    answer_ids = output_ids[0][prompt_length:]
    answer = str(
        tokenizer.decode(answer_ids, skip_special_tokens=True)
    ).strip()
    if not answer:
        raise GenerationError("Generation model returned an empty answer")
    return answer
