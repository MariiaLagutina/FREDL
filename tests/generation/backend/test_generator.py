"""Tests for deterministic answer generation with a loaded backend."""

from dataclasses import dataclass
from typing import Any

import pytest

from src.generation import (
    ChatMessage,
    GenerationConfig,
    GenerationError,
    LoadedGenerationBackend,
    generate_answer,
)


class FakeInputIds:
    """Expose only the prompt shape used by answer slicing."""

    shape = (1, 3)


class FakeBatch(dict[str, Any]):
    """Record device placement for tokenizer output."""

    def __init__(self) -> None:
        super().__init__(input_ids=FakeInputIds(), attention_mask="mask")
        self.device: str | None = None

    def to(self, device: str) -> "FakeBatch":
        """Record and return the same mapping like a BatchEncoding."""
        self.device = device
        return self


class FakeTokenizer:
    """Record chat-template and decoding arguments."""

    def __init__(self, decoded_answer: str = "  grounded answer  ") -> None:
        self.batch = FakeBatch()
        self.template_call: (
            tuple[list[dict[str, str]], dict[str, Any]] | None
        ) = None
        self.decode_call: tuple[Any, bool] | None = None
        self.decoded_answer = decoded_answer

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> FakeBatch:
        """Capture the normalized messages and rendering controls."""
        self.template_call = (messages, kwargs)
        return self.batch

    def decode(self, token_ids: Any, *, skip_special_tokens: bool) -> str:
        """Capture the generated-only token slice."""
        self.decode_call = (token_ids, skip_special_tokens)
        return self.decoded_answer


@dataclass
class FakeModel:
    """Return a prompt followed by two inspectable answer tokens."""

    generation_call: dict[str, Any] | None = None

    def generate(self, **kwargs: Any) -> list[list[int]]:
        """Capture deterministic decoding options."""
        self.generation_call = kwargs
        return [[10, 11, 12, 90, 91]]


def test_generate_answer_uses_qwen_template_and_decodes_new_tokens() -> None:
    """Generation excludes prompt tokens and passes reproducible controls."""
    tokenizer = FakeTokenizer()
    model = FakeModel()
    backend = LoadedGenerationBackend(tokenizer, model, "cuda")
    config = GenerationConfig(max_new_tokens=42)
    messages = [
        ChatMessage("system", "Use only supplied context."),
        ChatMessage("user", "Where is the parser?"),
    ]

    answer = generate_answer(messages, backend, config)

    assert answer == "grounded answer"
    assert tokenizer.template_call == (
        [
            {"role": "system", "content": "Use only supplied context."},
            {"role": "user", "content": "Where is the parser?"},
        ],
        {
            "add_generation_prompt": True,
            "enable_thinking": False,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        },
    )
    assert tokenizer.batch.device == "cuda"
    assert model.generation_call == {
        "input_ids": tokenizer.batch["input_ids"],
        "attention_mask": "mask",
        "max_new_tokens": 42,
        "do_sample": False,
    }
    assert tokenizer.decode_call == ([90, 91], True)


@pytest.mark.parametrize(
    ("role", "content", "message"),
    [
        ("tool", "result", "Unsupported chat role"),
        ("user", "  ", "content must not be empty"),
    ],
)
def test_chat_message_rejects_invalid_values(
    role: str,
    content: str,
    message: str,
) -> None:
    """Messages fail before malformed input reaches the tokenizer."""
    with pytest.raises(ValueError, match=message):
        ChatMessage(role, content)


def test_generate_answer_rejects_empty_conversation() -> None:
    """The backend does not create an answer without an explicit prompt."""
    backend = LoadedGenerationBackend(FakeTokenizer(), FakeModel(), "cpu")

    with pytest.raises(ValueError, match="At least one chat message"):
        generate_answer([], backend, GenerationConfig())


def test_generate_answer_rejects_empty_decoded_output() -> None:
    """Whitespace-only model output cannot masquerade as an answer."""
    backend = LoadedGenerationBackend(
        FakeTokenizer("  \n"), FakeModel(), "cpu"
    )

    with pytest.raises(GenerationError, match="empty answer"):
        generate_answer(
            [ChatMessage("user", "Question")],
            backend,
            GenerationConfig(),
        )
