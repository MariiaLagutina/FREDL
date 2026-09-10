"""Validate structural grounding guarantees in generated answers."""

import re
from collections.abc import Sequence

from src.generation.backend import GenerationError
from src.generation.prompt.template import INSUFFICIENT_CONTEXT_RESPONSE
from src.models import MinimalSource


CONFLICT_PREFIX = "The sources conflict:"
SOURCE_CITATION_PATTERN = re.compile(r"\[Source ([1-9][0-9]*)\]")


class GroundedAnswerValidationError(GenerationError):
    """Report generated text that violates the grounding contract."""


def validate_grounded_answer(
    answer: str,
    sources: Sequence[MinimalSource],
) -> None:
    """Require valid source citations or the exact fallback response."""
    if answer == INSUFFICIENT_CONTEXT_RESPONSE:
        return

    citations = {
        int(match.group(1))
        for match in SOURCE_CITATION_PATTERN.finditer(answer)
    }
    if not citations:
        raise GroundedAnswerValidationError(
            "Generated answer does not cite any retrieved source"
        )

    source_count = len(sources)
    if any(citation > source_count for citation in citations):
        raise GroundedAnswerValidationError(
            "Generated answer cites a source outside the prompt context"
        )

    if answer.startswith(CONFLICT_PREFIX) and len(citations) < 2:
        raise GroundedAnswerValidationError(
            "Generated conflict answer must cite at least two sources"
        )
