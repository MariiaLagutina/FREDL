"""Recognize structural blocks in Markdown and plain text sources."""

from dataclasses import dataclass
import re


_ATX_HEADING = re.compile(
    r"^(?P<indent> {0,3})(?P<marks>#{1,6})(?:[ \t]+|$)(?P<title>.*)$"
)


@dataclass(frozen=True, slots=True)
class _MarkdownHeading:
    """Store one exact Markdown heading and its normalized title."""

    level: int
    title: str
    start: int
    end: int

    def __post_init__(self) -> None:
        """Reject invalid heading levels and source ranges."""
        if not 1 <= self.level <= 6:
            message = "Markdown heading level must be between 1 and 6"
            raise ValueError(message)
        if not self.title:
            message = "Markdown heading title must not be empty"
            raise ValueError(message)
        if self.start < 0 or self.end <= self.start:
            message = "Markdown heading range must be positive"
            raise ValueError(message)


def _heading_from_line(
    line: str,
    start: int,
) -> _MarkdownHeading | None:
    """Parse one ATX heading line without changing its source range."""
    line_content = line.removesuffix("\r\n")
    line_content = line_content.removesuffix("\r").removesuffix("\n")
    match = _ATX_HEADING.match(line_content)
    if match is None:
        return None

    raw_title = match.group("title")
    if re.fullmatch(r"#+[ \t]*", raw_title):
        title = ""
    else:
        title = re.sub(r"[ \t]+#+[ \t]*$", "", raw_title)
    title = title.strip()
    if not title:
        return None

    return _MarkdownHeading(
        level=len(match.group("marks")),
        title=title,
        start=start,
        end=start + len(line),
    )


def _update_heading_stack(
    stack: tuple[_MarkdownHeading, ...],
    heading: _MarkdownHeading,
) -> tuple[_MarkdownHeading, ...]:
    """Replace the current heading and all of its descendants."""
    ancestors = tuple(
        current for current in stack
        if current.level < heading.level
    )
    return (*ancestors, heading)


def _section_path(
    stack: tuple[_MarkdownHeading, ...],
) -> tuple[str, ...]:
    """Return retrieval-facing titles from the active heading stack."""
    return tuple(heading.title for heading in stack)
