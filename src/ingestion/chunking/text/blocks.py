"""Recognize structural blocks in Markdown and plain text sources."""

from dataclasses import dataclass
import re


_ATX_HEADING = re.compile(
    r"^(?P<indent> {0,3})(?P<marks>#{1,6})(?:[ \t]+|$)(?P<title>.*)$"
)
_FENCE_OPENING = re.compile(
    r"^ {0,3}(?P<marks>`{3,}|~{3,})(?P<info>.*)$"
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


@dataclass(frozen=True, slots=True)
class _MarkdownFence:
    """Store the marker and exact opening range of a fenced code block."""

    marker: str
    length: int
    start: int
    end: int

    def __post_init__(self) -> None:
        """Reject markers and ranges that cannot form an opening fence."""
        if self.marker not in {"`", "~"} or self.length < 3:
            message = "Markdown fence marker must repeat at least three times"
            raise ValueError(message)
        if self.start < 0 or self.end <= self.start:
            message = "Markdown fence range must be positive"
            raise ValueError(message)


def _opening_fence_from_line(
    line: str,
    start: int,
) -> _MarkdownFence | None:
    """Parse one CommonMark-style opening code fence."""
    line_content = _without_line_ending(line)
    match = _FENCE_OPENING.match(line_content)
    if match is None:
        return None

    marks = match.group("marks")
    info = match.group("info")
    if marks.startswith("`") and "`" in info:
        return None

    return _MarkdownFence(
        marker=marks[0],
        length=len(marks),
        start=start,
        end=start + len(line),
    )


def _is_closing_fence(line: str, fence: _MarkdownFence) -> bool:
    """Return whether a line closes the supplied opening fence."""
    line_content = _without_line_ending(line)
    closing = re.compile(
        rf"^ {{0,3}}{re.escape(fence.marker)}"
        rf"{{{fence.length},}}[ \t]*$"
    )
    return closing.match(line_content) is not None


def _headings_outside_fences(text: str) -> list[_MarkdownHeading]:
    """Collect headings while ignoring all fenced code block content."""
    headings: list[_MarkdownHeading] = []
    active_fence: _MarkdownFence | None = None
    cursor = 0

    for line in text.splitlines(keepends=True):
        if active_fence is not None:
            if _is_closing_fence(line, active_fence):
                active_fence = None
        else:
            active_fence = _opening_fence_from_line(line, cursor)
            if active_fence is None:
                heading = _heading_from_line(line, cursor)
                if heading is not None:
                    headings.append(heading)
        cursor += len(line)

    return headings


def _heading_from_line(
    line: str,
    start: int,
) -> _MarkdownHeading | None:
    """Parse one ATX heading line without changing its source range."""
    line_content = _without_line_ending(line)
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


def _without_line_ending(line: str) -> str:
    """Remove one original line ending without changing other whitespace."""
    content = line.removesuffix("\r\n")
    return content.removesuffix("\r").removesuffix("\n")
