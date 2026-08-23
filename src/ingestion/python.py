"""Map Python AST positions to exact source character offsets."""

from dataclasses import dataclass, field
import re


@dataclass(frozen=True, slots=True)
class _PythonSourceMap:
    """Translate one-based AST lines and UTF-8 columns into string indexes."""

    text: str
    _line_starts: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Precompute the absolute character index of every source line."""
        line_starts = [0]
        for match in re.finditer(r"\r\n|\r|\n", self.text):
            if match.end() < len(self.text):
                line_starts.append(match.end())
        object.__setattr__(self, "_line_starts", tuple(line_starts))

    def character_offset(self, line: int, utf8_column: int) -> int:
        """Return an absolute character index for one AST source position."""
        if line < 1 or line > len(self._line_starts):
            message = "AST line is outside the source text"
            raise ValueError(message)
        if utf8_column < 0:
            message = "AST UTF-8 column must not be negative"
            raise ValueError(message)

        line_start = self._line_starts[line - 1]
        if line < len(self._line_starts):
            line_end = self._line_starts[line]
        else:
            line_end = len(self.text)
        line_text = self.text[line_start:line_end]
        if line_text.endswith("\r\n"):
            line_text = line_text[:-2]
        elif line_text.endswith(("\r", "\n")):
            line_text = line_text[:-1]
        encoded_line = line_text.encode("utf-8")

        if utf8_column > len(encoded_line):
            message = "AST UTF-8 column is outside its source line"
            raise ValueError(message)

        encoded_prefix = encoded_line[:utf8_column]
        try:
            character_prefix = encoded_prefix.decode("utf-8")
        except UnicodeDecodeError as error:
            message = "AST UTF-8 column splits a source character"
            raise ValueError(message) from error

        return line_start + len(character_prefix)
