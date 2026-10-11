# Product Decision Log

This log records product decisions that change what FREDL presents to the
reader or treats as learning content. Each entry preserves the reason for the
decision, the evidence available at the time, and the conditions that may
justify revisiting it.

Implementation history belongs in Git. This document records product meaning.

## PD-001: Represent a passage through its verified section and exact PDF page

**Date:** 12 October 2026  
**Status:** accepted

### Context

A PDF may expose several page identities. Printed numbering can begin after
the cover and front matter, while a PDF viewer addresses physical pages from
the beginning of the file. An embedded outline may also include navigation
labels such as `Cover` and `Copyright`. These labels help navigate the file but
do not represent knowledge the reader wants to retrieve.

A chapter and section path is more meaningful in the interface than a page
number alone. FREDL still needs the physical PDF page because every retrieved
passage must lead back to an exact source location.

### Decision

- Present the verified chapter and section path as the primary structural
  context for a retrieved passage.
- Preserve the one-based physical PDF page as the exact navigation and
  provenance coordinate.
- Show a printed book page only when its relationship to the physical PDF page
  has been verified.
- Preserve embedded outline entries for extraction audit, but classify
  `Cover` and `Copyright` as non-content. They do not become section metadata
  and do not enter RAG retrieval content.
- Treat an outline destination as a page hint. Confirm its heading against
  extracted text and layout before assigning the section to a passage.
- Keep unresolved or ambiguous headings explicit instead of inventing a
  section boundary.

### Evidence

The first active technical book contains 195 embedded outline entries.
Layout-aware matching confirmed all 193 content headings. The remaining two
entries, `Cover` and `Copyright`, describe file navigation rather than learning
content.

The experiment also exposed several PDF-specific cases: headings split across
visual lines, omitted whitespace in extracted text, ligature characters,
chapter markers stored separately from titles, and a heading repeated as the
first word of the following paragraph. Text, font size, and fragment position
were all needed to resolve these cases without guessing.

### Consequences

- PDF chunks need a verified section path alongside physical-page provenance.
- Reader links use physical PDF pages even when the interface also displays a
  printed page number.
- Outline labels alone are insufficient evidence for retrieval metadata.
- Front matter may remain available for document audit without polluting the
  reader's knowledge search.

### Revisit when

Revisit this decision when representative books lack usable outlines, use page
labels that cannot be mapped reliably, or contain meaningful learning content
under a label currently classified as non-content.
