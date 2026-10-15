# Product Direction

FREDL is an AI-assisted learning system for technical, scientific, and
educational materials. It is not a general-purpose PDF reader or a
fiction-reading application. Its first intended user is its developer, who is
studying programming and AI engineering. Supporting learning across other
disciplines is a longer-term ambition that requires future validation.

Source-grounded explanations, verifiable book/page provenance, and the ability
to inspect original sources are core product requirements; how the original
page is shown when extracted text falls short is recorded in
[PD-004](product-decisions.md#pd-004-show-the-original-pdf-page-when-extracted-text-cannot-carry-the-evidence).
Personal learning memory and broader multimodal document support remain future
product work; the current PDF utilities do not implement the complete learning
flow.

## The problem FREDL is exploring

People who learn from technical and scientific books often collect more PDFs
than they can read. The collection itself is not the failure. A book may be
useful for one urgent question, one project decision, or one explanation, and
there may be no reason to read it from beginning to end.

The harder problem appears later. The reader remembers an idea but forgets
where it came from, why it mattered, what they tried, and whether the result was
useful. Highlights preserve source text. They rarely preserve the reader's
reason for looking, the project around the question, or the way their
understanding changed.

FREDL explores whether a learning tool can preserve that personal path and
bring it back at a useful moment without creating guilt, streaks, or a rigid
reading plan.

## The learning object

FREDL treats a question as more than a search string. A useful learning record
connects:

- the question in the reader's own words;
- why it arose;
- the book, page, and passage that informed it;
- a note, voice reflection, code fragment, or quick reaction;
- the project or practical situation in which the idea was used;
- a later return and what changed after it.

The product should show the reader's own thoughts more prominently than copied
quotes. Source passages remain available as evidence and context.

## Two retrieval spaces

FREDL needs two related retrieval systems.

The first searches the user's PDFs. It must find relevant passages, preserve
page-level provenance, and allow explanations from several books to be
compared. The user supplies the documents and controls which model, if any,
receives their content.

The second searches the user's learning memory. It returns earlier questions,
reasons, notes, decisions, attempts, and follow-up questions. It should retain
short records of what mattered rather than copying entire books into memory.

## Feedback and evidence

Signals do not have equal meaning. FREDL should prefer them in this order:

1. a written or spoken reflection;
2. an explicit quick action;
3. a return to a suggested book, passage, or question;
4. passive interaction such as scrolling or time on a page.

Passive behavior is weak evidence. Time on a page may mean careful reading, a
break, or distraction. FREDL should keep the observed event separate from any
inference and let the user correct inferred preferences.

Useful quick actions may include:

- return later;
- understood;
- concept unclear;
- interesting;
- find another explanation;
- this source does not suit me.

Repeated negative feedback about a source may reduce how often it is suggested,
while still allowing it to appear when it contains uniquely relevant evidence.

## When a concept is unclear

An unclear concept should open choices rather than a single forced answer. The
reader may ask for another book, a simpler explanation, a concrete example, a
diagram, a step-by-step path, or a later return after an easier prerequisite.

FREDL may learn which explanation style the reader often chooses and make that
option more prominent. Other options must remain easy to reach, especially when
the preferred style did not help. New external sources require the reader's
explicit choice.

## Resurfacing

The first version should let the reader choose when to return. Later versions
may suggest timing from actual return patterns, book type, question type, and
explicit feedback. A universal schedule is unlikely to fit every book or every
kind of learning.

The interface should make resurfacing calm and dismissible. A retriever helper
may offer one earlier thought or question in a side area. Closing it, continuing
to read, asking another question, opening another source, or returning to the
original passage are ambiguous interaction signals, not automatic evidence of
usefulness. Their meaning requires combined evidence and explicit feedback.
FREDL should not ask for a rating after every explanation.

## Current validation boundary

The founder is the first user. The immediate goal is to learn whether the
question-and-context loop helps with real engineering study and project work.
The prototype is evidence only about that use. Broader claims require broader
testing later.

Technical PDF validation now covers four books, with known extraction and layout
limitations. It does not establish cross-book retrieval quality, answerability,
or the usefulness of the full learning loop. New evaluation baselines remain
pending; the implemented utilities and their limits are described in
[Technical Foundation](technical-foundation.md).

A private Task 6 package now contains 22 human-approved candidate questions.
This approves their evaluation intent, not the system's answers. A versioned
multi-book evaluation contract and a four-question human judgment pilot exist,
but the Golden Set is not frozen, and full judgments, library-wide retrieval
runs and new baselines remain incomplete.

The contract records what a question means separately from how well the books
support it. When a question has several materially different meanings, FREDL
should ask which one the reader intends instead of guessing, as recorded in
[PD-005](product-decisions.md#pd-005-ask-for-clarification-instead-of-guessing-the-users-meaning).
This is an evaluation reference; FREDL does not yet ask for clarification.
