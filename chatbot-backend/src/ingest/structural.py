"""Heading-aware chunking.

Stage 3. The naive character window splits mid-sentence, mid-list and
mid-code-block, which cost recall on questions whose answer is a list or a
SQL example. This splitter cuts on markdown headings instead, keeps fenced
code blocks whole, and stamps every chunk with its heading path so a chunk
lifted out of the middle of a document still says what it is about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.ingest.chunker import TextChunk

#: A markdown ATX heading: captures the level and the text.
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
#: Opening or closing fence of a code block.
FENCE_RE = re.compile(r"^\s*(```|~~~)")

#: Sections longer than this are split further, on paragraph boundaries.
MAX_CHUNK_CHARS = 1200
#: A tail shorter than this is merged back rather than left as a fragment.
MIN_CHUNK_CHARS = 120


@dataclass(frozen=True, slots=True)
class Section:
    """One heading and the body beneath it."""

    path: tuple[str, ...]
    body: str

    @property
    def title(self) -> str:
        return self.path[-1] if self.path else ""


def split_sections(text: str) -> list[Section]:
    """Split markdown into sections, tracking the heading path.

    Headings inside fenced code blocks are ignored — a `# comment` in a
    shell example is not a document heading.
    """
    sections: list[Section] = []
    stack: list[str] = []
    buf: list[str] = []
    in_fence = False
    current: tuple[str, ...] = ()

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(path=current, body=body))
        buf.clear()

    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            buf.append(line)
            continue

        heading = None if in_fence else HEADING_RE.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            # Trim the stack to the parent level, then push this heading.
            del stack[level - 1 :]
            stack.append(title)
            current = tuple(stack)
        else:
            buf.append(line)

    flush()
    return sections


def _split_long_body(body: str, limit: int) -> list[str]:
    """Split an over-long section on blank lines, never inside a code fence."""
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False

    for para in body.split("\n\n"):
        fences = len(FENCE_RE.findall(para))
        candidate = "\n\n".join([*current, para]).strip()

        if current and not in_fence and len(candidate) > limit:
            blocks.append("\n\n".join(current).strip())
            current = [para]
        else:
            current.append(para)

        # An odd number of fences in this paragraph flips fence state, so a
        # code block split across paragraphs is never broken up.
        if fences % 2 == 1:
            in_fence = not in_fence

    if current:
        blocks.append("\n\n".join(current).strip())

    # Fold a runt tail back into its predecessor. Pop first: evaluating
    # blocks[-2] and blocks.pop() in one statement shifts the index.
    if len(blocks) > 1 and len(blocks[-1]) < MIN_CHUNK_CHARS:
        tail = blocks.pop()
        blocks[-1] = f"{blocks[-1]}\n\n{tail}"
    return [b for b in blocks if b]


def chunk_by_structure(
    text: str, *, max_chars: int = MAX_CHUNK_CHARS
) -> list[tuple[TextChunk, str]]:
    """Split `text` into chunks, each paired with its heading path.

    Returns (chunk, section_title) pairs. The heading path is prepended to
    the chunk content as well, so the embedding carries that context and a
    retrieved excerpt is self-describing.
    """
    chunks: list[tuple[TextChunk, str]] = []
    index = 0

    for section in split_sections(text):
        heading = " > ".join(section.path)
        for body in _split_long_body(section.body, max_chars):
            content = f"{heading}\n\n{body}" if heading else body
            chunks.append((TextChunk(index=index, content=content), heading))
            index += 1

    return chunks
