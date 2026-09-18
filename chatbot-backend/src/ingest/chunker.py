"""Text splitting.

Stage 2 is deliberately naive: a fixed character window with overlap, blind
to document structure. It is the baseline that stage 3's heading-aware
chunking will be measured against.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A piece of text before it is embedded."""

    index: int
    content: str


def chunk_text(text: str, *, size: int = 500, overlap: int = 50) -> list[TextChunk]:
    """Split `text` into `size`-character pieces overlapping by `overlap`.

    The overlap keeps a phrase that straddles a boundary intact in at least
    one chunk: consecutive pieces repeat each other's edges.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        # step would be <= 0 and the loop below would never terminate
        raise ValueError("overlap must be >= 0 and < size")

    text = text.strip()
    if not text:
        return []

    step = size - overlap
    chunks: list[TextChunk] = []
    start = 0
    index = 0

    while start < len(text):
        piece = text[start : start + size].strip()
        if piece:
            chunks.append(TextChunk(index=index, content=piece))
            index += 1
        start += step

    return chunks
