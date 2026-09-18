"""Turning retrieved chunks into the text the model actually sees."""

from __future__ import annotations

from collections.abc import Sequence

from src.retrieval.base import RetrievedChunk

NO_CONTEXT_NOTE = "(no documentation excerpts matched this question)"


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Render chunks as labelled excerpts.

    Each block carries its chunk id so the model can cite it and so a wrong
    answer can be traced back to the text that caused it.
    """
    if not chunks:
        return NO_CONTEXT_NOTE

    blocks = []
    for chunk in chunks:
        heading = f"[chunk {chunk.chunk_id}]"
        if chunk.section_title:
            heading = f"{heading} {chunk.section_title}"
        blocks.append(f"{heading}\n{chunk.content}")
    return "\n\n---\n\n".join(blocks)


def build_user_message(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    """Compose the user turn: excerpts first, then the question.

    The question goes last so it stays adjacent to the model's own output —
    long context in front of it is easier for the model to reference than
    context trailing behind the instruction.
    """
    return f"Documentation excerpts:\n\n{format_context(chunks)}\n\n---\n\nQuestion: {question}"
