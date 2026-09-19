"""Prompt loading.

Prompts live as versioned files under `prompts/` rather than as string
literals: stage 3 changes them one at a time and measures each change, which
needs a diffable history and a name to record in eval results.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

#: Prompt used by the chat endpoint. Bump the suffix to introduce a new
#: version and record the name alongside eval results.
SYSTEM_PROMPT = "system_v2"


@cache
def load_prompt(name: str = SYSTEM_PROMPT) -> str:
    """Read a prompt by name, caching it for the life of the process."""
    path = PROMPTS_DIR / f"{name}.md"
    if not path.is_file():
        available = sorted(p.stem for p in PROMPTS_DIR.glob("*.md"))
        raise FileNotFoundError(f"prompt {name!r} not found; available: {available}")
    return path.read_text(encoding="utf-8").strip()
