"""Reading source documents off disk."""

from __future__ import annotations

from pathlib import Path

from src.ingest.base import SourceDocument

DEFAULT_VERSION = "1"


def load_markdown_file(
    path: Path,
    *,
    source_uri: str | None = None,
    version: str = DEFAULT_VERSION,
) -> SourceDocument:
    """Read one markdown file into a SourceDocument.

    The title is taken from the first `# ` heading, falling back to the file
    name when the document has none.
    """
    text = path.read_text(encoding="utf-8")
    return SourceDocument(
        title=_extract_title(text) or path.stem,
        content=text,
        # Relative to the project root, not absolute: source_uri is the key
        # re-ingestion matches on, and an absolute path silently stops
        # matching the moment the project is moved or cloned elsewhere.
        source_uri=source_uri or _relative_uri(path),
        version=version,
    )


def _relative_uri(path: Path) -> str:
    """Path relative to the project root, with a stable POSIX separator."""
    root = Path(__file__).resolve().parents[2]
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        # Outside the project tree — fall back to the absolute path.
        return resolved.as_posix()


def load_directory(
    directory: Path,
    *,
    pattern: str = "*.md",
    version: str = DEFAULT_VERSION,
) -> list[SourceDocument]:
    """Load every file matching `pattern`, sorted by name for a stable order."""
    return [load_markdown_file(path, version=version) for path in sorted(directory.glob(pattern))]


def _extract_title(text: str) -> str | None:
    """Return the text of the first level-1 markdown heading, if any."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None
