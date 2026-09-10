"""The project files an agent keeps: the lab notebook and the living plan.

Two markdown files in the project directory, beside ``slab.toml``, hold
what a session decided to keep. ``NOTEBOOK.md`` is append-only: dated
entries for decisions, results with run ids, and diagnosed failures.
``PLAN.md`` is rewritten whole: the goal, the steps with their status,
the open questions. Both outlive any context window, and both are read
by whoever works in the project next, a person, the resident agent, or
an external harness over MCP. That is why they live here and not in one
agent's session: the files are the project's, and every client writes
them the same way.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

NOTEBOOK_FILE = "NOTEBOOK.md"
PLAN_FILE = "PLAN.md"


def notebook_path(project: Path) -> Path:
    return Path(project) / NOTEBOOK_FILE


def plan_path(project: Path) -> Path:
    return Path(project) / PLAN_FILE


def notebook_append(
    project: Path, entry: str, *, heading: str | None = None, author: str | None = None
) -> Path:
    """Append one dated entry to the notebook, creating it on the first write.

    *author* labels the entry when the writer is not the session's owner:
    a delegated specialist, or a client that wants its entries told apart.
    The notebook is a curated record, so only what a writer calls a result
    lands here; harness machinery writes elsewhere.

    Examples:
        >>> import tempfile
        >>> project = Path(tempfile.mkdtemp())
        >>> _ = notebook_append(project, "a = 3.60 Å (run ab12cd)", heading="lattice")
        >>> text = notebook_path(project).read_text()
        >>> text.startswith("# Lab notebook\\n\\n## ") and " — lattice\\n" in text
        True
        >>> _ = notebook_append(project, "converged", author="dft-expert")
        >>> " [dft-expert]\\n" in notebook_path(project).read_text()
        True
    """
    path = notebook_path(project)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    title = f" — {heading}" if heading else ""
    label = f" [{author}]" if author else ""
    block = f"\n## {stamp}{title}{label}\n\n{entry.rstrip()}\n"
    if not path.exists():
        block = "# Lab notebook\n" + block
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(block)
    return path


def notebook_tail(project: Path, max_chars: int = 3_000) -> str:
    """The notebook's last entries, capped for a context window; empty when none."""
    path = notebook_path(project)
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    if len(text) <= max_chars:
        return text
    return f"[... earlier notebook entries omitted ...]\n{text[-max_chars:]}"


def plan_read(project: Path) -> str:
    """The current plan, or empty when none has been written yet."""
    path = plan_path(project)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def plan_write(project: Path, content: str) -> Path:
    """Rewrite the plan whole; the text is normalized to end with one newline.

    Examples:
        >>> import tempfile
        >>> project = Path(tempfile.mkdtemp())
        >>> _ = plan_write(project, "1. relax Cu")
        >>> plan_read(project)
        '1. relax Cu\\n'
    """
    path = plan_path(project)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    return path
