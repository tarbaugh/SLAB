"""Session state: where Mason works, what it remembers, what it may do.

A session binds a project directory to the agent configuration and owns the
three durable files that carry a research project across context windows and
across weeks (the file-first memory pattern — notes outlive transcripts):

* ``NOTEBOOK.md`` — the append-only lab notebook. Mason records decisions,
  results (with run ids), and failures here; compaction summaries land here
  too, so nothing important lives only in a context window.
* ``PLAN.md`` — the living plan, rewritten as understanding changes.
* ``.slab/mason/sessions/*.jsonl`` — append-only transcripts (one typed
  event per line: messages, tool results, compactions, token counts).
  Resuming replays the newest transcript's messages.

Both markdown files sit in the project directory on purpose: they are
scientific provenance, meant to be read by humans and committed to version
control, not hidden in an agent-private store.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from slab.config import AgentConfig, HpcConfig, SlabConfig, load_config
from slab.errors import SlabError

Approver = Callable[[str, str], bool]
"""``(tool_name, preview) -> allow?`` — the permission gate for mutating tools."""

# Shell control operators disqualify a command from allowlist auto-approval.
_SHELL_CONTROL = re.compile(r"[;&|`<>\n]|\$\(")


class SessionError(SlabError):
    """A session could not be created or resumed."""


def _approve_nothing(tool: str, preview: str) -> bool:
    """The default gate for non-interactive use: mutating tools refuse."""
    return False


class MasonSession:
    """One agent session in one project directory.

    Args:
        cwd: The project directory Mason works in (files, notebook, plan).
        workspace_root: The SLAB workspace for runs (default: resolved the
            usual way — flag > ``$SLAB_WORKSPACE`` > config > ``./.slab``).
        config: The full slab config; loaded from *cwd* when omitted.
        approver: Callback deciding mutating tool calls when approval mode
            is ``"ask"``; the default refuses (safe for non-interactive
            runs — pass an interactive prompt or use approval ``"auto"``).
        auto_approve: True overrides the config's approval mode to allow
            every tool call this session (the ``--auto`` flag).
    """

    def __init__(
        self,
        cwd: str | os.PathLike[str] | None = None,
        *,
        workspace_root: str | os.PathLike[str] | None = None,
        config: SlabConfig | None = None,
        approver: Approver | None = None,
        auto_approve: bool = False,
    ) -> None:
        self.cwd = Path(cwd if cwd is not None else Path.cwd()).resolve()
        loaded = config if config is not None else load_config(self.cwd)
        self.agent: AgentConfig = loaded.agent
        self.hpc: HpcConfig = loaded.hpc
        from slab._ops import resolve_root

        self.workspace_root = (
            Path(workspace_root) if workspace_root is not None else resolve_root(None)
        )
        self.approver: Approver = approver if approver is not None else _approve_nothing
        self.auto_approve = auto_approve
        # Unset compute_profile derives from the machine: a config that declares
        # SLURM partitions is a cluster, anything else is treated as a laptop —
        # the conservative guess, since over-sizing a calculation wastes hours
        # while under-sizing it wastes minutes.
        self.compute_profile = self.agent.compute_profile or (
            "cluster" if self.hpc.partitions else "laptop"
        )
        self.notebook_path = self.cwd / "NOTEBOOK.md"
        self.plan_path = self.cwd / "PLAN.md"
        self.read_files: set[Path] = set()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.sessions_dir = self.workspace_root / "mason" / "sessions"
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        self.transcript_path = self.sessions_dir / f"{stamp}-{os.getpid()}.jsonl"

    # -- permission gate ------------------------------------------------------

    def allows(self, tool_name: str, preview: str, *, requires_approval: bool) -> bool:
        """Whether this tool call may run under the session's approval policy."""
        if not requires_approval or self.auto_approve or self.agent.approval == "auto":
            return True
        return self.approver(tool_name, preview)

    def shell_allowlisted(self, command: str) -> bool:
        """True when the command matches an allowlist prefix at a word boundary.

        Two guards keep a prefix from approving more than it names: the match
        must end at a word boundary (``ls`` approves ``ls -la``, never
        ``lsblk``), and a command containing shell control operators
        (``;``, ``&``, ``|``, backticks, ``$(``, redirection, newlines) never
        auto-approves — ``ls; rm -rf ~`` is not an ``ls``.

        Examples:
            >>> from slab.config import SlabConfig
            >>> config = SlabConfig.model_validate(
            ...     {"agent": {"shell_allowlist": ["git status", "ls"]}})
            >>> session = MasonSession("/tmp", config=config)
            >>> session.shell_allowlisted("git status --short")
            True
            >>> session.shell_allowlisted("git push")
            False
            >>> session.shell_allowlisted("ls; rm -rf ~")
            False
            >>> session.shell_allowlisted("lsblk")
            False
        """
        stripped = command.strip()
        if _SHELL_CONTROL.search(stripped):
            return False
        for prefix in self.agent.shell_allowlist:
            prefix = prefix.strip()
            if prefix and (stripped == prefix or stripped.startswith(prefix + " ")):
                return True
        return False

    # -- durable files --------------------------------------------------------

    def notebook_append(self, entry: str, *, heading: str | None = None) -> None:
        """Append one entry to the lab notebook (created on first write)."""
        stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        title = f" — {heading}" if heading else ""
        block = f"\n## {stamp}{title}\n\n{entry.rstrip()}\n"
        if not self.notebook_path.exists():
            block = "# Lab notebook\n" + block
        with open(self.notebook_path, "a", encoding="utf-8") as handle:
            handle.write(block)

    def notebook_tail(self, max_chars: int = 3_000) -> str:
        """The notebook's last entries, budget-capped for the context."""
        if not self.notebook_path.exists():
            return ""
        text = self.notebook_path.read_text(encoding="utf-8")
        if len(text) <= max_chars:
            return text
        return f"[... earlier notebook entries omitted ...]\n{text[-max_chars:]}"

    def plan_text(self) -> str:
        """The current plan, or empty when none has been written yet."""
        if not self.plan_path.exists():
            return ""
        return self.plan_path.read_text(encoding="utf-8")

    # -- transcript -----------------------------------------------------------

    def record(self, event: dict[str, Any]) -> None:
        """Append one event to the session transcript (JSONL, append-only)."""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        stamped = {"at": datetime.now(UTC).isoformat(), **event}
        with open(self.transcript_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(stamped, ensure_ascii=False) + "\n")

    def count_usage(self, prompt_tokens: int | None, completion_tokens: int | None) -> None:
        if prompt_tokens:
            self.prompt_tokens += prompt_tokens
        if completion_tokens:
            self.completion_tokens += completion_tokens

    def latest_transcript(self) -> Path | None:
        """The newest transcript in this workspace, or None."""
        if not self.sessions_dir.is_dir():
            return None
        candidates = sorted(p for p in self.sessions_dir.glob("*.jsonl") if p.is_file())
        return candidates[-1] if candidates else None

    def load_messages(self, transcript: Path) -> list[dict[str, Any]]:
        """Replay a transcript's message events (for ``--resume``).

        Only ``message`` events matter for the model; tool results are
        stored inside them. A malformed line is an error, not a skip — a
        corrupt transcript must surface, not silently resume half a
        conversation.
        """
        messages: list[dict[str, Any]] = []
        number = 0
        try:
            with open(transcript, encoding="utf-8") as handle:
                for number, line in enumerate(handle, start=1):  # noqa: B007 - named for the error path
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if event.get("type") == "message":
                        messages.append(event["message"])
        except (OSError, json.JSONDecodeError, KeyError) as e:
            raise SessionError(f"cannot resume from {transcript} (line {number}): {e}") from e
        return messages
