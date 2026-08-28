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
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mason.config import AgentConfig, load_config
from mason.errors import MasonError
from slab.config import HpcConfig
from slab.config import load_config as load_slab_config

Approver = Callable[[str, str], bool]
"""``(tool_name, preview) -> allow?`` — the permission gate for mutating tools."""

Observer = Callable[[str, str, str], None]
"""``(kind, attribution, text)`` — live step output for interactive display.

``kind`` is ``"reasoning"`` (the model's thinking, when the server's
reasoning parser separates it) or ``"text"`` (assistant prose emitted
alongside tool calls, which otherwise never reaches the terminal).
``attribution`` is the delegated agent's marker, empty for the session
owner. Display only: the transcript records reasoning regardless.
"""

# Shell control operators disqualify a command from allowlist auto-approval.
_SHELL_CONTROL = re.compile(r"[;&|`<>\n]|\$\(")

# What a conversation transcript is named; delegation transcripts append
# the agent name and an ordinal and are never resumed as conversations.
_CONVERSATION_TRANSCRIPT = re.compile(r"^\d{8}-\d{6}-\d+\.jsonl$")


class SessionError(MasonError):
    """A session could not be created or resumed."""


def _approve_nothing(tool: str, preview: str) -> bool:
    """The default gate for non-interactive use: mutating tools refuse."""
    return False


def transcript_groups(
    workspace_root: str | os.PathLike[str],
) -> list[tuple[Path, list[Path]]]:
    """Conversation transcripts with their delegation siblings, oldest first.

    A conversation transcript is ``<stamp>-<pid>.jsonl``; the delegation
    transcripts its turns produced share its stem with an agent name and
    an ordinal appended. Grouping them keeps a sweep from deleting a
    conversation while stranding its specialists' archives, or the
    reverse. This is the one layout fact ``slab-stack purge`` needs, so
    it lives here with the layout's owner.
    """
    sessions = Path(workspace_root) / "mason" / "sessions"
    if not sessions.is_dir():
        return []
    conversations = sorted(
        p
        for p in sessions.glob("*.jsonl")
        if p.is_file() and _CONVERSATION_TRANSCRIPT.match(p.name)
    )
    return [
        (
            conversation,
            sorted(
                p
                for p in sessions.glob(f"{conversation.stem}-*.jsonl")
                if p.is_file()
            ),
        )
        for conversation in conversations
    ]


class MasonSession:
    """One agent session in one project directory.

    Args:
        cwd: The project directory Mason works in (files, notebook, plan).
        workspace_root: The SLAB workspace for runs (default: resolved the
            usual way — flag > ``$SLAB_WORKSPACE`` > config > ``./.slab``).
        agent: The ``[agent]`` table (:mod:`mason.config`); read from *cwd*
            when omitted.
        hpc: The ``[hpc]`` table (:mod:`slab.config`); read from *cwd* when
            omitted. Each table comes from the package that owns it, so
            pinning one does not require inventing the other.
        approver: Callback deciding mutating tool calls when approval mode
            is ``"ask"``; the default refuses (safe for non-interactive
            runs — pass an interactive prompt or use approval ``"auto"``).
        auto_approve: True overrides the config's approval mode to allow
            every tool call this session (the ``--auto`` flag).
        observer: Callback receiving live step output (reasoning, interim
            assistant text) for display; ``None`` (the default) shows
            nothing. Delegated children inherit it.
    """

    def __init__(
        self,
        cwd: str | os.PathLike[str] | None = None,
        *,
        workspace_root: str | os.PathLike[str] | None = None,
        agent: AgentConfig | None = None,
        hpc: HpcConfig | None = None,
        approver: Approver | None = None,
        auto_approve: bool = False,
        observer: Observer | None = None,
    ) -> None:
        self.cwd = Path(cwd if cwd is not None else Path.cwd()).resolve()
        # Each table comes from the package that owns it. Passing one in skips
        # only that package's file read, so a caller can pin the agent without
        # inventing an [hpc] section it does not care about.
        self.agent: AgentConfig = agent if agent is not None else load_config(self.cwd).agent
        # The configuration as loaded, before flags, roster tables, or
        # endpoint discovery mutate self.agent. Delegated agents derive
        # their effective config from this, so one agent's table never
        # leaks into another's.
        self.base_agent: AgentConfig = self.agent
        self.hpc: HpcConfig = hpc if hpc is not None else load_slab_config(self.cwd).hpc
        from foundation._ops import resolve_root

        self.workspace_root = (
            Path(workspace_root) if workspace_root is not None else resolve_root(None)
        )
        self.approver: Approver = approver if approver is not None else _approve_nothing
        self.auto_approve = auto_approve
        self.observer: Observer | None = observer
        # Which agent card this session runs as; the loop sets it from the
        # spec it resolves. Delegated child sessions carry the specialist's
        # name for attribution in approvals and notebook entries.
        self.agent_name = "pi"
        # CLI flag overrides, kept so they can be re-asserted over
        # [agent.roster.<name>] tables: a flag outranks config.
        self.flag_updates: dict[str, object] = {}
        self._parent: MasonSession | None = None
        self._children_spawned = 0
        # Unset compute_profile derives from the machine: a config that declares
        # SLURM partitions is a cluster, anything else is treated as a laptop —
        # the conservative guess, since over-sizing a calculation wastes hours
        # while under-sizing it wastes minutes.
        self.compute_profile = self.agent.compute_profile or (
            "cluster" if self.hpc.partitions else "laptop"
        )
        self.endpoint = ""
        self.endpoint_origin = ""
        self.resolve_endpoint()
        self.notebook_path = self.cwd / "NOTEBOOK.md"
        self.plan_path = self.cwd / "PLAN.md"
        self.read_files: set[Path] = set()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.sessions_dir = self.workspace_root / "mason" / "sessions"
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        self.transcript_path = self.sessions_dir / f"{stamp}-{os.getpid()}.jsonl"
        self._lock_handle: Any | None = None

    # -- one running session per workspace ------------------------------------

    def acquire_session_lock(self) -> None:
        """Refuse to run alongside another mason in the same workspace.

        Two concurrent sessions interleave ``NOTEBOOK.md`` entries and race
        each other's view of the plan, so the loop takes an advisory lock
        before its first turn. Contention is refused loudly, naming the
        holder. Delegated children never call this — they run inside the
        parent's lock. A filesystem that cannot lock (some parallel
        filesystems) degrades to a warning: an undetected concurrent session
        beats a workspace nobody can use. ``[agent] session_lock = false``
        turns the lock off.
        """
        if not self.agent.session_lock or self._lock_handle is not None:
            return
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX platform
            return
        lock_path = self.sessions_dir.parent / "session.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "a+", encoding="utf-8")  # noqa: SIM115 - held for the process's life
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            holder = handle.read().strip() or "holder unknown"
            handle.close()
            raise MasonError(
                f"another mason session is already working in this workspace "
                f"({holder}). Finish or stop it first, or set [agent] "
                f"session_lock = false to run concurrent sessions anyway."
            ) from None
        except OSError as e:
            handle.close()
            warnings.warn(
                f"the workspace filesystem cannot hold the session lock ({e}); "
                f"concurrent mason sessions here will not be detected",
                stacklevel=2,
            )
            return
        handle.seek(0)
        handle.truncate()
        handle.write(
            f"pid {os.getpid()}, cwd {self.cwd}, "
            f"started {datetime.now(UTC).isoformat(timespec='seconds')}\n"
        )
        handle.flush()
        self._lock_handle = handle

    def release_session_lock(self) -> None:
        """Let the workspace go (normally implicit in process exit).

        The lock rides the open file handle, so a finished process releases
        it without help. This exists for the caller that ends one session
        and starts another in the same process — a resume, a test.
        """
        if self._lock_handle is not None:
            self._lock_handle.close()
            self._lock_handle = None

    # -- where the model lives ------------------------------------------------

    def resolve_endpoint(self, override: str | None = None) -> None:
        """Settle which endpoint this session talks to, and remember why.

        A ``--endpoint`` flag outranks everything; otherwise the answer comes
        from :func:`mason.serve.discover_endpoint` — config, then the
        record a running server job wrote, then the provider's default. Call
        again after changing ``provider`` or ``model``: which server is the
        right one depends on both.
        """
        from mason.serve import discover_endpoint

        if override:
            self.agent = self.agent.model_copy(update={"endpoint": override})
            self.endpoint, self.endpoint_origin = override, "--endpoint"
            return
        endpoint, origin = discover_endpoint(self.agent, self.workspace_root)
        if endpoint != self.agent.resolved_endpoint:
            self.agent = self.agent.model_copy(update={"endpoint": endpoint})
        self.endpoint, self.endpoint_origin = endpoint, origin

    # -- delegation -----------------------------------------------------------

    def spawn(self, agent_name: str, agent: AgentConfig) -> MasonSession:
        """A delegated child session: shared gate and memory, its own transcript.

        The child shares the project directory, the workspace, the ``[hpc]``
        view, the approver, the auto-approve policy, and the observer — one
        permission regime and one terminal per session, whoever asks. Its
        token usage chains upward so the parent's totals stay whole-session
        truths. Fresh per child: the
        read-files staleness guard (a specialist must read a file before
        editing it even when the parent read it), and the transcript, named
        after the parent's with the agent and an ordinal so ``--resume``
        can tell conversations from delegations apart.
        """
        child = MasonSession(
            self.cwd,
            workspace_root=self.workspace_root,
            agent=agent,
            hpc=self.hpc,
            approver=self.approver,
            auto_approve=self.auto_approve,
            observer=self.observer,
        )
        child.agent_name = agent_name
        child._parent = self
        # A flag outranks config for everyone: the child's loop re-asserts
        # these over its own [agent.roster] table exactly as the parent did.
        child.flag_updates = dict(self.flag_updates)
        self._children_spawned += 1
        child.transcript_path = self.transcript_path.with_name(
            f"{self.transcript_path.stem}-{agent_name}-{self._children_spawned}.jsonl"
        )
        return child

    @property
    def session_id(self) -> str:
        """The chat's id, stamped on every run this session launches.

        The value is the root transcript's stem, so one chat has one id and a
        delegated specialist's runs join the chat that asked for them. Foundation
        stores it on each run, which is what makes ``foundation promote
        --session <id>`` able to promote a whole conversation's results.
        """
        if self._parent is not None:
            return self._parent.session_id
        return self.transcript_path.stem

    def attribution(self) -> str:
        """The ``[agent]`` marker for approval previews — children only."""
        return f"[{self.agent_name}] " if self._parent is not None else ""

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
            >>> from mason.config import AgentConfig
            >>> from slab.config import HpcConfig
            >>> agent = AgentConfig.model_validate(
            ...     {"shell_allowlist": ["git status", "ls"]})
            >>> session = MasonSession("/tmp", agent=agent, hpc=HpcConfig())
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
        # The notebook is shared across the whole session (it is the group's
        # blackboard), so a delegated agent's entries carry its name.
        label = f" [{self.agent_name}]" if self._parent is not None else ""
        block = f"\n## {stamp}{title}{label}\n\n{entry.rstrip()}\n"
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
        if self._parent is not None:
            self._parent.count_usage(prompt_tokens, completion_tokens)

    def latest_transcript(self) -> Path | None:
        """The newest conversation transcript in this workspace, or None.

        Delegation transcripts (``<stamp>-<pid>-<agent>-<n>.jsonl``) are
        archives of one errand, not conversations; resuming one would
        replay a specialist's context as if it were the session. Only
        parent-pattern names qualify.
        """
        if not self.sessions_dir.is_dir():
            return None
        candidates = sorted(
            p
            for p in self.sessions_dir.glob("*.jsonl")
            if p.is_file() and _CONVERSATION_TRANSCRIPT.match(p.name)
        )
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
