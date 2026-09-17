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

import hashlib
import json
import os
import re
import threading
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import EllipsisType
from typing import Any

from foundation import project as project_files
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
_DELEGATION_TRANSCRIPT = re.compile(r"^(\d{8}-\d{6}-\d+)-.+-\d+\.jsonl$")


class SessionError(MasonError):
    """A session could not be created or resumed."""


def _approve_nothing(tool: str, preview: str) -> bool:
    """The default gate for non-interactive use: mutating tools refuse."""
    return False


def transcript_groups(
    workspace_root: str | os.PathLike[str],
    *,
    include_orphans: bool = False,
) -> list[tuple[Path, list[Path]]]:
    """Conversation transcripts with their delegation siblings, oldest first.

    A conversation transcript is ``<stamp>-<pid>.jsonl``; the delegation
    transcripts its turns produced share its stem with an agent name and
    an ordinal appended. Grouping them keeps a sweep from deleting a
    conversation while stranding its specialists' archives, or the
    reverse. A delegation transcript whose conversation is gone is an
    orphan. With *include_orphans* each orphan is a group of its own, so
    a sweep still reaches it; the readers leave them out, because an
    orphan is not a conversation to resume or report. This is the one
    layout fact ``slab purge`` needs, so it lives here with the layout's
    owner.
    """
    sessions = Path(workspace_root) / "mason" / "sessions"
    if not sessions.is_dir():
        return []
    conversations = sorted(
        p
        for p in sessions.glob("*.jsonl")
        if p.is_file() and _CONVERSATION_TRANSCRIPT.match(p.name)
    )
    stems = {conversation.stem for conversation in conversations}
    groups = [
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
    if not include_orphans:
        return groups
    orphans = sorted(
        p
        for p in sessions.glob("*.jsonl")
        if p.is_file()
        and (match := _DELEGATION_TRANSCRIPT.match(p.name)) is not None
        and match.group(1) not in stems
    )
    groups.extend((orphan, []) for orphan in orphans)
    return sorted(groups)


def unrecognised_session_files(workspace_root: str | os.PathLike[str]) -> list[Path]:
    """The files under ``mason/sessions`` and ``mason/reviews`` no transcript group claims.

    A file that is neither a conversation transcript, a delegation
    transcript, nor a sidecar of one (``<stem>.compactions.md``, or a
    review record of one) is listed here and never deleted silently:
    ``slab purge`` prints these and removes them only with
    ``--all-sessions``.

    Examples:
        >>> import tempfile
        >>> root = Path(tempfile.mkdtemp())
        >>> sessions = root / "mason" / "sessions"
        >>> sessions.mkdir(parents=True)
        >>> _ = (sessions / "20260901-120000-1.jsonl").write_text("{}\\n")
        >>> _ = (sessions / "20260901-120000-1.compactions.md").write_text("#\\n")
        >>> _ = (sessions / "notes.txt").write_text("stray\\n")
        >>> _ = (sessions / "20260801-100000-9.compactions.md").write_text("#\\n")
        >>> [p.name for p in unrecognised_session_files(root)]
        ['20260801-100000-9.compactions.md', 'notes.txt']
    """
    mason = Path(workspace_root) / "mason"
    claimed: set[Path] = set()
    for conversation, siblings in transcript_groups(workspace_root, include_orphans=True):
        claimed.update((conversation, *siblings))
        claimed.update(session_sidecars(workspace_root, conversation))
    found: list[Path] = []
    for directory in (mason / "sessions", mason / "reviews"):
        if directory.is_dir():
            found.extend(p for p in directory.iterdir() if p.is_file() and p not in claimed)
    return sorted(found)


def stale_locks(workspace_root: str | os.PathLike[str]) -> list[Path]:
    """The session lock files under ``mason/locks`` that no process holds.

    A lock rides an open file handle, so a lock that can be taken without
    blocking has no holder: its session ended, or its process died. The
    probe takes and releases each lock in turn and never touches a held
    one. On a filesystem that cannot lock, every file is reported held.

    Examples:
        >>> import tempfile
        >>> root = Path(tempfile.mkdtemp())
        >>> locks = root / "mason" / "locks"
        >>> locks.mkdir(parents=True)
        >>> _ = (locks / "abc.lock").write_text("pid 1, cwd /x\\n")
        >>> [p.name for p in stale_locks(root)]
        ['abc.lock']
    """
    locks = Path(workspace_root) / "mason" / "locks"
    if not locks.is_dir():
        return []
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX platform
        return []
    stale: list[Path] = []
    for path in sorted(locks.glob("*.lock")):
        if not path.is_file():
            continue
        try:
            with open(path, "a+", encoding="utf-8") as handle:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            continue
        stale.append(path)
    return stale


def workspace_containing(path: str | os.PathLike[str] | None = None) -> Path | None:
    """The workspace that *path* (default: the current directory) is inside, or None.

    A directory is a workspace when it holds a run store (``runs.db``) or a
    ``mason/sessions`` directory. The walk starts at *path* and climbs to
    the filesystem root, so standing anywhere inside a workspace, its
    sessions directory included, finds it. A project directory is not
    inside its own ``.slab``, so the usual resolution still applies there.

    Examples:
        >>> import tempfile
        >>> root = Path(tempfile.mkdtemp())
        >>> workspace_containing(root) is None
        True
        >>> (root / "mason" / "sessions").mkdir(parents=True)
        >>> workspace_containing(root / "mason" / "sessions") == root.resolve()
        True
    """
    start = Path(path if path is not None else Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "runs.db").is_file() or (candidate / "mason" / "sessions").is_dir():
            return candidate
    return None


def session_sidecars(workspace_root: str | os.PathLike[str], conversation: Path) -> list[Path]:
    """The files a conversation leaves beside its transcripts: compaction
    summaries and review records, for it and for its delegation sessions.

    A session writes ``<stem>.compactions.md`` next to its transcript and
    ``<workspace>/mason/reviews/<stem>-review-<n>.md`` for each review. A
    delegation session's stem is the conversation's stem with an agent
    name and an ordinal appended, so a prefix match on the stem collects
    the whole group. ``slab purge`` deletes these with the transcripts;
    the notebook and the plan live in the project and are never touched.
    """
    root = Path(workspace_root) / "mason"
    stem = conversation.stem
    files = [
        p
        for p in (root / "sessions").glob(f"{stem}*.compactions.md")
        if p.is_file()
    ]
    reviews = root / "reviews"
    if reviews.is_dir():
        files.extend(p for p in reviews.glob(f"{stem}*-review-*.md") if p.is_file())
    return sorted(files)


def session_header(transcript: Path) -> dict[str, Any]:
    """The ``session`` header event a transcript opens with, or ``{}``.

    Older transcripts have none; a damaged first line is skipped rather than
    refused, because the header is a convenience for listings, not a gate.
    Only the first few lines are read: the header is the first event a
    session records.
    """
    try:
        with open(transcript, encoding="utf-8") as handle:
            for _ in range(5):
                line = handle.readline()
                if not line:
                    break
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and event.get("type") == "session":
                    return event
    except OSError:
        pass
    return {}


def transcript_stamp(transcript: Path) -> str:
    """The launch time a transcript's name carries, as ``YYYY-MM-DD HH:MM:SS``.

    Examples:
        >>> transcript_stamp(Path("20260901-234154-2590.jsonl"))
        '2026-09-01 23:41:54'
        >>> transcript_stamp(Path("odd-name.jsonl"))
        'odd-name'
    """
    stem = transcript.stem
    parts = stem.split("-")
    if len(parts) >= 2 and len(parts[0]) == 8 and len(parts[1]) == 6 and parts[0].isdigit():
        d, t = parts[0], parts[1]
        return f"{d[:4]}-{d[4:6]}-{d[6:]} {t[:2]}:{t[2:4]}:{t[4:]}"
    return stem


def transcript_for(workspace_root: str | os.PathLike[str], session: str) -> Path:
    """The conversation transcript for a session id, or a unique prefix of one.

    The session id is the transcript's stem, so this is the lookup behind
    ``slab mason report --session`` and the benchmark scorer. An id no
    transcript matches, or a prefix several match, raises naming them.
    """
    conversations = [conversation for conversation, _ in transcript_groups(workspace_root)]
    exact = [c for c in conversations if c.stem == session]
    if exact:
        return exact[0]
    matches = [c for c in conversations if c.stem.startswith(session)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SessionError(f"no session transcript matches {session!r} under {workspace_root}")
    names = ", ".join(c.stem for c in matches)
    raise SessionError(f"session prefix {session!r} is ambiguous: {names}")


class WaveLocks:
    """One lock per piece of state a wave of specialists shares.

    A wave runs several specialist loops in threads inside the lead's
    process (:func:`mason.tools._run_children`), so the state a child
    reaches through its parent chain has more than one writer. Every
    session in one tree holds the same instance, handed down by
    :meth:`MasonSession.spawn`. The locks are reentrant because one writer
    recurses: ``count_usage`` chains to the parent, which takes the same
    lock again.

    Nothing here is a transaction boundary for the run store. That store
    opens its own connection per call and serialises its own writes.
    """

    __slots__ = ("approval", "memories", "notebook", "observer", "setups", "usage")

    def __init__(self) -> None:
        self.usage = threading.RLock()
        self.memories = threading.RLock()
        self.setups = threading.RLock()
        self.notebook = threading.RLock()
        self.approval = threading.RLock()
        self.observer = threading.RLock()


def _beat_until_stopped(
    workspace_root: Path, session_id: str, beat_s: float, stop: threading.Event
) -> None:
    """Stamp one lease every *beat_s* seconds until *stop* is set.

    A free function, not a method: the heartbeat thread must not keep its
    session alive. A beat that cannot be written is skipped, because the
    session's work matters more than the bookkeeping.
    """
    import sqlite3

    from foundation.errors import FoundationError
    from foundation.runtime import Workspace

    while not stop.wait(beat_s):
        try:
            with Workspace(workspace_root) as ws:
                ws.beat_lease(session_id)
        except (FoundationError, sqlite3.Error, OSError):
            continue


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
        from foundation.config import apply_gpu_exclusion

        self.workspace_root = (
            Path(workspace_root) if workspace_root is not None else resolve_root(None)
        )
        # [workspace] exclude_gpus: a broken device stays out of this
        # session's budget and every launch's, unless the environment
        # already says otherwise (the sandbox exports the partition's list).
        apply_gpu_exclusion(self.cwd)
        self.approver: Approver = approver if approver is not None else _approve_nothing
        self.auto_approve = auto_approve
        # Whether a person is at the keyboard. An interactive session reads
        # Ctrl-C as "stop this turn", so the loop leaves SIGINT alone there;
        # a batch session reads it as the end of the session.
        self.interactive = False
        # API keys this session (and its delegates) read from the
        # environment, withdrawn from os.environ once read so nothing the
        # model drives can print them back. Shared with a parent session.
        self.api_keys: dict[str, str] = {}
        # The setup digests whose lines a command event of this conversation
        # already carries in full: a later event names the digest only, so
        # a campaign's transcripts hold each build's setup block once.
        # Shared with the delegations, like the key store.
        self.recorded_setups: set[str] = set()
        self.observer: Observer | None = observer
        # Which agent card this session runs as; the loop sets it from the
        # spec it resolves. Delegated child sessions carry the specialist's
        # name for attribution in approvals and notebook entries.
        self.agent_name = "pi"
        # CLI flag overrides, kept so they can be re-asserted over
        # [agent.roster.<name>] tables: a flag outranks config.
        self.flag_updates: dict[str, object] = {}
        # The benchmark condition this session runs under (mason.mechanisms)
        # and the mechanisms switched off from it, for the transcript
        # header; None is Mason as configured.
        self.condition: str | None = None
        self.ablated: tuple[str, ...] = ()
        self._parent: MasonSession | None = None
        self._children_spawned = 0
        # This session's handle when it is a delegated child: the agent name
        # and the ordinal its parent gave it, which is also the tail of its
        # transcript name. A lead's own session has none.
        self.handle: str | None = None
        # The conversation transcript this session replayed at start, when
        # it was resumed. The specialists of that conversation write beside
        # it, so a continue after a resume finds them there and not beside
        # this session's own new file. One hop back: a specialist reaches
        # the conversation that was resumed, not the one before it.
        self.resumed_from: Path | None = None
        # The specialists this session briefed, by handle, still holding
        # their messages: a lead continues one instead of briefing a fresh
        # one that would start from zero. Values are mason.loop.Mason
        # objects, typed loosely because session must not import the loop.
        # They live as long as the process; nothing is kept for a critic.
        self.children: dict[str, Any] = {}
        # The locks the specialists of a wave share; one instance per
        # session tree, so a child takes the same lock its lead would.
        self.locks = WaveLocks()
        # Unset compute_profile derives from the machine: a config that declares
        # SLURM partitions is a cluster, anything else is treated as a laptop —
        # the conservative guess, since over-sizing a calculation wastes hours
        # while under-sizing it wastes minutes.
        self.compute_profile = self.agent.compute_profile or (
            "cluster" if self.hpc.partitions else "laptop"
        )
        # What this process may use, counted once at start for the transcript
        # header. The report divides the cpu-hours and gpu-hours the
        # session's runs held by this budget over the session's wall time.
        self.budget: dict[str, int] = self._count_budget()
        self.endpoint = ""
        self.endpoint_origin = ""
        self.resolve_endpoint()
        self.notebook_path = project_files.notebook_path(self.cwd)
        self.plan_path = project_files.plan_path(self.cwd)
        self.read_files: set[Path] = set()
        # The newest answer to each catalog and material lookup this session
        # (mason.tools.FACT_TOOLS), handed to the critic with a review brief.
        self.facts: dict[str, str] = {}
        # Whether a critic has approved the plan, for a card that spends no
        # compute before one has (review_first). The loop sets it from the
        # persisted reviews at start; the review tool sets it on approval.
        self.plan_approved = False
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_prompt_tokens = 0
        self.sessions_dir = self.workspace_root / "mason" / "sessions"
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        self.transcript_path = self.sessions_dir / f"{stamp}-{os.getpid()}.jsonl"
        self._lock_handle: Any | None = None
        # The lease this session holds over its runs: open from the first
        # turn to the last, beaten while it works, closed when it ends. A
        # delegated child rides its lead's lease, so only a root opens one.
        self.lease_open = False
        self._beat_thread: threading.Thread | None = None
        self._beat_stop = threading.Event()
        self._lease_warned = False
        self._deadline: datetime | EllipsisType | None = ...
        self._software_versions: dict[str, str] | None = None
        # The machine memories this session and its delegates wrote, in
        # order: name, description, evidence, unverified, agent, and the
        # kept version the first write replaced. A delegate's list is what
        # its lead reads back; the root's scopes what forget may undo.
        self.memories_written: list[dict[str, Any]] = []

    @staticmethod
    def _count_budget() -> dict[str, int]:
        from slab.resources import budget

        return budget().counts

    def software_versions(self) -> dict[str, str]:
        """The software present now, probed once per session and then reused.

        Machine memory stamps a fact with these versions and flags a memory
        whose software has changed since. The probe runs the engine and
        builder version checks, so it happens on the first call only: the
        prompt after a compaction and every ``remember`` reuse the result.
        """
        if self._software_versions is None:
            from slab._ops import software_versions

            self._software_versions = software_versions()
        return self._software_versions

    def note_memory(self, entry: dict[str, Any]) -> None:
        """Record a memory write here and in every session above this one.

        The lead of a delegation reads the child's list after it returns,
        so a memory a grandchild wrote reaches every lead on the way up.
        """
        with self.locks.memories:
            session: MasonSession | None = self
            while session is not None:
                session.memories_written.append(entry)
                session = session._parent

    def written_memory(self, name: str) -> dict[str, Any] | None:
        """The first write of *name* in this session tree, or None.

        The first write matters because it holds the version that existed
        before the session touched the memory, which is what forget puts
        back.
        """
        root = self
        while root._parent is not None:
            root = root._parent
        return next((e for e in root.memories_written if e["name"] == name), None)

    # -- one running session per workspace ------------------------------------

    def acquire_session_lock(self) -> None:
        """Refuse to run alongside another mason in the same project directory.

        Two concurrent sessions in one project interleave ``NOTEBOOK.md``
        entries and race each other's view of ``PLAN.md``, so the loop takes
        an advisory lock before its first turn. Both files are the
        project's, so the lock is keyed by the project directory: two
        campaigns in two projects share a workspace freely (the run store
        journals for that). Contention is refused loudly, naming the
        holder. Delegated children never call this — they run inside the
        parent's lock. A filesystem that cannot lock (some parallel
        filesystems) degrades to a warning: an undetected concurrent session
        beats a project nobody can use. ``[agent] session_lock = false``
        turns the lock off.
        """
        if not self.agent.session_lock or self._lock_handle is not None:
            return
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX platform
            return
        lock_path = self.session_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "a+", encoding="utf-8")  # noqa: SIM115 - held for the process's life
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.seek(0)
            holder = handle.read().strip() or "holder unknown"
            handle.close()
            raise MasonError(
                f"another mason session is already working in this project "
                f"directory ({holder}). Finish or stop it first, run the new "
                f"session from another project directory, or set [agent] "
                f"session_lock = false to run concurrent sessions here anyway."
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

    def session_lock_path(self) -> Path:
        """Where this project's lock lives: under the workspace, keyed by the project.

        The real path of the project directory is hashed so a symlinked
        and a direct path to one project take the same lock.
        """
        digest = hashlib.sha256(str(self.cwd.resolve()).encode("utf-8")).hexdigest()[:16]
        return self.sessions_dir.parent / "locks" / f"{digest}.lock"

    def release_session_lock(self) -> None:
        """Let the workspace go (normally implicit in process exit).

        The lock rides the open file handle, so a finished process releases
        it without help. This exists for the caller that ends one session
        and starts another in the same process — a resume, a test.
        """
        if self._lock_handle is not None:
            self._lock_handle.close()
            self._lock_handle = None


    # -- the lease this session holds over its runs ---------------------------

    def open_lease(self) -> None:
        """Claim this session's runs for as long as the session lives.

        The lease is what makes a dead session's runs settleable from
        another process: a reader that finds it ended, past its job's end,
        or silent fails those runs instead of reading them as active. A
        heartbeat thread stamps it every ``[agent] lease_beat_s`` seconds,
        and the loop stamps it at every step as well, so a session working
        through a long tool call is never taken for dead. A workspace that
        cannot be opened is not a reason to refuse the session: the first
        workspace tool call names the fault with its recovery.
        """
        if self._parent is not None or self.lease_open:
            return
        if not self._with_workspace(
            lambda ws: ws.open_lease(
                self.session_id, harness="mason", agent=self.agent_name, cwd=self.cwd
            )
        ):
            return
        self.lease_open = True
        self._beat_stop.clear()
        # The thread holds the workspace root and the id, never the session:
        # a session nobody closed must still be collectable, and its lock
        # released with it.
        thread = threading.Thread(
            target=_beat_until_stopped,
            args=(self.workspace_root, self.session_id, self.lease_beat_s(), self._beat_stop),
            name="mason-lease",
            daemon=True,
        )
        self._beat_thread = thread
        thread.start()

    def lease_beat_s(self) -> float:
        """How often this session beats: the config, and never a slow tenth.

        ``[workspace] lease_silence_s`` is how long a reader waits before
        it settles a silent session's runs, and a session beats ten times
        inside that window, so a beat slower than a tenth of it is brought
        down to one.
        """
        from foundation.config import lease_silence_s

        return min(self.agent.lease_beat_s, lease_silence_s(self.cwd) / 10)

    def beat_lease(self) -> None:
        """Say this session is still working (cheap enough for every step)."""
        root = self
        while root._parent is not None:
            root = root._parent
        if not root.lease_open:
            return
        root._with_workspace(lambda ws: ws.beat_lease(root.session_id))

    def end_lease(self, reason: str) -> list[Any]:
        """Close the lease and end the runs this session was still executing.

        Whatever ended the session, its runs end with it: each one is
        stopped here and marked failed naming the session and *reason*, so
        no later reader finds a run of a session that is over. Returns the
        runs ended.
        """
        if self._parent is not None or not self.lease_open:
            return []
        self.lease_open = False
        self._beat_stop.set()
        if self._beat_thread is not None:
            self._beat_thread.join(timeout=2.0)
            self._beat_thread = None
        ended: list[Any] = []

        def close(ws: Any) -> None:
            _lease, runs = ws.end_session(self.session_id, reason=reason)
            ended.extend(runs)

        self._with_workspace(close)
        return ended

    def close(self, reason: str = "finished") -> list[Any]:
        """End this session: record why, close the lease, release the lock.

        Every exit runs through here, so a session that stops for any
        reason leaves no run at ``running`` behind it and no later reader
        has to guess. Returns the runs ended with the session.
        """
        if self._parent is not None:
            return []
        ended = self.end_lease(reason)
        self.record(
            {
                "type": "session_end",
                "reason": reason,
                "runs_ended": [run.id for run in ended],
            }
        )
        self.release_session_lock()
        return ended

    def job_ends_at(self) -> datetime | None:
        """When this session's job ends, or None outside a job (read once)."""
        if self._deadline is ...:
            from foundation.runtime import job_deadline

            self._deadline = job_deadline()
        return self._deadline

    def _with_workspace(self, action: Callable[[Any], Any]) -> bool:
        """Run *action* on the run store; report a fault once and carry on.

        A beat that cannot be written must never raise into the loop: the
        session's work matters more than the bookkeeping, and a silent
        lease is settled by the next reader anyway.
        """
        import sqlite3

        from foundation.errors import FoundationError
        from foundation.runtime import Workspace

        try:
            with Workspace(self.workspace_root) as ws:
                action(ws)
        except (FoundationError, sqlite3.Error, OSError) as e:
            if not self._lease_warned:
                self._lease_warned = True
                warnings.warn(
                    f"this session's lease could not be written ({e}); its runs will "
                    f"be settled by whoever reads them next",
                    stacklevel=2,
                )
            return False
        return True

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

    def spawn(
        self, agent_name: str, agent: AgentConfig, *, handle: str | None = None
    ) -> MasonSession:
        """A delegated child session: shared gate and memory, its own transcript.

        The child shares the project directory, the workspace, the ``[hpc]``
        view, the approver, the auto-approve policy, and the observer — one
        permission regime and one terminal per session, whoever asks. Its
        token usage chains upward so the parent's totals stay whole-session
        truths. Fresh per child: the
        read-files staleness guard (a specialist must read a file before
        editing it even when the parent read it), and the transcript, named
        after the parent's with the agent and an ordinal so ``--resume``
        can tell conversations from delegations apart. That tail is the
        child's handle. Pass *handle* to take an existing one instead of a
        new ordinal, which is how a specialist continued in a later process
        writes on into its own transcript.
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
        child.api_keys = self.api_keys  # one key store per conversation
        child.recorded_setups = self.recorded_setups  # one full copy per conversation
        child.locks = self.locks  # one set of locks per session tree
        child._software_versions = self._software_versions  # probed once, if at all
        # A flag outranks config for everyone: the child's loop re-asserts
        # these over its own [agent.roster] table exactly as the parent did.
        child.flag_updates = dict(self.flag_updates)
        if handle is None:
            self._children_spawned += 1
            handle = f"{agent_name}-{self._children_spawned}"
        child.handle = handle
        child.transcript_path = self.transcript_path.with_name(
            f"{self.transcript_path.stem}-{handle}.jsonl"
        )
        return child

    def resume_from_transcript(self, transcript: Path) -> None:
        """Note the conversation this session replays, and adopt its handles.

        The specialists of that conversation keep their transcripts and
        their handles, so a brief with ``continues`` reaches them. This
        session's next child takes the next ordinal after the highest they
        used, and never shadows one.
        """
        self.resumed_from = transcript
        highest = 0
        for path in transcript.parent.glob(f"{transcript.stem}-*.jsonl"):
            ordinal = path.stem.rpartition("-")[2]
            if ordinal.isdigit():
                highest = max(highest, int(ordinal))
        self._children_spawned = max(self._children_spawned, highest)

    @property
    def session_id(self) -> str:
        """The chat's id, stamped on every run this session launches.

        The value is the root transcript's stem, so one chat has one id and a
        delegated specialist's runs join the chat that asked for them. Foundation
        stores it on each run, which is what makes ``slab promote
        --session <id>`` able to promote a whole conversation's results.
        """
        if self._parent is not None:
            return self._parent.session_id
        return self.transcript_path.stem

    def observe(self, kind: str, text: str) -> None:
        """Send one line to the session's observer, one writer at a time.

        A wave of specialists shares the lead's terminal, so the lock keeps
        each line whole. The attribution already names the agent.
        """
        observer = self.observer
        if observer is None:
            return
        with self.locks.observer:
            observer(kind, self.attribution(), text)

    def attribution(self) -> str:
        """The ``[agent]`` marker for approval previews — children only."""
        return f"[{self.agent_name}] " if self._parent is not None else ""

    # -- permission gate ------------------------------------------------------

    def allows(self, tool_name: str, preview: str, *, requires_approval: bool) -> bool:
        """Whether this tool call may run under the session's approval policy."""
        if not requires_approval or self.auto_approve or self.agent.approval == "auto":
            return True
        # One question at a time: a wave's specialists share the terminal,
        # and the preview names whose call it is.
        with self.locks.approval:
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
        """Append one entry to the lab notebook (created on first write).

        The notebook is a curated record: entries only land here when an
        agent (or a person) calls it a result. Machinery like context
        compaction writes elsewhere — see :meth:`compactions_append`."""
        # The notebook is shared across the whole session (it is the group's
        # blackboard), so a delegated agent's entries carry its name. The
        # file itself is the project's: foundation writes it the same way
        # for every client.
        # One writer at a time, so two specialists of a wave never
        # interleave the lines of one entry.
        with self.locks.notebook:
            project_files.notebook_append(
                self.cwd,
                entry,
                heading=heading,
                author=self.agent_name if self._parent is not None else None,
            )

    @property
    def compactions_path(self) -> Path:
        """Where context-compaction summaries land, one file per session.

        A sibling of the JSONL transcript. Not the lab notebook: the notebook
        is what the agent decided to keep, this is what the harness folded
        because the window filled. Reading it back is a debugging surface,
        not the running context — the summary is also prepended into the
        rebuilt conversation as a user message, and recorded in the
        transcript, so nothing here is load-bearing at run time."""
        return self.sessions_dir / f"{self.transcript_path.stem}.compactions.md"

    def compactions_append(self, summary: str) -> None:
        """Persist one context-compaction summary to the per-session file."""
        stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
        block = f"\n## {stamp} — context compaction\n\n{summary.rstrip()}\n"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        if not self.compactions_path.exists():
            block = "# Context compactions\n" + block
        with open(self.compactions_path, "a", encoding="utf-8") as handle:
            handle.write(block)

    def notebook_tail(self, max_chars: int = 3_000) -> str:
        """The notebook's last entries, budget-capped for the context."""
        return project_files.notebook_tail(self.cwd, max_chars)

    def plan_text(self) -> str:
        """The current plan, or empty when none has been written yet."""
        return project_files.plan_read(self.cwd)

    # -- transcript -----------------------------------------------------------

    def recorded(self, type_: str) -> list[dict[str, Any]]:
        """The events of one type this session's transcript holds, in order."""
        events: list[dict[str, Any]] = []
        try:
            with open(self.transcript_path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(event, dict) and event.get("type") == type_:
                        events.append(event)
        except OSError:
            return []
        return events

    def record(self, event: dict[str, Any]) -> None:
        """Append one event to the session transcript (JSONL, append-only)."""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        stamped = {"at": datetime.now(UTC).isoformat(), **event}
        with open(self.transcript_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(stamped, ensure_ascii=False) + "\n")

    def count_usage(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cached_prompt_tokens: int | None = None,
    ) -> None:
        with self.locks.usage:
            if prompt_tokens:
                self.prompt_tokens += prompt_tokens
            if completion_tokens:
                self.completion_tokens += completion_tokens
            if cached_prompt_tokens:
                self.cached_prompt_tokens += cached_prompt_tokens
            if self._parent is not None:
                self._parent.count_usage(prompt_tokens, completion_tokens, cached_prompt_tokens)

    def usage_text(self) -> str:
        """``P+C`` tokens, with the cached share when the server reported one.

        Examples:
            >>> import types
            >>> fake = types.SimpleNamespace(prompt_tokens=300, completion_tokens=20,
            ...                              cached_prompt_tokens=0)
            >>> MasonSession.usage_text(fake)
            '300+20'
            >>> fake.cached_prompt_tokens = 210
            >>> MasonSession.usage_text(fake)
            '300+20 (210 cached)'
        """
        text = f"{self.prompt_tokens}+{self.completion_tokens}"
        if self.cached_prompt_tokens:
            text += f" ({self.cached_prompt_tokens} cached)"
        return text

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
                    if not isinstance(event, dict):
                        raise ValueError("not a JSON object")
                    if event.get("type") == "message":
                        messages.append(event["message"])
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
            raise SessionError(f"cannot resume from {transcript} (line {number}): {e}") from e
        return messages
