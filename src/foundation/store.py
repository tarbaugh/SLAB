"""Run persistence: the :class:`RunStore` protocol and the SQLite implementation.

The protocol is the seam for swapping backends later (e.g. Postgres). The SQLite
implementation is single-file and zero-config; ``":memory:"`` gives an ephemeral
store for tests.

Concurrency model: every mutating operation runs inside ``BEGIN IMMEDIATE`` and
revalidates against the state read *inside* the transaction, so concurrent
transitions from threads, processes, or stale handles serialize safely — exactly
one writer wins and the loser gets a precise :class:`~foundation.errors.IllegalTransitionError`.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, cast, runtime_checkable

from foundation.errors import (
    AmbiguousRunIdError,
    AmbiguousSessionError,
    ArtifactExistsError,
    ArtifactNotFoundError,
    IllegalTransitionError,
    RunExistsError,
    RunNotFoundError,
    RunStateError,
    SchemaVersionError,
    SessionNotFoundError,
    StorageError,
)
from foundation.lifecycle import (
    ExecutionStatus,
    LifecycleState,
    is_terminal,
    requires_force,
    validate_status_change,
    validate_transition,
)
from foundation.models import (
    ArtifactRef,
    ArtifactRole,
    CheckResult,
    Run,
    SessionSummary,
    TaskRecord,
    Transition,
    utcnow,
)

SCHEMA_VERSION = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL DEFAULT '',
    state            TEXT NOT NULL,
    status           TEXT NOT NULL,
    intent           TEXT,
    session          TEXT,
    meta             TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    state_entered_at TEXT NOT NULL,
    started_at       TEXT,
    finished_at      TEXT,
    error            TEXT,
    failure          TEXT
);
CREATE INDEX IF NOT EXISTS ix_runs_state ON runs(state);
CREATE INDEX IF NOT EXISTS ix_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS ix_runs_created_at ON runs(created_at);
CREATE INDEX IF NOT EXISTS ix_runs_session ON runs(session);
CREATE TABLE IF NOT EXISTS transitions (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    from_state  TEXT NOT NULL,
    to_state    TEXT NOT NULL,
    actor       TEXT NOT NULL,
    reason      TEXT,
    forced      INTEGER NOT NULL DEFAULT 0,
    at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_transitions_run_id ON transitions(run_id);
CREATE TABLE IF NOT EXISTS artifacts (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    role        TEXT NOT NULL,
    hash        TEXT NOT NULL,
    size_bytes  INTEGER NOT NULL,
    recipe      TEXT,
    created_at  TEXT NOT NULL,
    UNIQUE (run_id, name)
);
CREATE INDEX IF NOT EXISTS ix_artifacts_run_id ON artifacts(run_id);
CREATE INDEX IF NOT EXISTS ix_artifacts_hash ON artifacts(hash);
CREATE TABLE IF NOT EXISTS tasks (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    status      TEXT NOT NULL,
    cache_hit   INTEGER NOT NULL DEFAULT 0,
    cache_key   TEXT NOT NULL,
    recipe      TEXT NOT NULL,
    inputs      TEXT NOT NULL,
    outputs     TEXT NOT NULL,
    error       TEXT,
    failure     TEXT,
    started_at  TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_tasks_run_id ON tasks(run_id);
CREATE INDEX IF NOT EXISTS ix_tasks_cache_key ON tasks(cache_key);
CREATE TABLE IF NOT EXISTS checks (
    seq       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    name      TEXT NOT NULL,
    kind      TEXT NOT NULL,
    passed    INTEGER NOT NULL,
    message   TEXT NOT NULL DEFAULT '',
    observed  TEXT NOT NULL DEFAULT 'null',
    expected  TEXT NOT NULL DEFAULT 'null',
    at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_checks_run_id ON checks(run_id);
"""

# Statements upgrading an existing database from (version - 1) to version.
# Fresh databases get _SCHEMA directly and skip these.
_MIGRATIONS: dict[int, tuple[str, ...]] = {
    2: (  # structured failure evidence on runs and tasks
        "ALTER TABLE runs ADD COLUMN failure TEXT",
        "ALTER TABLE tasks ADD COLUMN failure TEXT",
    ),
    3: (  # which client session created the run
        "ALTER TABLE runs ADD COLUMN session TEXT",
        "CREATE INDEX IF NOT EXISTS ix_runs_session ON runs(session)",
    ),
}


#: Filesystem types whose files may be open from several hosts at once. WAL
#: keeps its index in shared memory, which only the processes of one host
#: see, so a second host writing the same database through WAL can corrupt
#: it. Rollback journaling locks the file itself and works across hosts
#: wherever the filesystem honors POSIX locks.
NETWORK_FILESYSTEMS = frozenset(
    {
        "nfs",
        "nfs4",
        "lustre",
        "gpfs",
        "beegfs",
        "panfs",
        "ceph",
        "glusterfs",
        "cifs",
        "smb3",
        "smbfs",
        "afs",
        "9p",
        "virtiofs",
        "fuse.sshfs",
    }
)


def _unescape_mount(field: str) -> str:
    """Undo the octal escapes ``/proc/mounts`` uses for spaces and tabs."""
    return field.encode().decode("unicode_escape") if "\\" in field else field


def on_network_filesystem(path: str | os.PathLike[str], mounts: str | None = None) -> bool:
    """Whether *path* sits on a filesystem shared between hosts.

    Reads the mount table (``/proc/self/mounts`` on Linux; *mounts* is that
    text, for tests) and takes the type of the longest mount point that
    prefixes the path's real location. Platforms without a mount table read
    as local, and so does an unreadable one: a wrong "local" costs WAL's
    fallback path, a wrong "network" only speed.

    Examples:
        >>> table = "/dev/sda1 / ext4 rw 0 0\\nlustre@tcp:/fs /scratch lustre rw 0 0\\n"
        >>> on_network_filesystem("/scratch/me/ws/runs.db", table)
        True
        >>> on_network_filesystem("/home/me/ws/runs.db", table)
        False
    """
    if mounts is None:
        try:
            mounts = Path("/proc/self/mounts").read_text(encoding="utf-8")
        except OSError:
            return False
    target = os.path.realpath(os.path.dirname(os.fspath(path)) or ".")
    best: tuple[int, str] | None = None
    for line in mounts.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        point = _unescape_mount(fields[1]).rstrip("/") or "/"
        under = target == point or target.startswith(point.rstrip("/") + "/")
        if under and (best is None or len(point) > best[0]):
            best = (len(point), fields[2])
    return best is not None and best[1] in NETWORK_FILESYSTEMS


def journal_mode_for(path: str | os.PathLike[str]) -> str:
    """``wal`` on a local disk, ``delete`` on a network filesystem.

    ``$SLAB_SQLITE_JOURNAL`` (``wal`` or ``delete``) overrides the
    detection, for a filesystem the table does not name or a shared mount
    known to be used from one host only.

    Examples:
        >>> import os
        >>> os.environ["SLAB_SQLITE_JOURNAL"] = "delete"
        >>> journal_mode_for("/anywhere/runs.db")
        'delete'
        >>> del os.environ["SLAB_SQLITE_JOURNAL"]
    """
    override = os.environ.get("SLAB_SQLITE_JOURNAL", "").strip().lower()
    if override in ("wal", "delete"):
        return override
    return "delete" if on_network_filesystem(path) else "wal"


def _journal_mode(conn: sqlite3.Connection) -> str:
    return str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()


def settle_journal_mode(conn: sqlite3.Connection, wanted: str) -> str:
    """Ask the database for *wanted* journaling; keep what it can give.

    Leaving WAL needs every other connection closed, and entering it needs
    a filesystem with shared memory. SQLite reports both refusals as an
    ``OperationalError``, and neither is a reason to refuse the open: the
    store works in the other mode, only slower or only from this host, and
    the switch lands the next time the file is opened with nothing else
    holding it. The case that matters is a running campaign holding the
    database in WAL mode while a newer SLAB asks for rollback journaling.
    Refusing the open deadlocks the workspace under a hot upgrade, and four
    refused opens once did exactly that.

    Returns the mode in force after the attempt.

    Examples:
        >>> import os, tempfile
        >>> conn = sqlite3.connect(os.path.join(tempfile.mkdtemp(), "runs.db"))
        >>> settle_journal_mode(conn, "delete")
        'delete'
        >>> settle_journal_mode(conn, "wal")
        'wal'
        >>> settle_journal_mode(conn, "wal")
        'wal'
        >>> conn.close()
    """
    current = _journal_mode(conn)
    if current == wanted:
        return current
    try:
        conn.execute(f"PRAGMA journal_mode = {wanted.upper()}")
    except sqlite3.OperationalError:
        if wanted == "wal":
            # A filesystem that looked local but refuses WAL's shared-memory
            # index: rollback journaling is slower but works wherever the
            # filesystem locks at all, and refusing to open would be worse.
            with suppress(sqlite3.OperationalError):
                conn.execute("PRAGMA journal_mode = DELETE")
    return _journal_mode(conn)


@runtime_checkable
class RunStore(Protocol):
    """Storage interface for runs. Implement this to add a backend (e.g. Postgres)."""

    def create(self, run: Run) -> Run:
        """Persist a new run."""
        ...

    def get(self, run_id: str) -> Run:
        """Fetch a run by full id or unique prefix."""
        ...

    def resolve(self, run_id: str) -> str:
        """Resolve a full id or unique prefix to the full run id."""
        ...

    def resolve_session(self, session: str) -> str:
        """Resolve a full session id or unique prefix to the full session id."""
        ...

    def list_runs(
        self,
        *,
        state: LifecycleState | str | None = None,
        status: ExecutionStatus | str | None = None,
        session: str | None = None,
        limit: int | None = None,
    ) -> list[Run]:
        """List runs, newest first, optionally filtered."""
        ...

    def list_sessions(self, *, limit: int | None = None) -> list[SessionSummary]:
        """Summarize the sessions that created runs, newest first."""
        ...

    def transition(
        self,
        run_id: str,
        to_state: LifecycleState | str,
        *,
        actor: str = "user",
        reason: str | None = None,
        force: bool = False,
        expected: LifecycleState | str | None = None,
    ) -> Run:
        """Atomically move a run to a new lifecycle state."""
        ...

    def set_status(
        self,
        run_id: str,
        status: ExecutionStatus | str,
        *,
        error: str | None = None,
        failure: dict[str, object] | None = None,
    ) -> Run:
        """Change a run's execution status."""
        ...

    def set_intent(self, run_id: str, intent: str | None) -> Run:
        """Set or clear a run's intent note."""
        ...

    def history(self, run_id: str) -> list[Transition]:
        """Return the run's lifecycle transitions, oldest first."""
        ...

    def delete_run(self, run_id: str) -> Run:
        """Delete one expired run and every row that references it."""
        ...

    def add_task(self, record: TaskRecord) -> TaskRecord:
        """Record a traced task call on its run."""
        ...

    def update_task(
        self,
        seq: int,
        *,
        status: ExecutionStatus | str,
        outputs: dict[str, str] | None = None,
        error: str | None = None,
        failure: dict[str, object] | None = None,
        finished_at: datetime | None = None,
    ) -> TaskRecord:
        """Finalize a provisional (running) task row."""
        ...

    def list_tasks(self, run_id: str) -> list[TaskRecord]:
        """List a run's traced task calls, oldest first."""
        ...

    def find_cached_task(self, cache_key: str) -> TaskRecord | None:
        """Return the most recent completed task with this cache key, if any."""
        ...

    def add_check_results(self, run_id: str, results: Sequence[CheckResult]) -> list[CheckResult]:
        """Record verification results on a run."""
        ...

    def list_check_results(self, run_id: str) -> list[CheckResult]:
        """List a run's verification results, oldest first."""
        ...

    def add_artifact(
        self,
        run_id: str,
        *,
        name: str,
        role: ArtifactRole | str,
        hash: str,
        size_bytes: int,
        recipe: dict[str, object] | None = None,
    ) -> ArtifactRef:
        """Record an artifact reference on a run."""
        ...

    def list_artifacts(
        self, run_id: str, *, role: ArtifactRole | str | None = None
    ) -> list[ArtifactRef]:
        """List a run's artifact references, oldest first."""
        ...

    def get_artifact(self, run_id: str, name: str) -> ArtifactRef:
        """Fetch one of the run's artifact references by name."""
        ...

    def close(self) -> None:
        """Release underlying resources."""
        ...


class SQLiteRunStore:
    """Single-file SQLite run store. Zero-config; safe across threads and processes.

    Args:
        path: Database file (created, parents included, if missing), or
            ``":memory:"`` for an ephemeral in-memory store.

    Examples:
        >>> store = SQLiteRunStore(":memory:")
        >>> run = store.create(Run(name="si-relax"))
        >>> store.get(run.id).state.value
        'quarantined'
        >>> store.close()
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._path = str(path)
        if self._path != ":memory:":
            resolved = Path(self._path).expanduser()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self._path = str(resolved)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False, isolation_level=None)
        try:
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA busy_timeout = 5000")
            #: The journaling this location calls for: ``wal`` or ``delete``.
            self.journal_mode_wanted = (
                journal_mode_for(self._path) if self._path != ":memory:" else "wal"
            )
            #: The journaling actually in force. It differs from the wanted
            #: mode while another process holds the database open in the
            #: other one; see :func:`settle_journal_mode`.
            self.journal_mode = settle_journal_mode(self._conn, self.journal_mode_wanted)
            self._init_schema()
        except BaseException:
            # A connection that outlives a failed open keeps the database
            # open in whatever mode it found, and blocks every later attempt
            # to switch: four such leaks once deadlocked a shared workspace
            # for an hour.
            self._conn.close()
            raise

    # -- lifecycle of the store itself ------------------------------------------------

    def close(self) -> None:
        """Close the underlying connection. Further operations raise.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> store.close()
        """
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"SQLiteRunStore({self._path!r})"

    # -- writes -----------------------------------------------------------------------

    def create(self, run: Run) -> Run:
        """Persist a new run and return it unchanged.

        Raises:
            RunExistsError: A run with this id already exists.
            StorageError: ``run.meta`` is not JSON-serializable.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> run = store.create(Run(name="si-relax", intent="baseline"))
            >>> store.get(run.id).intent
            'baseline'
            >>> store.close()
        """
        try:
            meta_json = json.dumps(run.meta, sort_keys=True)
        except TypeError as e:
            raise StorageError(f"run.meta must be JSON-serializable: {e}") from e
        with self._txn() as conn:
            try:
                conn.execute(
                    "INSERT INTO runs (id, name, state, status, intent, session, meta,"
                    " created_at, updated_at, state_entered_at, started_at, finished_at,"
                    " error, failure)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        run.id,
                        run.name,
                        run.state.value,
                        run.status.value,
                        run.intent,
                        run.session,
                        meta_json,
                        run.created_at.isoformat(),
                        run.updated_at.isoformat(),
                        run.state_entered_at.isoformat(),
                        _fmt_dt(run.started_at),
                        _fmt_dt(run.finished_at),
                        run.error,
                        _fmt_json(run.failure),
                    ),
                )
            except sqlite3.IntegrityError as e:
                raise RunExistsError(run.id) from e
        return run

    def transition(
        self,
        run_id: str,
        to_state: LifecycleState | str,
        *,
        actor: str = "user",
        reason: str | None = None,
        force: bool = False,
        expected: LifecycleState | str | None = None,
    ) -> Run:
        """Atomically move a run to *to_state*; return the updated snapshot.

        The run's current state is re-read inside the transaction and the
        transition validated against it, so racing callers serialize: one wins,
        the rest get :class:`~foundation.errors.IllegalTransitionError` describing the
        *actual* current state. Pass ``expected`` for compare-and-swap semantics:
        the transition is refused unless the run is still in that state — the
        right tool for automated sweeps deciding from a possibly-stale listing.

        Every transition is recorded with ``actor``, ``reason``, and a ``forced``
        flag. ``forced`` is true only when the transition needed ``force=True``
        (a force-promotion) — passing ``force=True`` on a normally-legal
        transition records ``forced=False``. The run's ``state_entered_at``
        clock resets, which is what retention TTLs anchor to.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> r = store.create(Run(name="demo"))
            >>> store.transition(r.id, "verified", actor="checks", reason="fmax<0.05").state.value
            'verified'
            >>> store.transition(r.id, LifecycleState.PROMOTED, expected="verified").state.value
            'promoted'
            >>> [t.to_state.value for t in store.history(r.id)]
            ['verified', 'promoted']
            >>> store.close()
        """
        to = LifecycleState(to_state)
        want = None if expected is None else LifecycleState(expected)
        with self._txn() as conn:
            rid = self._resolve(run_id)
            row = conn.execute("SELECT state FROM runs WHERE id = ?", (rid,)).fetchone()
            current = LifecycleState(row["state"])
            if want is not None and current is not want:
                raise IllegalTransitionError(
                    current,
                    to,
                    detail=f"expected state {want.value!r}, found {current.value!r}",
                )
            validate_transition(current, to, force=force)
            now = utcnow().isoformat()
            cur = conn.execute(
                "UPDATE runs SET state = ?, updated_at = ?, state_entered_at = ?"
                " WHERE id = ? AND state = ?",
                (to.value, now, now, rid, current.value),
            )
            if cur.rowcount != 1:  # pragma: no cover - unreachable: _txn serializes writers
                raise StorageError(f"concurrent modification of run {rid}")
            conn.execute(
                "INSERT INTO transitions (run_id, from_state, to_state, actor, reason,"
                " forced, at) VALUES (?,?,?,?,?,?,?)",
                (
                    rid,
                    current.value,
                    to.value,
                    actor,
                    reason,
                    int(requires_force(current, to)),
                    now,
                ),
            )
            return self._get_exact(conn, rid)

    def set_status(
        self,
        run_id: str,
        status: ExecutionStatus | str,
        *,
        error: str | None = None,
        failure: dict[str, object] | None = None,
    ) -> Run:
        """Change a run's execution status; return the updated snapshot.

        Stamps ``started_at`` when entering ``running`` and ``finished_at`` when
        entering ``completed`` or ``failed``. A run served from cache may go
        ``pending -> completed`` directly, finishing without ever starting.
        Pass ``error`` (a one-liner) and/or ``failure`` (structured evidence,
        see :func:`foundation.errors.failure_record`) — only with status ``failed`` —
        to record why.

        Raises:
            IllegalStatusChangeError: The change is not permitted
                (e.g. ``completed -> running``).
            ValueError: ``error``/``failure`` was passed with a non-``failed``
                status.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> r = store.create(Run())
            >>> store.set_status(r.id, "running").status.value
            'running'
            >>> done = store.set_status(r.id, "failed", error="OOM killed",
            ...                         failure={"type": "Killed", "message": "OOM"})
            >>> (done.finished_at is not None, done.error, done.failure["type"])
            (True, 'OOM killed', 'Killed')
            >>> store.close()
        """
        new = ExecutionStatus(status)
        if (error is not None or failure is not None) and new is not ExecutionStatus.FAILED:
            raise ValueError(
                "error=/failure= are only recordable when setting status to 'failed'"
            )
        with self._txn() as conn:
            rid = self._resolve(run_id)
            row = conn.execute(
                "SELECT status, started_at, finished_at FROM runs WHERE id = ?", (rid,)
            ).fetchone()
            current = ExecutionStatus(row["status"])
            validate_status_change(current, new)
            now = utcnow().isoformat()
            sets = ["status = ?", "updated_at = ?"]
            params: list[object] = [new.value, now]
            if new is ExecutionStatus.RUNNING and row["started_at"] is None:
                sets.append("started_at = ?")
                params.append(now)
            if (
                new in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED)
                and row["finished_at"] is None
            ):
                sets.append("finished_at = ?")
                params.append(now)
            if error is not None:
                sets.append("error = ?")
                params.append(error)
            if failure is not None:
                sets.append("failure = ?")
                params.append(_fmt_json(failure))
            params.append(rid)
            conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id = ?", params)
            return self._get_exact(conn, rid)

    def set_intent(self, run_id: str, intent: str | None) -> Run:
        """Set (or clear, with ``None``) a run's intent note; return the updated snapshot.

        Allowed in any lifecycle state — intent is annotation, not data, and
        post-hoc narrative ("this was the good one") is worth capturing.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> r = store.create(Run())
            >>> store.set_intent(r.id, "probe k-mesh sensitivity").intent
            'probe k-mesh sensitivity'
            >>> store.close()
        """
        with self._txn() as conn:
            rid = self._resolve(run_id)
            conn.execute(
                "UPDATE runs SET intent = ?, updated_at = ? WHERE id = ?",
                (intent, utcnow().isoformat(), rid),
            )
            return self._get_exact(conn, rid)

    def add_artifact(
        self,
        run_id: str,
        *,
        name: str,
        role: ArtifactRole | str,
        hash: str,
        size_bytes: int,
        recipe: dict[str, object] | None = None,
    ) -> ArtifactRef:
        """Record an artifact reference on a run; return it.

        This records *metadata only* — put the bytes in an
        :class:`~foundation.artifacts.ArtifactStore` first and pass the returned hash.
        The ``recipe`` (inputs, code and engine versions, parameters) is what
        makes hash-and-discard honest: enough to recompute the artifact after
        its bytes are dropped. Names are unique within a run.

        Raises:
            ArtifactExistsError: The run already has an artifact with this name.
            RunStateError: The run is in a terminal state (expired/archived).
            StorageError: ``recipe`` is not JSON-serializable.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> r = store.create(Run(name="demo"))
            >>> ref = store.add_artifact(
            ...     r.id, name="relaxed.xyz", role="terminal",
            ...     hash="ab" * 32, size_bytes=1234,
            ...     recipe={"task": "relax", "engine": "mace==0.3.5"},
            ... )
            >>> ref.role.value
            'terminal'
            >>> store.close()
        """
        try:
            recipe_json = None if recipe is None else json.dumps(recipe, sort_keys=True)
        except TypeError as e:
            raise StorageError(f"artifact recipe must be JSON-serializable: {e}") from e
        with self._txn() as conn:
            rid = self._resolve(run_id)
            row = conn.execute("SELECT state FROM runs WHERE id = ?", (rid,)).fetchone()
            state = LifecycleState(row["state"])
            if is_terminal(state):
                raise RunStateError(rid, state, "add artifacts to")
            ref = ArtifactRef(
                run_id=rid,
                name=name,
                role=ArtifactRole(role),
                hash=hash,
                size_bytes=size_bytes,
                recipe=recipe,
            )
            try:
                conn.execute(
                    "INSERT INTO artifacts (run_id, name, role, hash, size_bytes, recipe,"
                    " created_at) VALUES (?,?,?,?,?,?,?)",
                    (
                        rid,
                        ref.name,
                        ref.role.value,
                        ref.hash,
                        ref.size_bytes,
                        recipe_json,
                        ref.created_at.isoformat(),
                    ),
                )
            except sqlite3.IntegrityError as e:
                raise ArtifactExistsError(rid, name) from e
        return ref

    def add_task(self, record: TaskRecord) -> TaskRecord:
        """Record a traced task call; return it with its store-assigned ``seq``.

        The tracer records a *provisional* row (status ``running``) before a
        task executes — making the task's input references visible to gc for
        the whole execution — and finalizes it with :meth:`update_task`.
        Cache hits and already-finished work are recorded directly as
        ``completed``/``failed``. ``pending`` rows make no sense here.

        Raises:
            ValueError: The status is ``pending``.
            RunStateError: The run is in a terminal lifecycle state.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> r = store.create(Run())
            >>> rec = store.add_task(TaskRecord(
            ...     run_id=r.id, name="relax", status="completed", cache_key="ab" * 32,
            ...     inputs={"atoms": "cd" * 32}, outputs={"return": "ef" * 32},
            ...     started_at=utcnow(),
            ... ))
            >>> rec.seq > 0
            True
            >>> store.close()
        """
        if record.status is ExecutionStatus.PENDING:
            raise ValueError(
                "tasks are recorded at execution time: status must be 'running',"
                " 'completed', or 'failed', got 'pending'"
            )
        with self._txn() as conn:
            rid = self._resolve(record.run_id)
            row = conn.execute("SELECT state FROM runs WHERE id = ?", (rid,)).fetchone()
            state = LifecycleState(row["state"])
            if is_terminal(state):
                raise RunStateError(rid, state, "record tasks on")
            cursor = conn.execute(
                "INSERT INTO tasks (run_id, name, status, cache_hit, cache_key, recipe,"
                " inputs, outputs, error, failure, started_at, finished_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    rid,
                    record.name,
                    record.status.value,
                    int(record.cache_hit),
                    record.cache_key,
                    json.dumps(record.recipe, sort_keys=True, default=repr),
                    json.dumps(record.inputs, sort_keys=True),
                    json.dumps(record.outputs, sort_keys=True),
                    record.error,
                    _fmt_json(record.failure),
                    record.started_at.isoformat(),
                    _fmt_dt(record.finished_at),
                ),
            )
            task_row = conn.execute(
                "SELECT * FROM tasks WHERE seq = ?", (cursor.lastrowid,)
            ).fetchone()
        return _row_to_task(task_row)

    def update_task(
        self,
        seq: int,
        *,
        status: ExecutionStatus | str,
        outputs: dict[str, str] | None = None,
        error: str | None = None,
        failure: dict[str, object] | None = None,
        finished_at: datetime | None = None,
    ) -> TaskRecord:
        """Finalize a provisional (``running``) task row; return the updated record.

        ``error`` (one-liner) and ``failure`` (structured evidence, see
        :func:`foundation.errors.failure_record`) accompany status ``failed``.

        Raises:
            ValueError: No task with this ``seq``, the row is already final,
                *status* is not final, or ``error``/``failure`` accompanies
                a ``completed`` status.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> r = store.create(Run())
            >>> live = store.add_task(TaskRecord(
            ...     run_id=r.id, name="relax", status="running", cache_key="ab" * 32,
            ...     inputs={"x": "cd" * 32}, started_at=utcnow(),
            ... ))
            >>> done = store.update_task(live.seq, status="completed",
            ...                          outputs={"return": "ef" * 32}, finished_at=utcnow())
            >>> done.status.value
            'completed'
            >>> store.close()
        """
        final = ExecutionStatus(status)
        if final not in (ExecutionStatus.COMPLETED, ExecutionStatus.FAILED):
            raise ValueError(
                f"update_task finalizes a task: status must be 'completed' or 'failed',"
                f" got {final.value!r}"
            )
        if (error is not None or failure is not None) and final is not ExecutionStatus.FAILED:
            raise ValueError("error=/failure= are only recordable when finalizing as 'failed'")
        with self._txn() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE seq = ?", (seq,)).fetchone()
            if row is None:
                raise ValueError(f"no task with seq {seq}")
            if ExecutionStatus(row["status"]) is not ExecutionStatus.RUNNING:
                raise ValueError(
                    f"task {seq} is already finalized ({row['status']}); tasks finalize once"
                )
            conn.execute(
                "UPDATE tasks SET status = ?, outputs = ?, error = ?, failure = ?,"
                " finished_at = ? WHERE seq = ?",
                (
                    final.value,
                    json.dumps(outputs or {}, sort_keys=True),
                    error,
                    _fmt_json(failure),
                    _fmt_dt(finished_at),
                    seq,
                ),
            )
            task_row = conn.execute("SELECT * FROM tasks WHERE seq = ?", (seq,)).fetchone()
        return _row_to_task(task_row)

    def add_check_results(self, run_id: str, results: Sequence[CheckResult]) -> list[CheckResult]:
        """Record verification results on a run; return them stamped with its full id.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> r = store.create(Run())
            >>> _ = store.add_check_results(r.id, [CheckResult(
            ...     run_id=r.id, name="fmax", kind="converged", passed=True,
            ...     message="fmax=0.03 < 0.05",
            ... )])
            >>> store.list_check_results(r.id)[0].passed
            True
            >>> store.close()
        """
        with self._txn() as conn:
            rid = self._resolve(run_id)
            stamped = [result.model_copy(update={"run_id": rid}) for result in results]
            for result in stamped:
                conn.execute(
                    "INSERT INTO checks (run_id, name, kind, passed, message, observed,"
                    " expected, at) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        rid,
                        result.name,
                        result.kind,
                        int(result.passed),
                        result.message,
                        json.dumps(result.observed, sort_keys=True, default=repr),
                        json.dumps(result.expected, sort_keys=True, default=repr),
                        result.at.isoformat(),
                    ),
                )
        return stamped

    # -- reads ------------------------------------------------------------------------

    def list_tasks(self, run_id: str) -> list[TaskRecord]:
        """List a run's traced task calls, oldest first.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> r = store.create(Run())
            >>> store.list_tasks(r.id)
            []
            >>> store.close()
        """
        with self._lock:
            rid = self._resolve(run_id)
            rows = self._conn.execute(
                "SELECT * FROM tasks WHERE run_id = ? ORDER BY seq", (rid,)
            ).fetchall()
        return [_row_to_task(row) for row in rows]

    def find_cached_task(self, cache_key: str) -> TaskRecord | None:
        """Return the most recent *completed* task with this cache key, if any.

        This is the cache lookup the tracer uses: failed attempts never
        populate the cache, and lookups span all runs in the store.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> store.find_cached_task("ab" * 32) is None
            True
            >>> store.close()
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE cache_key = ? AND status = ? ORDER BY seq DESC LIMIT 1",
                (cache_key, ExecutionStatus.COMPLETED.value),
            ).fetchone()
        return None if row is None else _row_to_task(row)

    def list_check_results(self, run_id: str) -> list[CheckResult]:
        """List a run's verification results, oldest first.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> r = store.create(Run())
            >>> store.list_check_results(r.id)
            []
            >>> store.close()
        """
        with self._lock:
            rid = self._resolve(run_id)
            rows = self._conn.execute(
                "SELECT * FROM checks WHERE run_id = ? ORDER BY seq", (rid,)
            ).fetchall()
        return [
            CheckResult(
                run_id=row["run_id"],
                name=row["name"],
                kind=row["kind"],
                passed=bool(row["passed"]),
                message=row["message"],
                observed=json.loads(row["observed"]),
                expected=json.loads(row["expected"]),
                at=datetime.fromisoformat(row["at"]),
            )
            for row in rows
        ]

    def list_artifacts(
        self, run_id: str, *, role: ArtifactRole | str | None = None
    ) -> list[ArtifactRef]:
        """List a run's artifact references, oldest first, optionally by role.

        References persist even after their bytes are discarded; check byte
        availability with ``artifact_store.has(ref.hash)``.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> r = store.create(Run())
            >>> _ = store.add_artifact(r.id, name="out", role="terminal",
            ...                        hash="cd" * 32, size_bytes=10)
            >>> [a.name for a in store.list_artifacts(r.id, role="terminal")]
            ['out']
            >>> store.list_artifacts(r.id, role="intermediate")
            []
            >>> store.close()
        """
        sql = "SELECT * FROM artifacts WHERE run_id = ?"
        with self._lock:
            rid = self._resolve(run_id)
            params: list[object] = [rid]
            if role is not None:
                sql += " AND role = ?"
                params.append(ArtifactRole(role).value)
            rows = self._conn.execute(sql + " ORDER BY seq", params).fetchall()
        return [_row_to_artifact(row) for row in rows]

    def get_artifact(self, run_id: str, name: str) -> ArtifactRef:
        """Fetch one of the run's artifact references by name.

        Raises:
            ArtifactNotFoundError: The run has no artifact with this name.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> r = store.create(Run())
            >>> _ = store.add_artifact(r.id, name="bands.json", role="terminal",
            ...                        hash="ef" * 32, size_bytes=42)
            >>> store.get_artifact(r.id, "bands.json").size_bytes
            42
            >>> store.close()
        """
        with self._lock:
            rid = self._resolve(run_id)
            row = self._conn.execute(
                "SELECT * FROM artifacts WHERE run_id = ? AND name = ?", (rid, name)
            ).fetchone()
        if row is None:
            raise ArtifactNotFoundError.for_name(rid, name)
        return _row_to_artifact(row)

    def get(self, run_id: str) -> Run:
        """Fetch a run by full id or unique prefix (git-style).

        Raises:
            RunNotFoundError: Nothing matches.
            AmbiguousRunIdError: The prefix matches more than one run.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> r = store.create(Run(name="demo"))
            >>> store.get(r.id[:10]).id == r.id
            True
            >>> store.close()
        """
        with self._lock:
            rid = self._resolve(run_id)
            row = self._conn.execute("SELECT * FROM runs WHERE id = ?", (rid,)).fetchone()
        return _row_to_run(row)

    def resolve(self, run_id: str) -> str:
        """Resolve a full id or unique prefix to the full run id.

        An exact match always wins, even if it is also a prefix of other ids.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> r = store.create(Run())
            >>> store.resolve(r.id[:8]) == r.id
            True
            >>> store.close()
        """
        with self._lock:
            return self._resolve(run_id)

    def list_runs(
        self,
        *,
        state: LifecycleState | str | None = None,
        status: ExecutionStatus | str | None = None,
        session: str | None = None,
        limit: int | None = None,
    ) -> list[Run]:
        """List runs, newest first, optionally filtered by state, status, session.

        The *session* filter takes a full session id or a unique prefix, and is
        resolved the same way run ids are.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> _ = store.create(Run(name="a"))
            >>> _ = store.create(Run(name="b", session="chat-1"))
            >>> [r.name for r in store.list_runs(state="quarantined", limit=1)]
            ['b']
            >>> [r.name for r in store.list_runs(session="chat")]
            ['b']
            >>> store.list_runs(state=LifecycleState.PROMOTED)
            []
            >>> store.close()
        """
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")
        clauses: list[str] = []
        params: list[object] = []
        if state is not None:
            clauses.append("state = ?")
            params.append(LifecycleState(state).value)
        if status is not None:
            clauses.append("status = ?")
            params.append(ExecutionStatus(status).value)
        if session is not None:
            clauses.append("session = ?")
            params.append(self.resolve_session(session))
        sql = "SELECT * FROM runs"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC, id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_run(row) for row in rows]

    def resolve_session(self, session: str) -> str:
        """Resolve a full session id or unique prefix to the full session id.

        An exact match always wins, even if it is also a prefix of other
        sessions — the same rule run ids follow.

        Raises:
            SessionNotFoundError: No run carries a matching session.
            AmbiguousSessionError: The prefix matches several sessions.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> _ = store.create(Run(session="20260828-013504-48123"))
            >>> store.resolve_session("20260828")
            '20260828-013504-48123'
            >>> store.close()
        """
        if not session:
            raise ValueError("session id (or prefix) must be non-empty")
        with self._lock:
            exact = self._conn.execute(
                "SELECT 1 FROM runs WHERE session = ? LIMIT 1", (session,)
            ).fetchone()
            if exact is not None:
                return session
            escaped = session.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            rows = self._conn.execute(
                "SELECT DISTINCT session FROM runs WHERE session LIKE ? ESCAPE '\\'"
                " ORDER BY session LIMIT 6",
                (escaped + "%",),
            ).fetchall()
        if not rows:
            raise SessionNotFoundError(session)
        if len(rows) > 1:
            raise AmbiguousSessionError(session, [str(r["session"]) for r in rows])
        return str(rows[0]["session"])

    def list_sessions(self, *, limit: int | None = None) -> list[SessionSummary]:
        """Summarize the sessions that created runs, newest run first.

        Runs with no session are not a session and never appear here; count
        them with ``list_runs`` if you need the number.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> _ = store.create(Run(session="chat-1"))
            >>> _ = store.create(Run(session="chat-1"))
            >>> _ = store.create(Run())
            >>> [(s.session, s.runs, s.breakdown()) for s in store.list_sessions()]
            [('chat-1', 2, '2 quarantined')]
            >>> store.close()
        """
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")
        sql = (
            "SELECT session, state, COUNT(*) AS n, MAX(created_at) AS newest"
            " FROM runs WHERE session IS NOT NULL GROUP BY session, state"
        )
        with self._lock:
            rows = self._conn.execute(sql).fetchall()
        states: dict[str, dict[str, int]] = {}
        newest: dict[str, str] = {}
        for row in rows:
            session = str(row["session"])
            states.setdefault(session, {})[str(row["state"])] = int(row["n"])
            if str(row["newest"]) > newest.get(session, ""):
                newest[session] = str(row["newest"])
        summaries = [
            SessionSummary(
                session=session,
                runs=sum(counts.values()),
                states=counts,
                newest_at=datetime.fromisoformat(newest[session]),
            )
            for session, counts in states.items()
        ]
        summaries.sort(key=lambda s: (s.newest_at, s.session), reverse=True)
        return summaries if limit is None else summaries[:limit]

    def history(self, run_id: str) -> list[Transition]:
        """Return the run's lifecycle transitions, oldest first.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> r = store.create(Run())
            >>> _ = store.transition(r.id, "verified", actor="checks")
            >>> [(t.from_state.value, t.to_state.value, t.actor) for t in store.history(r.id)]
            [('quarantined', 'verified', 'checks')]
            >>> store.close()
        """
        with self._lock:
            rid = self._resolve(run_id)
            rows = self._conn.execute(
                "SELECT * FROM transitions WHERE run_id = ? ORDER BY seq", (rid,)
            ).fetchall()
        return [
            Transition(
                run_id=row["run_id"],
                from_state=LifecycleState(row["from_state"]),
                to_state=LifecycleState(row["to_state"]),
                actor=row["actor"],
                reason=row["reason"],
                forced=bool(row["forced"]),
                at=datetime.fromisoformat(row["at"]),
            )
            for row in rows
        ]

    def delete_run(self, run_id: str) -> Run:
        """Delete one expired run and every row that references it.

        Only ``expired`` runs can be deleted: expiry is the lifecycle's
        release of the evidence, and deletion is the purge phase acting on
        that release. Any other state is refused — expire it first, or
        promote it to keep it. Child rows (transitions, artifact references,
        tasks, checks) go with the run via the schema's ``ON DELETE
        CASCADE``. Returns the run as it was, for reporting.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> r = store.create(Run(name="scratch"))
            >>> try:
            ...     store.delete_run(r.id)
            ... except RunStateError as e:
            ...     print(e.operation, e.state.value)
            delete quarantined
            >>> _ = store.transition(r.id, "expired", actor="ttl")
            >>> store.delete_run(r.id).name
            'scratch'
            >>> store.list_runs()
            []
            >>> store.close()
        """
        with self._txn() as conn:
            rid = self._resolve(run_id)
            run = self._get_exact(conn, rid)
            if run.state is not LifecycleState.EXPIRED:
                raise RunStateError(rid, run.state, "delete")
            conn.execute("DELETE FROM runs WHERE id = ?", (rid,))
            return run

    # -- internals --------------------------------------------------------------------

    @contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
        """Serialize a write: take the in-process lock and a SQLite write lock."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def _init_schema(self) -> None:
        with self._txn() as conn:
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise SchemaVersionError(self._path, found=version, supported=SCHEMA_VERSION)
            if version == SCHEMA_VERSION:
                return
            if version == 0:  # fresh database: current schema directly
                for statement in _SCHEMA.strip().split(";\n"):
                    if statement.strip():
                        conn.execute(statement)
            else:  # existing database: apply each migration in order
                for target in range(version + 1, SCHEMA_VERSION + 1):
                    for statement in _MIGRATIONS[target]:
                        conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    def _resolve(self, run_id: str) -> str:
        """Resolve id-or-prefix to a full id. Caller must hold the lock."""
        if not run_id:
            raise ValueError("run id (or prefix) must be non-empty")
        exact = self._conn.execute("SELECT id FROM runs WHERE id = ?", (run_id,)).fetchone()
        if exact is not None:
            return str(exact["id"])
        escaped = run_id.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        rows = self._conn.execute(
            "SELECT id FROM runs WHERE id LIKE ? ESCAPE '\\' ORDER BY id LIMIT 6",
            (escaped + "%",),
        ).fetchall()
        if not rows:
            raise RunNotFoundError(run_id)
        if len(rows) > 1:
            raise AmbiguousRunIdError(run_id, [str(r["id"]) for r in rows])
        return str(rows[0]["id"])

    def _get_exact(self, conn: sqlite3.Connection, rid: str) -> Run:
        """Read a run by exact id on *conn* (inside a transaction)."""
        row = conn.execute("SELECT * FROM runs WHERE id = ?", (rid,)).fetchone()
        return _row_to_run(row)


def _fmt_dt(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _parse_dt(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)


def _fmt_json(value: dict[str, object] | None) -> str | None:
    return None if value is None else json.dumps(value, sort_keys=True, default=repr)


def _parse_json(value: str | None) -> dict[str, object] | None:
    return None if value is None else json.loads(value)


def _row_to_run(row: sqlite3.Row) -> Run:
    return Run(
        id=row["id"],
        name=row["name"],
        state=LifecycleState(row["state"]),
        status=ExecutionStatus(row["status"]),
        intent=row["intent"],
        session=row["session"],
        meta=json.loads(row["meta"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        state_entered_at=datetime.fromisoformat(row["state_entered_at"]),
        started_at=_parse_dt(row["started_at"]),
        finished_at=_parse_dt(row["finished_at"]),
        error=row["error"],
        failure=_parse_json(row["failure"]),
    )


def _row_to_task(row: sqlite3.Row) -> TaskRecord:
    return TaskRecord(
        run_id=row["run_id"],
        seq=row["seq"],
        name=row["name"],
        status=ExecutionStatus(row["status"]),
        cache_hit=bool(row["cache_hit"]),
        cache_key=row["cache_key"],
        recipe=json.loads(row["recipe"]),
        inputs=json.loads(row["inputs"]),
        outputs=json.loads(row["outputs"]),
        error=row["error"],
        failure=_parse_json(row["failure"]),
        started_at=datetime.fromisoformat(row["started_at"]),
        finished_at=_parse_dt(row["finished_at"]),
    )


def _row_to_artifact(row: sqlite3.Row) -> ArtifactRef:
    recipe = row["recipe"]
    return ArtifactRef(
        run_id=row["run_id"],
        name=row["name"],
        role=ArtifactRole(row["role"]),
        hash=row["hash"],
        size_bytes=row["size_bytes"],
        recipe=None if recipe is None else json.loads(recipe),
        created_at=datetime.fromisoformat(row["created_at"]),
    )


if TYPE_CHECKING:
    # Static assertion: SQLiteRunStore structurally satisfies RunStore.
    _conformance: RunStore = cast(SQLiteRunStore, None)
