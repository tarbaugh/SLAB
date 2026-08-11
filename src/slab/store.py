"""Run persistence: the :class:`RunStore` protocol and the SQLite implementation.

The protocol is the seam for swapping backends later (e.g. Postgres). The SQLite
implementation is single-file and zero-config; ``":memory:"`` gives an ephemeral
store for tests.

Concurrency model: every mutating operation runs inside ``BEGIN IMMEDIATE`` and
revalidates against the state read *inside* the transaction, so concurrent
transitions from threads, processes, or stale handles serialize safely — exactly
one writer wins and the loser gets a precise :class:`~slab.errors.IllegalTransitionError`.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, cast, runtime_checkable

from slab.errors import (
    AmbiguousRunIdError,
    RunExistsError,
    RunNotFoundError,
    SchemaVersionError,
    StorageError,
)
from slab.lifecycle import (
    ExecutionStatus,
    LifecycleState,
    requires_force,
    validate_status_change,
    validate_transition,
)
from slab.models import Run, Transition, utcnow

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL DEFAULT '',
    state       TEXT NOT NULL,
    status      TEXT NOT NULL,
    intent      TEXT,
    meta        TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_runs_state ON runs(state);
CREATE INDEX IF NOT EXISTS ix_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS ix_runs_created_at ON runs(created_at);
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
"""


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

    def list_runs(
        self,
        *,
        state: LifecycleState | str | None = None,
        status: ExecutionStatus | str | None = None,
        limit: int | None = None,
    ) -> list[Run]:
        """List runs, newest first, optionally filtered."""
        ...

    def transition(
        self,
        run_id: str,
        to_state: LifecycleState | str,
        *,
        actor: str = "user",
        reason: str | None = None,
        force: bool = False,
    ) -> Run:
        """Atomically move a run to a new lifecycle state."""
        ...

    def set_status(self, run_id: str, status: ExecutionStatus | str) -> Run:
        """Change a run's execution status."""
        ...

    def set_intent(self, run_id: str, intent: str | None) -> Run:
        """Set or clear a run's intent note."""
        ...

    def history(self, run_id: str) -> list[Transition]:
        """Return the run's lifecycle transitions, oldest first."""
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
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA journal_mode = WAL")
        try:
            self._init_schema()
        except Exception:
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
                    "INSERT INTO runs (id, name, state, status, intent, meta, created_at,"
                    " updated_at, started_at, finished_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        run.id,
                        run.name,
                        run.state.value,
                        run.status.value,
                        run.intent,
                        meta_json,
                        run.created_at.isoformat(),
                        run.updated_at.isoformat(),
                        _fmt_dt(run.started_at),
                        _fmt_dt(run.finished_at),
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
    ) -> Run:
        """Atomically move a run to *to_state*; return the updated snapshot.

        The run's current state is re-read inside the transaction and the
        transition validated against it, so racing callers serialize: one wins,
        the rest get :class:`~slab.errors.IllegalTransitionError` describing the
        *actual* current state.

        Every transition is recorded with ``actor``, ``reason``, and a ``forced``
        flag. ``forced`` is true only when the transition needed ``force=True``
        (a force-promotion) — passing ``force=True`` on a normally-legal
        transition records ``forced=False``.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> r = store.create(Run(name="demo"))
            >>> store.transition(r.id, "verified", actor="checks", reason="fmax<0.05").state.value
            'verified'
            >>> store.transition(r.id, LifecycleState.PROMOTED).state.value
            'promoted'
            >>> [t.to_state.value for t in store.history(r.id)]
            ['verified', 'promoted']
            >>> store.close()
        """
        to = LifecycleState(to_state)
        with self._txn() as conn:
            rid = self._resolve(run_id)
            row = conn.execute("SELECT state FROM runs WHERE id = ?", (rid,)).fetchone()
            current = LifecycleState(row["state"])
            validate_transition(current, to, force=force)
            now = utcnow().isoformat()
            cur = conn.execute(
                "UPDATE runs SET state = ?, updated_at = ? WHERE id = ? AND state = ?",
                (to.value, now, rid, current.value),
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

    def set_status(self, run_id: str, status: ExecutionStatus | str) -> Run:
        """Change a run's execution status; return the updated snapshot.

        Stamps ``started_at`` when entering ``running`` and ``finished_at`` when
        entering ``completed`` or ``failed``. A run served from cache may go
        ``pending -> completed`` directly, finishing without ever starting.

        Raises:
            IllegalStatusChangeError: The change is not permitted
                (e.g. ``completed -> running``).

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> r = store.create(Run())
            >>> store.set_status(r.id, "running").status.value
            'running'
            >>> done = store.set_status(r.id, "completed")
            >>> done.finished_at is not None
            True
            >>> store.close()
        """
        new = ExecutionStatus(status)
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

    # -- reads ------------------------------------------------------------------------

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
        limit: int | None = None,
    ) -> list[Run]:
        """List runs, newest first, optionally filtered by state and/or status.

        Examples:
            >>> store = SQLiteRunStore(":memory:")
            >>> _ = store.create(Run(name="a"))
            >>> _ = store.create(Run(name="b"))
            >>> [r.name for r in store.list_runs(state="quarantined", limit=1)]
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
            if version < SCHEMA_VERSION:
                for statement in _SCHEMA.strip().split(";\n"):
                    if statement.strip():
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


def _row_to_run(row: sqlite3.Row) -> Run:
    return Run(
        id=row["id"],
        name=row["name"],
        state=LifecycleState(row["state"]),
        status=ExecutionStatus(row["status"]),
        intent=row["intent"],
        meta=json.loads(row["meta"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        started_at=_parse_dt(row["started_at"]),
        finished_at=_parse_dt(row["finished_at"]),
    )


if TYPE_CHECKING:
    # Static assertion: SQLiteRunStore structurally satisfies RunStore.
    _conformance: RunStore = cast(SQLiteRunStore, None)
