"""Core data models: :class:`Run` and :class:`Transition`.

Both are frozen (immutable) pydantic models. A ``Run`` is a *snapshot*: the
store is the source of truth, and mutating operations return fresh snapshots
rather than updating objects in place.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from foundation._ids import new_run_id
from foundation.lifecycle import ExecutionStatus, LifecycleState


def utcnow() -> datetime:
    """Return the current time as a timezone-aware UTC datetime.

    Examples:
        >>> utcnow().tzinfo is UTC
        True
    """
    return datetime.now(UTC)


class Transition(BaseModel):
    """One recorded lifecycle transition of a run.

    Transitions carry narrative provenance — *who* moved the run and *why* —
    alongside the state change itself. ``forced`` records whether the transition
    was only legal because ``force=True`` was passed (i.e. a force-promotion).

    Examples:
        >>> t = Transition(
        ...     run_id="abc",
        ...     from_state=LifecycleState.VERIFIED,
        ...     to_state=LifecycleState.PROMOTED,
        ...     reason="best of batch",
        ... )
        >>> (t.actor, t.forced)
        ('user', False)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    from_state: LifecycleState
    to_state: LifecycleState
    actor: str = "user"
    reason: str | None = None
    forced: bool = False
    at: datetime = Field(default_factory=utcnow)


class Run(BaseModel):
    """A single workflow execution and its retention state. Immutable snapshot.

    Runs are born ``quarantined`` (ephemeral) with execution status ``pending``.
    The lifecycle state and execution status evolve through the store
    (:meth:`foundation.store.SQLiteRunStore.transition` /
    :meth:`foundation.store.SQLiteRunStore.set_status`), which returns new snapshots.

    Fields:
        id: 26-char time-ordered ULID; unique prefixes are accepted anywhere a
            run id is expected (git-style).
        name: Short human/agent-readable workflow name (e.g. ``"si-relax"``).
        state: Lifecycle (retention) state.
        status: Execution status, orthogonal to ``state``.
        intent: Free-text narrative provenance — the stated goal of the run.
        session: Opaque id of the client session that created the run (a Mason
            chat, an MCP client). Runs sharing one id were produced by one
            conversation, so they can be listed and promoted together.
        meta: Small JSON-serializable extras; not for bulk data.
        created_at / updated_at: Aware UTC timestamps, maintained by the store.
        state_entered_at: When the run entered its current lifecycle state —
            the anchor for retention TTLs (a run's clock restarts on transition).
        started_at / finished_at: Execution timestamps, stamped by the store on
            status changes (a cache-served run may finish without ever starting).
        error: One-line failure message, recorded when the status is set to
            ``failed`` — the compact form for listings.
        failure: Structured failure evidence (exception type, message, trimmed
            traceback, diagnostic notes — see :func:`foundation.errors.failure_record`),
            recorded alongside ``error``. The full form, fetched by ``show``.

    Examples:
        >>> run = Run(name="si-relax", intent="baseline lattice constant")
        >>> (run.state.value, run.status.value)
        ('quarantined', 'pending')
        >>> run.created_at == run.updated_at
        True
        >>> len(run.id)
        26
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(default_factory=new_run_id)
    name: str = ""
    state: LifecycleState = LifecycleState.QUARANTINED
    status: ExecutionStatus = ExecutionStatus.PENDING
    intent: str | None = None
    session: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    state_entered_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    failure: dict[str, Any] | None = None

    @field_validator("created_at", "updated_at", "state_entered_at", "started_at", "finished_at")
    @classmethod
    def _aware_utc(cls, value: datetime | None) -> datetime | None:
        """Timestamps are aware UTC: a naive one would persist, sort as a
        string beside aware ones, and crash the retention sweep's comparison."""
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("run timestamps must be timezone-aware (use foundation.models.utcnow)")
        return value.astimezone(UTC)

    @model_validator(mode="before")
    @classmethod
    def _sync_timestamps(cls, data: Any) -> Any:
        """Default ``updated_at``/``state_entered_at`` to ``created_at`` at birth."""
        if isinstance(data, dict):
            created = data.get("created_at")
            if created is None:
                created = utcnow()
                data = {**data, "created_at": created}
            for key in ("updated_at", "state_entered_at"):
                if data.get(key) is None:
                    data = {**data, key: created}
        return data


class SessionSummary(BaseModel):
    """One client session's runs, counted. Immutable.

    Produced by :meth:`foundation.store.SQLiteRunStore.list_sessions`, which
    groups the runs table by :attr:`Run.session`. It answers the question a
    user asks before promoting a session: which chat produced how many runs,
    in what states, and how recently.

    Examples:
        >>> s = SessionSummary(
        ...     session="20260828-013504-48123",
        ...     runs=4,
        ...     states={"verified": 3, "quarantined": 1},
        ...     newest_at=utcnow(),
        ... )
        >>> s.breakdown()
        '3 verified, 1 quarantined'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    session: str
    runs: int = Field(ge=0)
    states: dict[str, int] = Field(default_factory=dict)
    newest_at: datetime

    def breakdown(self) -> str:
        """The state counts as one line, most common first, ties by state name.

        Examples:
            >>> SessionSummary(
            ...     session="s", runs=2, states={"promoted": 1, "expired": 1},
            ...     newest_at=utcnow(),
            ... ).breakdown()
            '1 expired, 1 promoted'
            >>> SessionSummary(session="s", runs=0, newest_at=utcnow()).breakdown()
            'no runs'
        """
        if not self.states:
            return "no runs"
        ordered = sorted(self.states.items(), key=lambda item: (-item[1], item[0]))
        return ", ".join(f"{count} {state}" for state, count in ordered)


class ArtifactRole(StrEnum):
    """Role of an artifact within its run — this is what retention tiers on.

    - ``TERMINAL``: a declared final output of the run (relaxed structure, band
      structure, report). Full bytes are retained for promoted runs.
    - ``INTERMEDIATE``: a reproducible byproduct (wavefunctions, trajectories,
      scratch files). Hash + recipe are always kept; bytes are droppable per
      policy — anything can be recomputed on demand.
    - ``INPUT``: a recompute *root* — data that entered the workflow from
      outside rather than being produced by a traced task. Kept for promoted
      runs by default: without the roots, "recompute on demand" would be a
      dangling promise.
    """

    TERMINAL = "terminal"
    INTERMEDIATE = "intermediate"
    INPUT = "input"


class ArtifactRef(BaseModel):
    """A run's reference to a content-addressed artifact. Immutable.

    The reference (this record, in the run database) and the bytes (in the
    :class:`~foundation.artifacts.ArtifactStore`) have independent lifetimes: dropping
    bytes per retention policy never deletes the reference, so the hash and the
    recipe to recompute the artifact survive. Whether bytes are currently
    available is a property of the artifact store: ``artifact_store.has(ref.hash)``.

    Examples:
        >>> ref = ArtifactRef(
        ...     run_id="abc", name="relaxed.xyz", role="terminal",
        ...     hash="c0ffee" + "0" * 58, size_bytes=1234,
        ... )
        >>> ref.role
        <ArtifactRole.TERMINAL: 'terminal'>
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    name: str = Field(min_length=1)
    role: ArtifactRole
    hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    recipe: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=utcnow)


class TaskRecord(BaseModel):
    """One traced task call inside a run. Immutable.

    Recorded after the fact by the ``@task`` tracer: either the function ran
    (``completed``/``failed``) or its result was served from cache
    (``completed`` with ``cache_hit=True``). Input and output values live in
    the artifact store under the hashes recorded here; the DAG is derived by
    hash equality — if this task's input hash equals another task's output
    hash, the data flowed between them.

    Fields:
        seq: Store-assigned position (global insertion order).
        name: Task name (function name unless overridden).
        recipe: How to recompute: module, qualname, code hash, engine
            versions, python/slab versions, and human-readable params.
        inputs: Bound argument name -> content hash.
        outputs: ``"return"`` (or ``"return[i]"`` for tuple returns) -> content hash.
        cache_key: Fingerprint of (code identity, engine versions, input hashes).
        error: One-line failure message for ``failed`` tasks.
        failure: Structured failure evidence for ``failed`` tasks
            (see :func:`foundation.errors.failure_record`).

    Examples:
        >>> rec = TaskRecord(
        ...     run_id="r", name="relax", status="completed",
        ...     cache_key="ab" * 32, recipe={"module": "wf"},
        ...     inputs={"atoms": "cd" * 32}, outputs={"return": "ef" * 32},
        ...     started_at=utcnow(),
        ... )
        >>> rec.cache_hit
        False
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    seq: int = 0
    name: str = Field(min_length=1)
    status: ExecutionStatus
    cache_hit: bool = False
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    recipe: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, str] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    failure: dict[str, Any] | None = None
    started_at: datetime
    finished_at: datetime | None = None


class CheckResult(BaseModel):
    """The outcome of one verification assertion, stored on the run.

    Produced when a run's ``@check`` hooks are evaluated at completion. All
    results passing (and at least one existing) is what gates
    ``quarantined -> verified``.

    Examples:
        >>> r = CheckResult(run_id="r", name="forces_converged", kind="converged",
        ...                 passed=True, message="fmax=0.03 < 0.05")
        >>> r.passed
        True
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    name: str = Field(min_length=1)
    kind: str = "custom"
    passed: bool
    message: str = ""
    observed: Any = None
    expected: Any = None
    at: datetime = Field(default_factory=utcnow)
