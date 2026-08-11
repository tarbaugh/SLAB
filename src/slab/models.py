"""Core data models: :class:`Run` and :class:`Transition`.

Both are frozen (immutable) pydantic models. A ``Run`` is a *snapshot*: the
store is the source of truth, and mutating operations return fresh snapshots
rather than updating objects in place.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slab._ids import new_run_id
from slab.lifecycle import ExecutionStatus, LifecycleState


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
    (:meth:`slab.store.SQLiteRunStore.transition` /
    :meth:`slab.store.SQLiteRunStore.set_status`), which returns new snapshots.

    Fields:
        id: 26-char time-ordered ULID; unique prefixes are accepted anywhere a
            run id is expected (git-style).
        name: Short human/agent-readable workflow name (e.g. ``"si-relax"``).
        state: Lifecycle (retention) state.
        status: Execution status, orthogonal to ``state``.
        intent: Free-text narrative provenance — the stated goal of the run.
        meta: Small JSON-serializable extras; not for bulk data.
        created_at / updated_at: Aware UTC timestamps, maintained by the store.
        started_at / finished_at: Execution timestamps, stamped by the store on
            status changes (a cache-served run may finish without ever starting).

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
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def _sync_timestamps(cls, data: Any) -> Any:
        """Default ``updated_at`` to ``created_at`` so fresh runs carry identical stamps."""
        if isinstance(data, dict):
            created = data.get("created_at")
            if created is None:
                created = utcnow()
                data = {**data, "created_at": created}
            if data.get("updated_at") is None:
                data = {**data, "updated_at": created}
        return data
