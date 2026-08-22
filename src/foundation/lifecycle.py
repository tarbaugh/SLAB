"""Run lifecycle: states, the allowed-transition relation, and validation.

The lifecycle encodes SLAB's core asymmetry — runs are born ephemeral and move
toward permanence only by explicit action; nothing is born permanent::

    quarantined ──checks pass──▶ verified ──promote──▶ promoted ──archive──▶ archived
        │  │                        │
        │  └────────force-promote───┼──────────────▶ (promoted)
        │ ttl                       │ ttl
        ▼                           ▼
     expired ◀──────────────────────┘

Two rules are structural, not policy:

* ``promoted`` and ``archived`` runs never expire. There is no transition to
  ``expired`` from either state, with or without ``force``.
* ``expired`` and ``archived`` are terminal. Nothing leaves them.

Orthogonal to the lifecycle, each run has an *execution status*
(``pending -> running -> completed | failed``). A run that fails execution simply
stays quarantined and ages out — failure needs no special lifecycle handling.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Final

from foundation.errors import IllegalStatusChangeError, IllegalTransitionError


class LifecycleState(StrEnum):
    """Retention state of a run.

    - ``QUARANTINED``: freshly created; ephemeral; expires by TTL unless promoted.
    - ``VERIFIED``: verification checks passed; eligible for promotion, but still
      ephemeral — only promotion confers permanence.
    - ``PROMOTED``: explicitly kept; retention guarantees apply; never expires.
    - ``ARCHIVED``: promoted data moved to cold storage; terminal; never expires.
    - ``EXPIRED``: terminal; the ephemeral window lapsed without promotion.
    """

    QUARANTINED = "quarantined"
    VERIFIED = "verified"
    PROMOTED = "promoted"
    ARCHIVED = "archived"
    EXPIRED = "expired"


class ExecutionStatus(StrEnum):
    """Execution status of a run, orthogonal to its lifecycle state.

    - ``PENDING``: created but not yet executing.
    - ``RUNNING``: currently executing.
    - ``COMPLETED``: finished successfully (verification may still fail it later).
    - ``FAILED``: raised or was killed; stays quarantined and ages out.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


ALLOWED_TRANSITIONS: Final[Mapping[LifecycleState, frozenset[LifecycleState]]] = {
    LifecycleState.QUARANTINED: frozenset({LifecycleState.VERIFIED, LifecycleState.EXPIRED}),
    LifecycleState.VERIFIED: frozenset({LifecycleState.PROMOTED, LifecycleState.EXPIRED}),
    LifecycleState.PROMOTED: frozenset({LifecycleState.ARCHIVED}),
    LifecycleState.ARCHIVED: frozenset(),
    LifecycleState.EXPIRED: frozenset(),
}
"""Transitions permitted without ``force``."""

FORCE_TRANSITIONS: Final[frozenset[tuple[LifecycleState, LifecycleState]]] = frozenset(
    {(LifecycleState.QUARANTINED, LifecycleState.PROMOTED)}
)
"""Transitions additionally permitted with ``force=True`` (recorded as forced)."""

TERMINAL_STATES: Final[frozenset[LifecycleState]] = frozenset(
    {LifecycleState.ARCHIVED, LifecycleState.EXPIRED}
)
"""States with no outgoing transitions."""

ALLOWED_STATUS_CHANGES: Final[Mapping[ExecutionStatus, frozenset[ExecutionStatus]]] = {
    # pending -> completed covers cache hits (a result served without executing);
    # pending -> failed covers setup failures before execution started.
    ExecutionStatus.PENDING: frozenset(
        {ExecutionStatus.RUNNING, ExecutionStatus.COMPLETED, ExecutionStatus.FAILED}
    ),
    ExecutionStatus.RUNNING: frozenset({ExecutionStatus.COMPLETED, ExecutionStatus.FAILED}),
    ExecutionStatus.COMPLETED: frozenset(),
    ExecutionStatus.FAILED: frozenset(),
}
"""Permitted execution-status changes. ``completed`` and ``failed`` are final."""


def is_terminal(state: LifecycleState) -> bool:
    """Return True if no transition leaves *state*.

    Examples:
        >>> is_terminal(LifecycleState.EXPIRED)
        True
        >>> is_terminal(LifecycleState.PROMOTED)
        False
    """
    return state in TERMINAL_STATES


def requires_force(from_state: LifecycleState, to_state: LifecycleState) -> bool:
    """Return True if this transition is legal only with ``force=True``.

    Examples:
        >>> requires_force(LifecycleState.QUARANTINED, LifecycleState.PROMOTED)
        True
        >>> requires_force(LifecycleState.VERIFIED, LifecycleState.PROMOTED)
        False
    """
    return (from_state, to_state) in FORCE_TRANSITIONS


def can_transition(
    from_state: LifecycleState, to_state: LifecycleState, *, force: bool = False
) -> bool:
    """Return True if the lifecycle permits *from_state* -> *to_state*.

    ``force=True`` additionally unlocks the transitions in
    :data:`FORCE_TRANSITIONS` (currently only quarantined -> promoted). It never
    unlocks anything else — in particular, nothing makes promoted data expire.

    Examples:
        >>> can_transition(LifecycleState.QUARANTINED, LifecycleState.VERIFIED)
        True
        >>> can_transition(LifecycleState.QUARANTINED, LifecycleState.PROMOTED)
        False
        >>> can_transition(LifecycleState.QUARANTINED, LifecycleState.PROMOTED, force=True)
        True
        >>> can_transition(LifecycleState.PROMOTED, LifecycleState.EXPIRED, force=True)
        False
    """
    if to_state in ALLOWED_TRANSITIONS[from_state]:
        return True
    return force and requires_force(from_state, to_state)


def validate_transition(
    from_state: LifecycleState, to_state: LifecycleState, *, force: bool = False
) -> None:
    """Raise :class:`~foundation.errors.IllegalTransitionError` unless the transition is permitted.

    The raised error's message explains *why* the transition is illegal and, when
    ``force=True`` would permit it, says so (``force_would_allow`` is set).

    Examples:
        >>> validate_transition(LifecycleState.QUARANTINED, LifecycleState.VERIFIED)
        >>> try:
        ...     validate_transition(LifecycleState.QUARANTINED, LifecycleState.PROMOTED)
        ... except IllegalTransitionError as e:
        ...     print(e)
        illegal lifecycle transition: quarantined -> promoted (promoting an unverified run \
requires force=True)
        >>> try:
        ...     validate_transition(LifecycleState.PROMOTED, LifecycleState.EXPIRED)
        ... except IllegalTransitionError as e:
        ...     print(e)
        illegal lifecycle transition: promoted -> expired (promoted runs are permanent; \
only archiving is allowed)
    """
    if can_transition(from_state, to_state, force=force):
        return

    force_would_allow = False
    detail: str | None
    if from_state == to_state:
        detail = f"run is already {from_state.value}"
    elif requires_force(from_state, to_state):
        detail = "promoting an unverified run requires force=True"
        force_would_allow = True
    elif is_terminal(from_state):
        detail = f"{from_state.value} is a terminal state; no transitions leave it"
    elif from_state is LifecycleState.PROMOTED:
        detail = "promoted runs are permanent; only archiving is allowed"
    else:
        targets = sorted(s.value for s in ALLOWED_TRANSITIONS[from_state])
        detail = f"allowed transitions from {from_state.value}: {', '.join(targets)}"
    raise IllegalTransitionError(
        from_state, to_state, detail=detail, force_would_allow=force_would_allow
    )


def validate_status_change(from_status: ExecutionStatus, to_status: ExecutionStatus) -> None:
    """Raise :class:`~foundation.errors.IllegalStatusChangeError` unless the change is permitted.

    Examples:
        >>> validate_status_change(ExecutionStatus.PENDING, ExecutionStatus.RUNNING)
        >>> try:
        ...     validate_status_change(ExecutionStatus.COMPLETED, ExecutionStatus.RUNNING)
        ... except IllegalStatusChangeError as e:
        ...     print(e)
        illegal status change: completed -> running (completed is final)
    """
    if to_status in ALLOWED_STATUS_CHANGES[from_status]:
        return
    if from_status == to_status:
        detail = f"run is already {from_status.value}"
    elif not ALLOWED_STATUS_CHANGES[from_status]:
        detail = f"{from_status.value} is final"
    else:
        targets = sorted(s.value for s in ALLOWED_STATUS_CHANGES[from_status])
        detail = f"allowed from {from_status.value}: {', '.join(targets)}"
    raise IllegalStatusChangeError(from_status, to_status, detail=detail)
