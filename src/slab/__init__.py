"""SLAB — Simplest Layer for Atomistic Backends.

Agent-native workflow orchestration for atomistic materials modeling. Runs are
born ephemeral (``quarantined``) and move toward permanence only by explicit
action; anything never promoted eventually expires.

Examples:
    >>> from slab import LifecycleState, Run, SQLiteRunStore
    >>> store = SQLiteRunStore(":memory:")
    >>> r = store.create(Run(name="si-relax", intent="baseline lattice constant"))
    >>> store.transition(r.id, LifecycleState.VERIFIED, actor="checks").state.value
    'verified'
    >>> store.transition(r.id, LifecycleState.PROMOTED, reason="best of batch").state.value
    'promoted'
    >>> store.close()
"""

from slab.errors import (
    AmbiguousRunIdError,
    IllegalStatusChangeError,
    IllegalTransitionError,
    RunExistsError,
    RunNotFoundError,
    SchemaVersionError,
    SlabError,
    StorageError,
)
from slab.lifecycle import (
    ALLOWED_STATUS_CHANGES,
    ALLOWED_TRANSITIONS,
    FORCE_TRANSITIONS,
    TERMINAL_STATES,
    ExecutionStatus,
    LifecycleState,
    can_transition,
    is_terminal,
    requires_force,
    validate_status_change,
    validate_transition,
)
from slab.models import Run, Transition, utcnow
from slab.store import RunStore, SQLiteRunStore

__version__ = "0.1.0"

__all__ = [
    "ALLOWED_STATUS_CHANGES",
    "ALLOWED_TRANSITIONS",
    "FORCE_TRANSITIONS",
    "TERMINAL_STATES",
    "AmbiguousRunIdError",
    "ExecutionStatus",
    "IllegalStatusChangeError",
    "IllegalTransitionError",
    "LifecycleState",
    "Run",
    "RunExistsError",
    "RunNotFoundError",
    "RunStore",
    "SQLiteRunStore",
    "SchemaVersionError",
    "SlabError",
    "StorageError",
    "Transition",
    "__version__",
    "can_transition",
    "is_terminal",
    "requires_force",
    "utcnow",
    "validate_status_change",
    "validate_transition",
]
