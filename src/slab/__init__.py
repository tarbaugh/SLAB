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

from slab.artifacts import ArtifactStore
from slab.errors import (
    AmbiguousHashError,
    AmbiguousRunIdError,
    ArtifactExistsError,
    ArtifactNotFoundError,
    IllegalStatusChangeError,
    IllegalTransitionError,
    RunExistsError,
    RunNotFoundError,
    RunStateError,
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
from slab.models import ArtifactRef, ArtifactRole, Run, Transition, utcnow
from slab.retention import DEFAULT_POLICY, GcReport, RetentionPolicy, StateRule, expire_due, gc
from slab.store import RunStore, SQLiteRunStore

__version__ = "0.1.0"

__all__ = [
    "ALLOWED_STATUS_CHANGES",
    "ALLOWED_TRANSITIONS",
    "DEFAULT_POLICY",
    "FORCE_TRANSITIONS",
    "TERMINAL_STATES",
    "AmbiguousHashError",
    "AmbiguousRunIdError",
    "ArtifactExistsError",
    "ArtifactNotFoundError",
    "ArtifactRef",
    "ArtifactRole",
    "ArtifactStore",
    "ExecutionStatus",
    "GcReport",
    "IllegalStatusChangeError",
    "IllegalTransitionError",
    "LifecycleState",
    "RetentionPolicy",
    "Run",
    "RunExistsError",
    "RunNotFoundError",
    "RunStateError",
    "RunStore",
    "SQLiteRunStore",
    "SchemaVersionError",
    "SlabError",
    "StateRule",
    "StorageError",
    "Transition",
    "__version__",
    "can_transition",
    "expire_due",
    "gc",
    "is_terminal",
    "requires_force",
    "utcnow",
    "validate_status_change",
    "validate_transition",
]
