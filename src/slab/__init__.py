"""SLAB — Simplest Layer for Atomistic Backends.

Agent-native workflow orchestration for atomistic materials modeling. Runs are
born ephemeral (``quarantined``) and move toward permanence only by explicit
action; anything never promoted eventually expires. Workflows are plain
imperative Python — the graph is traced, never declared.

Examples:
    >>> import tempfile
    >>> from slab import Workspace, task, check, converged
    >>> @task
    ... def relax(x):
    ...     return x * 0.5, {"fmax": 0.03}
    >>> ws = Workspace(tempfile.mkdtemp())
    >>> with ws.start_run(name="si-relax", intent="baseline") as run:
    ...     structure, info = relax(1.0)
    ...     @check
    ...     def forces_converged():
    ...         return converged(info["fmax"], below=0.05)
    >>> ws.runs.get(run.id).state.value  # checks passed -> verified
    'verified'
    >>> ws.runs.transition(run.id, "promoted", reason="baseline worth keeping").state.value
    'promoted'
    >>> ws.close()
"""

from slab._version import __version__
from slab.artifacts import ArtifactStore
from slab.checks import Assertion, converged, finite, units, within_bounds
from slab.errors import (
    AmbiguousHashError,
    AmbiguousRunIdError,
    ArtifactExistsError,
    ArtifactNotFoundError,
    EngineNotAvailableError,
    IllegalStatusChangeError,
    IllegalTransitionError,
    NestedRunError,
    NoActiveRunError,
    RunExistsError,
    RunNotFoundError,
    RunStateError,
    SchemaVersionError,
    ScriptExitError,
    SerializationError,
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
from slab.models import (
    ArtifactRef,
    ArtifactRole,
    CheckResult,
    Run,
    TaskRecord,
    Transition,
    utcnow,
)
from slab.retention import DEFAULT_POLICY, GcReport, RetentionPolicy, StateRule, expire_due, gc
from slab.runtime import ActiveRun, Workspace, check, current_run
from slab.serialize import dumps, fingerprint, loads
from slab.store import RunStore, SQLiteRunStore
from slab.tracing import task

__all__ = [
    "ALLOWED_STATUS_CHANGES",
    "ALLOWED_TRANSITIONS",
    "DEFAULT_POLICY",
    "FORCE_TRANSITIONS",
    "TERMINAL_STATES",
    "ActiveRun",
    "AmbiguousHashError",
    "AmbiguousRunIdError",
    "ArtifactExistsError",
    "ArtifactNotFoundError",
    "ArtifactRef",
    "ArtifactRole",
    "ArtifactStore",
    "Assertion",
    "CheckResult",
    "EngineNotAvailableError",
    "ExecutionStatus",
    "GcReport",
    "IllegalStatusChangeError",
    "IllegalTransitionError",
    "LifecycleState",
    "NestedRunError",
    "NoActiveRunError",
    "RetentionPolicy",
    "Run",
    "RunExistsError",
    "RunNotFoundError",
    "RunStateError",
    "RunStore",
    "SQLiteRunStore",
    "SchemaVersionError",
    "ScriptExitError",
    "SerializationError",
    "SlabError",
    "StateRule",
    "StorageError",
    "TaskRecord",
    "Transition",
    "Workspace",
    "__version__",
    "can_transition",
    "check",
    "converged",
    "current_run",
    "dumps",
    "expire_due",
    "fingerprint",
    "finite",
    "gc",
    "is_terminal",
    "loads",
    "requires_force",
    "task",
    "units",
    "utcnow",
    "validate_status_change",
    "validate_transition",
    "within_bounds",
]
