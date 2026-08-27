"""Foundation — workflows and state for SLAB.

Runs are born ephemeral (``quarantined``) and move toward permanence only by
explicit action; anything never promoted eventually expires. Workflows are
plain imperative Python — the graph is traced, never declared.

Foundation sits above :mod:`slab`, which gives access to the computational
software, and below :mod:`mason`, the resident agent. It owns everything
between the two: runs, artifacts, caching, verification, and the surfaces
that expose them.

Examples:
    >>> import tempfile
    >>> from foundation import Workspace, task, check, converged
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

from foundation.artifacts import ArtifactStore
from foundation.checks import Assertion, converged, finite, units, within_bounds
from foundation.errors import (
    AmbiguousHashError,
    AmbiguousRunIdError,
    ArtifactExistsError,
    ArtifactNotFoundError,
    FoundationError,
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
    StorageError,
)
from foundation.lifecycle import (
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
from foundation.models import (
    ArtifactRef,
    ArtifactRole,
    CheckResult,
    Run,
    TaskRecord,
    Transition,
    utcnow,
)
from foundation.retention import (
    DEFAULT_POLICY,
    GcReport,
    PurgeReport,
    RetentionPolicy,
    StateRule,
    expire_due,
    gc,
    purge_expired,
)
from foundation.runtime import ActiveRun, Workspace, check, current_run
from foundation.serialize import dumps, fingerprint, loads
from foundation.store import RunStore, SQLiteRunStore
from foundation.tracing import task
from slab._version import __version__

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
    "ExecutionStatus",
    "FoundationError",
    "GcReport",
    "IllegalStatusChangeError",
    "IllegalTransitionError",
    "LifecycleState",
    "NestedRunError",
    "NoActiveRunError",
    "PurgeReport",
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
    "purge_expired",
    "requires_force",
    "task",
    "units",
    "utcnow",
    "validate_status_change",
    "validate_transition",
    "within_bounds",
]
