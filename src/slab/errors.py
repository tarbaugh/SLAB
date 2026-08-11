"""Exception types raised by SLAB.

All errors derive from :class:`SlabError`. Messages are written to be actionable
for the caller — including LLM agents, who read them verbatim — so they state
what was attempted, why it was refused, and what would be allowed instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from slab.lifecycle import ExecutionStatus, LifecycleState


class SlabError(Exception):
    """Base class for all SLAB errors."""


class IllegalTransitionError(SlabError):
    """A lifecycle transition the state machine does not permit.

    Attributes:
        from_state: The run's current lifecycle state.
        to_state: The requested target state.
        force_would_allow: True if retrying with ``force=True`` would permit
            this exact transition (only ever true for quarantined -> promoted).
    """

    def __init__(
        self,
        from_state: LifecycleState,
        to_state: LifecycleState,
        *,
        detail: str | None = None,
        force_would_allow: bool = False,
    ) -> None:
        self.from_state = from_state
        self.to_state = to_state
        self.force_would_allow = force_would_allow
        msg = f"illegal lifecycle transition: {from_state.value} -> {to_state.value}"
        if detail:
            msg = f"{msg} ({detail})"
        super().__init__(msg)


class IllegalStatusChangeError(SlabError):
    """An execution-status change that is not permitted.

    Attributes:
        from_status: The run's current execution status.
        to_status: The requested target status.
    """

    def __init__(
        self,
        from_status: ExecutionStatus,
        to_status: ExecutionStatus,
        *,
        detail: str | None = None,
    ) -> None:
        self.from_status = from_status
        self.to_status = to_status
        msg = f"illegal status change: {from_status.value} -> {to_status.value}"
        if detail:
            msg = f"{msg} ({detail})"
        super().__init__(msg)


class RunNotFoundError(SlabError):
    """No run matches the given id or id prefix."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"no run matches {run_id!r}")


class AmbiguousRunIdError(SlabError):
    """An id prefix matches more than one run.

    Attributes:
        prefix: The prefix that was looked up.
        matches: Matching run ids (capped at 6 by the store).
    """

    def __init__(self, prefix: str, matches: Sequence[str]) -> None:
        self.prefix = prefix
        self.matches = list(matches)
        count = "6 or more" if len(self.matches) >= 6 else str(len(self.matches))
        shown = ", ".join(self.matches[:5])
        more = ", ..." if len(self.matches) > 5 else ""
        super().__init__(f"run id prefix {prefix!r} is ambiguous ({count} matches): {shown}{more}")


class RunExistsError(SlabError):
    """A run with this id already exists in the store."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"run {run_id!r} already exists")


class RunStateError(SlabError):
    """An operation that is not allowed in the run's current lifecycle state."""

    def __init__(self, run_id: str, state: LifecycleState, operation: str) -> None:
        self.run_id = run_id
        self.state = state
        self.operation = operation
        super().__init__(
            f"cannot {operation} run {run_id!r}: its lifecycle state is {state.value!r}"
        )


class ArtifactNotFoundError(SlabError):
    """An artifact lookup failed — by content hash, or by name within a run.

    Attributes:
        digest: The content hash looked up, if the lookup was by hash.
        run_id / name: The run and artifact name, if the lookup was by name.
    """

    def __init__(
        self,
        message: str,
        *,
        digest: str | None = None,
        run_id: str | None = None,
        name: str | None = None,
    ) -> None:
        self.digest = digest
        self.run_id = run_id
        self.name = name
        super().__init__(message)

    @classmethod
    def for_hash(cls, digest: str) -> Self:
        """Bytes for *digest* are not in the artifact store."""
        return cls(
            f"no bytes stored for hash {digest!r} (never stored, or discarded by "
            f"retention policy; the hash and recipe may still be recorded on runs)",
            digest=digest,
        )

    @classmethod
    def for_name(cls, run_id: str, name: str) -> Self:
        """Run *run_id* has no artifact named *name*."""
        return cls(f"run {run_id!r} has no artifact named {name!r}", run_id=run_id, name=name)


class ArtifactExistsError(SlabError):
    """The run already has an artifact with this name."""

    def __init__(self, run_id: str, name: str) -> None:
        self.run_id = run_id
        self.name = name
        super().__init__(f"run {run_id!r} already has an artifact named {name!r}")


class AmbiguousHashError(SlabError):
    """A hash prefix matches more than one stored artifact.

    Attributes:
        prefix: The prefix that was looked up.
        matches: Matching content hashes.
    """

    def __init__(self, prefix: str, matches: Sequence[str]) -> None:
        self.prefix = prefix
        self.matches = list(matches)
        shown = ", ".join(m[:12] for m in self.matches[:5])
        more = ", ..." if len(self.matches) > 5 else ""
        super().__init__(
            f"hash prefix {prefix!r} is ambiguous ({len(self.matches)} matches): {shown}{more}"
        )


class SerializationError(SlabError):
    """A value could not be serialized for tracing/storage, or decoded back."""


class NoActiveRunError(SlabError):
    """A feature that needs an active run was used outside ``Workspace.start_run``."""

    def __init__(
        self,
        message: str = (
            "no active run: this can only be used inside a 'with ws.start_run(...)' block"
        ),
    ) -> None:
        super().__init__(message)


class NestedRunError(SlabError):
    """``start_run`` was called while another run is already active in this context."""

    def __init__(
        self,
        message: str = (
            "a run is already active in this context; nested start_run() is not supported"
        ),
    ) -> None:
        super().__init__(message)


class StorageError(SlabError):
    """A storage-layer failure (bad data, I/O, or invariant violation)."""


class SchemaVersionError(StorageError):
    """The on-disk database schema is newer than this version of slab supports."""

    def __init__(self, path: str, found: int, supported: int) -> None:
        self.path = path
        self.found = found
        self.supported = supported
        super().__init__(
            f"database {path!r} has schema version {found}, newer than this slab "
            f"supports ({supported}); upgrade slab to open it"
        )
