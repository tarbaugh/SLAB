"""Exception types raised by SLAB.

All errors derive from :class:`SlabError`. Messages are written to be actionable
for the caller — including LLM agents, who read them verbatim — so they state
what was attempted, why it was refused, and what would be allowed instead.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

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
