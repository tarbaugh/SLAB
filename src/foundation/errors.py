"""Exception types raised by Foundation, and how failures are reported.

All errors here derive from :class:`FoundationError`. Messages are written to
be actionable for the caller — including LLM agents, who read them verbatim —
so they state what was attempted, why it was refused, and what would be
allowed instead.

:func:`failure_record` is the other half of that contract: when a run or task
*fails*, the evidence (exception, trimmed traceback, diagnostic notes) is
captured as a size-bounded structure an agent can read to decide a correction,
instead of a one-line string it can only retry against.

:class:`FoundationError` deliberately does not derive from
:class:`slab.errors.SlabError`. The two packages are peers with separate
vocabularies, so a caller that wants both catches both. Foundation's own CLI
catches ``(FoundationError, SlabError)``; Mason's catches all three bases.
"""

from __future__ import annotations

import traceback as _traceback
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from foundation.lifecycle import ExecutionStatus, LifecycleState

_TRACEBACK_HEAD = 3  # outermost frames kept per group: where the work entered
_TRACEBACK_TAIL = 5  # innermost frames kept per group: where it died
_MAX_MESSAGE_CHARS = 2_000
_MAX_TRACEBACK_CHARS = 10_000
_MAX_NOTES = 10


def failure_record(exc: BaseException) -> dict[str, Any]:
    """Structured, size-bounded evidence of a failure.

    Stored on failed runs and task records and surfaced verbatim by
    ``slab show`` and the MCP ``show_run`` tool. The consumer is an LLM agent
    deciding what to do next, so the record aims to be information-rich but
    token-bounded:

    * ``type`` / ``message`` — the exception class name and its message
      (clipped past 2000 characters).
    * ``traceback`` — the formatted traceback, chained causes included.
      Contiguous frame groups keep the first 3 and last 5 frames (the entry
      point and the failure site) with the middle elided; every non-frame
      piece (notably the exception message) is clipped at 2000 characters so
      a giant message can never crowd the frames out; the whole text is
      capped at 10000 characters, keeping the end (the final exceptions of a
      long chain).
    * ``notes`` — ``Exception.add_note`` annotations, present only if any
      exist; at most 10 are listed. Tasks use notes to attach curated
      diagnostics (last energy, completed steps — see
      :func:`foundation.tasks.relax`); listing them separately lets a reader act
      without parsing the traceback at all.

    This function runs inside exception handlers, where raising would mask
    the very failure being recorded — so it never raises: hostile input
    (malformed ``__notes__``, a ``__str__`` that throws) degrades to
    placeholder text, mirroring the stdlib ``traceback`` module's own
    defenses.

    Examples:
        >>> try:
        ...     raise ValueError("SCF failed to converge in 200 iterations")
        ... except ValueError as e:
        ...     e.add_note("last SCF residual: 3.2e-2")
        ...     record = failure_record(e)
        >>> record["type"], record["message"]
        ('ValueError', 'SCF failed to converge in 200 iterations')
        >>> record["notes"]
        ['last SCF residual: 3.2e-2']
        >>> record["traceback"].startswith("Traceback (most recent call last)")
        True
    """
    record: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": _clip(_safe_str(exc), _MAX_MESSAGE_CHARS),
        "traceback": _trimmed_traceback(exc),
    }
    notes = _exception_notes(exc)
    if notes:
        record["notes"] = notes
    return record


def _exception_notes(exc: BaseException) -> list[str]:
    """Read ``exc.__notes__`` defensively; malformed values degrade, never raise.

    PEP 678 says ``__notes__`` is a sequence of strings, but nothing enforces
    that — mirror the stdlib ``traceback`` module: a non-sequence (or bare
    string) becomes a single note, unreadable values become placeholders.
    """
    try:
        raw = getattr(exc, "__notes__", None)
        if raw is None:
            return []
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            items: list[object] = [raw]
        else:
            items = list(raw)
    except Exception:
        return ["<__notes__ unreadable>"]
    notes = [_clip(_safe_str(note), _MAX_MESSAGE_CHARS) for note in items[:_MAX_NOTES]]
    if len(items) > _MAX_NOTES:
        notes.append(f"[... {len(items) - _MAX_NOTES} more note(s)]")
    return notes


def _trimmed_traceback(exc: BaseException) -> str:
    """Format *exc*'s traceback (chain included), size-bounded, never raising."""
    try:
        pieces = _traceback.format_exception(exc)
    except Exception as format_error:
        return f"<traceback unavailable: {type(format_error).__name__}: {format_error}>"
    out: list[str] = []
    frames: list[str] = []

    def flush() -> None:
        if len(frames) > _TRACEBACK_HEAD + _TRACEBACK_TAIL + 1:
            elided = len(frames) - _TRACEBACK_HEAD - _TRACEBACK_TAIL
            out.extend(frames[:_TRACEBACK_HEAD])
            out.append(f"  [... {elided} frame(s) elided ...]\n")
            out.extend(frames[-_TRACEBACK_TAIL:])
        else:
            out.extend(frames)
        frames.clear()

    for piece in pieces:
        if piece.startswith('  File "'):
            frames.append(_clip_piece(piece))
        else:
            flush()
            out.append(_clip_piece(piece))
    flush()
    text = "".join(out).rstrip("\n")
    if len(text) > _MAX_TRACEBACK_CHARS:
        text = (
            f"[traceback truncated to the final {_MAX_TRACEBACK_CHARS} characters]\n"
            + text[-_MAX_TRACEBACK_CHARS:]
        )
    return text


def _clip_piece(piece: str) -> str:
    """Clip one formatted-traceback piece, preserving its trailing newline."""
    if len(piece) <= _MAX_MESSAGE_CHARS:
        return piece
    return _clip(piece.rstrip("\n"), _MAX_MESSAGE_CHARS) + "\n"


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]} [... {len(text) - limit} characters clipped]"


def _safe_str(value: object) -> str:
    try:
        return str(value)
    except Exception:
        return f"<{type(value).__name__} str() failed>"


class FoundationError(Exception):
    """Base class for all Foundation errors."""


class IllegalTransitionError(FoundationError):
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


class IllegalStatusChangeError(FoundationError):
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


class RunNotFoundError(FoundationError):
    """No run matches the given id or id prefix."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"no run matches {run_id!r}")


class AmbiguousRunIdError(FoundationError):
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


class SessionNotFoundError(FoundationError):
    """No run carries the given session id or id prefix."""

    def __init__(self, session: str) -> None:
        self.session = session
        super().__init__(
            f"no run carries session {session!r}; list the known ones with "
            f"'slab sessions'"
        )


class AmbiguousSessionError(FoundationError):
    """A session id prefix matches more than one session.

    Attributes:
        prefix: The prefix that was looked up.
        matches: Matching session ids (capped at 6 by the store).
    """

    def __init__(self, prefix: str, matches: Sequence[str]) -> None:
        self.prefix = prefix
        self.matches = list(matches)
        count = "6 or more" if len(self.matches) >= 6 else str(len(self.matches))
        shown = ", ".join(self.matches[:5])
        more = ", ..." if len(self.matches) > 5 else ""
        super().__init__(
            f"session prefix {prefix!r} is ambiguous ({count} matches): {shown}{more}"
        )


class RunExistsError(FoundationError):
    """A run with this id already exists in the store."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(f"run {run_id!r} already exists")


class RunStateError(FoundationError):
    """An operation that is not allowed in the run's current lifecycle state."""

    def __init__(self, run_id: str, state: LifecycleState, operation: str) -> None:
        self.run_id = run_id
        self.state = state
        self.operation = operation
        super().__init__(
            f"cannot {operation} run {run_id!r}: its lifecycle state is {state.value!r}"
        )


class ArtifactNotFoundError(FoundationError):
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


class ArtifactExistsError(FoundationError):
    """The run already has an artifact with this name."""

    def __init__(self, run_id: str, name: str) -> None:
        self.run_id = run_id
        self.name = name
        super().__init__(f"run {run_id!r} already has an artifact named {name!r}")


class AmbiguousHashError(FoundationError):
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


class ScriptExitError(FoundationError):
    """A workflow script called ``sys.exit()`` with a nonzero code."""


class SerializationError(FoundationError):
    """A value could not be serialized for tracing/storage, or decoded back."""


class NoActiveRunError(FoundationError):
    """A feature that needs an active run was used outside ``Workspace.start_run``."""

    def __init__(
        self,
        message: str = (
            "no active run: this can only be used inside a 'with ws.start_run(...)' block"
        ),
    ) -> None:
        super().__init__(message)


class NestedRunError(FoundationError):
    """``start_run`` was called while another run is already active in this context."""

    def __init__(
        self,
        message: str = (
            "a run is already active in this context; nested start_run() is not supported"
        ),
    ) -> None:
        super().__init__(message)


class MemoryStoreError(FoundationError):
    """A memory that cannot be read or written, naming the file and the rule.

    Deliberately not named ``MemoryError``: that name belongs to the builtin
    raised when allocation fails, and shadowing it would make every
    ``except MemoryError`` in the stack ambiguous.
    """


class StorageError(FoundationError):
    """A storage-layer failure (bad data, I/O, or invariant violation)."""


class SchemaVersionError(StorageError):
    """The on-disk database schema is newer than this version of foundation supports."""

    def __init__(self, path: str, found: int, supported: int) -> None:
        self.path = path
        self.found = found
        self.supported = supported
        super().__init__(
            f"database {path!r} has schema version {found}, newer than this foundation "
            f"supports ({supported}); upgrade slab-stack to open it"
        )
