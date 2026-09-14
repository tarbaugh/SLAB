"""Where an artifact is: on its run, on the run a cache hit reused, or by hash.

A cache hit executes nothing, so a run whose task the cache served holds
none of the files that task keeps when it executes. Those files are
artifacts of the run where the task executed. This module follows that
edge for a read by name (:func:`find_artifact`), parses the
``run:<id>/<name>`` reference that stages a run's artifact beside a task's
script (:class:`RunReference`), and names the runs and tasks that
reference one content hash (:func:`hash_holders`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from foundation.artifacts import ArtifactStore
from foundation.errors import ArtifactNotFoundError
from foundation.models import ArtifactRef, TaskRecord
from foundation.store import SQLiteRunStore

#: The prefix that marks a ``files=`` entry as a run artifact, not a path.
RUN_REFERENCE_PREFIX = "run:"

_RUN_REFERENCE = re.compile(
    r"^run:(?P<run>[^/\s]+)/(?P<name>\S(?:.*?\S)?)(?:\s+as\s+(?P<basename>\S+))?$"
)
_HEX = re.compile(r"^[0-9a-f]+$")


@dataclass(frozen=True)
class RunReference:
    """A run's artifact named for staging: ``run:<id>/<name> [as <basename>]``.

    *run* is a run id or a unique prefix, *name* the artifact's name on
    that run, and *basename* the name the staged copy gets beside the
    script (the artifact's name by default).

    Examples:
        >>> RunReference.parse("run:01abc/md-final.data as start.data")
        RunReference(run='01abc', name='md-final.data', basename='start.data')
        >>> RunReference.parse("run:01abc/W.eam.fs").basename
        'W.eam.fs'
        >>> RunReference.parse("potentials/W.eam.fs") is None
        True
    """

    run: str
    name: str
    basename: str

    @classmethod
    def parse(cls, entry: object) -> RunReference | None:
        """The reference *entry* spells, or None when it is not one.

        Raises:
            ValueError: *entry* starts with ``run:`` and is malformed, or
                its basename has a directory component.
        """
        if not isinstance(entry, str) or not entry.startswith(RUN_REFERENCE_PREFIX):
            return None
        match = _RUN_REFERENCE.fullmatch(entry.strip())
        if match is None:
            raise ValueError(
                f"{entry!r} is not a run artifact reference; write "
                "'run:<id>/<name>' or 'run:<id>/<name> as <basename>'"
            )
        name = match["name"]
        basename = match["basename"] or name
        if "/" in basename or basename in (".", ".."):
            raise ValueError(
                f"{entry!r} stages under {basename!r}, which is not a bare basename; "
                "add 'as <basename>'"
            )
        return cls(run=match["run"], name=name, basename=basename)


@dataclass(frozen=True)
class FoundArtifact:
    """An artifact reference found for a read by name.

    *ref* is the reference found (``ref.run_id`` is the run that holds
    it), *asked* the full id of the run the caller named, and *via* the
    cache-hit task on that run whose producing run holds the reference,
    or None when the named run holds it itself.
    """

    ref: ArtifactRef
    asked: str
    via: TaskRecord | None = None

    @property
    def followed(self) -> bool:
        """Whether the read followed a cache hit to the producing run."""
        return self.via is not None

    def note(self) -> str | None:
        """One line that says the read followed the cache, or None.

        Examples:
            >>> from foundation.models import utcnow
            >>> ref = ArtifactRef(run_id="p" * 26, name="md.log", role="terminal",
            ...                   hash="ab" * 32, size_bytes=3)
            >>> hit = TaskRecord(run_id="r" * 26, seq=7, name="run_lammps",
            ...                  status="completed", cache_hit=True,
            ...                  cache_key="cd" * 32, started_at=utcnow())
            >>> print(FoundArtifact(ref, "r" * 26, hit).note())  # doctest: +NORMALIZE_WHITESPACE
            md.log is read from run pppppppppp: task 7 run_lammps on run rrrrrrrrrr
            was a cache hit of that run
        """
        if self.via is None:
            return None
        return (
            f"{self.ref.name} is read from run {self.ref.run_id[:10]}: task "
            f"{self.via.seq} {self.via.name} on run {self.asked[:10]} was a cache "
            f"hit of that run"
        )


def producing_runs(runs: SQLiteRunStore, run_id: str) -> list[tuple[TaskRecord, str]]:
    """Each cache-hit task of the run paired with the run it reused, in order.

    A task whose producing run is the run itself, or is gone, is left
    out, and each producing run appears once.
    """
    rid = runs.resolve(run_id)
    pairs: list[tuple[TaskRecord, str]] = []
    seen: set[str] = set()
    for record in runs.list_tasks(rid):
        producer = runs.producing_task(record)
        if producer is None or producer.run_id == rid or producer.run_id in seen:
            continue
        seen.add(producer.run_id)
        pairs.append((record, producer.run_id))
    return pairs


def _match(refs: list[ArtifactRef], wanted: str) -> ArtifactRef | None:
    """The reference named *wanted*, else the first whose hash starts with it."""
    for ref in refs:
        if ref.name == wanted:
            return ref
    if _HEX.fullmatch(wanted.lower()):
        for ref in refs:
            if ref.hash.startswith(wanted.lower()):
                return ref
    return None


def find_artifact(runs: SQLiteRunStore, run_id: str, name: str) -> FoundArtifact:
    """The run's artifact *name* (or hash prefix), following its cache hits.

    The run's own references come first. When none matches, each of the
    run's cache-hit tasks is followed to the run where that task
    executed, and the first match there is returned with the task that
    led to it.

    Raises:
        RunNotFoundError: No run matches *run_id*.
        ArtifactNotFoundError: Neither the run nor a run its cache hits
            reused holds the artifact. The message lists what each holds.
    """
    rid = runs.resolve(run_id)
    own = runs.list_artifacts(rid)
    if (found := _match(own, name)) is not None:
        return FoundArtifact(found, rid)
    elsewhere: list[str] = []
    for record, producer in producing_runs(runs, rid):
        theirs = runs.list_artifacts(producer)
        if (found := _match(theirs, name)) is not None:
            return FoundArtifact(found, rid, record)
        names = ", ".join(ref.name for ref in theirs) or "none"
        elsewhere.append(f"its cache hits reused run {producer[:10]}, which has: {names}")
    message = f"no artifact named {name!r} on run {rid[:10]}; it has: " + (
        ", ".join(ref.name for ref in own) or "none"
    )
    if elsewhere:
        message += "; " + "; ".join(elsewhere)
    raise ArtifactNotFoundError(message, run_id=rid, name=name)


@dataclass(frozen=True)
class ResolvedReference:
    """A :class:`RunReference` resolved to stored bytes."""

    reference: RunReference
    found: FoundArtifact
    path: Path

    @property
    def digest(self) -> str:
        """The SHA-256 of the artifact's bytes."""
        return self.found.ref.hash

    @property
    def canonical(self) -> str:
        """The entry's cache identity: the bytes and the staged name, no run id."""
        return f"sha256:{self.digest} as {self.reference.basename}"

    @property
    def source(self) -> str:
        """The run that holds the bytes, as a reference, for the recipe."""
        return f"run:{self.found.ref.run_id}/{self.found.ref.name}"


def resolve_reference(
    runs: SQLiteRunStore, artifacts: ArtifactStore, reference: RunReference
) -> ResolvedReference:
    """Find the bytes a run reference names, following the run's cache hits.

    Raises:
        RunNotFoundError: No run matches the reference's run.
        ArtifactNotFoundError: The run holds no such artifact, or
            retention has discarded its bytes.
    """
    found = find_artifact(runs, reference.run, reference.name)
    if not artifacts.has(found.ref.hash):
        raise ArtifactNotFoundError(
            f"the bytes of {found.ref.name!r} on run {found.ref.run_id[:10]} are no "
            f"longer stored (retention reclaimed them); the record keeps its hash "
            f"{found.ref.hash[:12]}",
            digest=found.ref.hash,
            run_id=found.ref.run_id,
            name=found.ref.name,
        )
    return ResolvedReference(reference, found, artifacts.get(found.ref.hash))


@dataclass(frozen=True)
class HashHolders:
    """The stored bytes one hash names, and who references them.

    *artifacts* are the named references on runs; *tasks* pair a task
    with the slot that names the hash (``output return[1]``,
    ``input atoms``).
    """

    digest: str
    path: Path
    artifacts: list[ArtifactRef] = field(default_factory=list)
    tasks: list[tuple[TaskRecord, str]] = field(default_factory=list)

    def lines(self) -> list[str]:
        """One line per reference, artifacts first.

        Examples:
            >>> from foundation.models import utcnow
            >>> ref = ArtifactRef(run_id="p" * 26, name="md.log", role="terminal",
            ...                   hash="ab" * 32, size_bytes=3)
            >>> t = TaskRecord(run_id="r" * 26, seq=4, name="run_lammps",
            ...                status="completed", cache_key="cd" * 32, started_at=utcnow())
            >>> holders = HashHolders("ab" * 32, Path("x"), [ref], [(t, "output return[0]")])
            >>> for line in holders.lines():
            ...     print(line)
            run pppppppppp artifact md.log (terminal)
            run rrrrrrrrrr task 4 run_lammps output return[0]
        """
        lines = [
            f"run {ref.run_id[:10]} artifact {ref.name} ({ref.role.value})"
            for ref in self.artifacts
        ]
        lines.extend(
            f"run {record.run_id[:10]} task {record.seq} {record.name} {slot}"
            for record, slot in self.tasks
        )
        return lines


def hash_holders(runs: SQLiteRunStore, artifacts: ArtifactStore, prefix: str) -> HashHolders:
    """The stored bytes a hash (or unique prefix of 6+ characters) names.

    Raises:
        ValueError: The prefix is shorter than 6 characters or not hex.
        AmbiguousHashError: The prefix matches several stored artifacts.
        ArtifactNotFoundError: Nothing stored matches. When runs still
            record the hash, the message says retention reclaimed the bytes.
    """
    try:
        digest = artifacts.resolve(prefix.strip())
    except ArtifactNotFoundError:
        recorded = runs.artifacts_with_hash(prefix.strip())
        if not recorded:
            raise
        where = ", ".join(f"run {ref.run_id[:10]} as {ref.name}" for ref in recorded[:5])
        raise ArtifactNotFoundError(
            f"the bytes of hash {prefix.strip()!r} are no longer stored (retention "
            f"reclaimed them); the hash is recorded on {where}",
            digest=prefix.strip(),
        ) from None
    tasks: list[tuple[TaskRecord, str]] = []
    for record in runs.tasks_with_hash(digest):
        for slot, value in record.inputs.items():
            if value == digest:
                tasks.append((record, f"input {slot}"))
        for slot, value in record.outputs.items():
            if value == digest:
                tasks.append((record, f"output {slot}"))
    return HashHolders(
        digest=digest,
        path=artifacts.get(digest),
        artifacts=[ref for ref in runs.artifacts_with_hash(digest) if ref.hash == digest],
        tasks=tasks,
    )
