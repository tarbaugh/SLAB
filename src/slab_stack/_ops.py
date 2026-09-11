"""The inventory behind ``slab purge``: every category it deletes, counted first.

Purge deletes what it knows of, and this module is where it learns of
everything: the expired runs and the blobs only they reach, the session
transcripts with their sidecars, the files under ``mason/sessions`` that
no transcript claims, the harness session records, the stale session
locks, the finished jobs' files, and the scratch directories no live
calculation owns. One function lists them all with counts and bytes, so
the dry run, the confirmation, and the deletion describe the same set.

This lives in ``slab_stack`` because the categories come from all three
layers: the run store and the sweep from ``foundation``, the transcript
layout and the locks from ``mason``, the scratch marker from ``slab``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from foundation.retention import sweep_scratch
from foundation.runtime import Workspace
from foundation.session_record import stale_records
from mason.serve import mason_dir
from mason.session import (
    session_sidecars,
    stale_locks,
    transcript_groups,
    unrecognised_session_files,
)

CATEGORIES = (
    "stale locks",
    "transcripts",
    "sidecars",
    "unrecognised",
    "harness records",
    "job files",
    "expired runs",
    "blobs",
    "scratch",
)
"""The categories in the order purge deletes them.

Stale locks go first, and are probed again just before they are unlinked,
because a session that started during the inventory holds the same file:
the lock path is the project's, so unlinking a held lock would let a third
session take a fresh one beside it.
"""


class Category(BaseModel):
    """One kind of thing purge deletes: its items and their size."""

    model_config = ConfigDict(frozen=True)

    name: str
    items: list[str]
    bytes: int = 0

    @property
    def count(self) -> int:
        return len(self.items)


class Kept(BaseModel):
    """One thing purge saw and leaves alone, with the reason."""

    model_config = ConfigDict(frozen=True)

    kind: str
    item: str
    reason: str


class Inventory(BaseModel):
    """What one purge deletes (or, with ``dry_run``, would delete), and what it keeps.

    ``categories`` follow :data:`CATEGORIES` in order. Workspace paths are
    relative to the root; scratch paths are as the sweep reports them,
    under ``[paths] scratch``.
    """

    model_config = ConfigDict(frozen=True)

    root: str
    all_sessions: bool
    dry_run: bool
    categories: list[Category]
    kept: list[Kept]

    @property
    def total_bytes(self) -> int:
        return sum(c.bytes for c in self.categories)

    def lines(self, verb: str, *, detail: bool = True) -> list[str]:
        """One line per category, its items under it with *detail*, then one per kept item.

        Examples:
            >>> inventory = Inventory(root="/ws", all_sessions=False, dry_run=True,
            ...     categories=[Category(name="blobs", items=["ab12"], bytes=7),
            ...                 Category(name="job files", items=[])],
            ...     kept=[Kept(kind="transcript", item="mason/sessions/x.jsonl",
            ...                reason="the newest conversation")])
            >>> for line in inventory.lines("would delete"):
            ...     print(line)
            would delete blobs: 1 (7 bytes)
              ab12
            would delete job files: none
            kept transcript mason/sessions/x.jsonl: the newest conversation
        """
        lines: list[str] = []
        for category in self.categories:
            if not category.items:
                lines.append(f"{verb} {category.name}: none")
                continue
            size = f" ({category.bytes} bytes)" if category.bytes else ""
            lines.append(f"{verb} {category.name}: {category.count}{size}")
            if detail:
                lines.extend(f"  {item}" for item in category.items)
        lines.extend(f"kept {k.kind} {k.item}: {k.reason}" for k in self.kept)
        return lines

    def summary(self) -> str:
        """The non-empty categories with their counts, for the confirmation.

        Examples:
            >>> Inventory(root="/ws", all_sessions=False, dry_run=False,
            ...     categories=[Category(name="blobs", items=["ab12"], bytes=7),
            ...                 Category(name="job files", items=[])], kept=[]).summary()
            'blobs: 1 (7 bytes in all)'
        """
        parts = [f"{c.name}: {c.count}" for c in self.categories if c.items]
        if not parts:
            return "nothing"
        return f"{', '.join(parts)} ({self.total_bytes} bytes in all)"


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _size(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


_JOB_FILE_PATTERNS = ("*.sbatch", "*.out")

JobIdOf = Callable[[str], str | None]
"""Maps a job file's name to the job id it carries, or ``None``."""


def job_files(
    root: Path, active: frozenset[str], job_id_of: JobIdOf
) -> tuple[list[Path], list[Path]]:
    """Finished jobs' scripts and SLURM output files, and the live ones kept.

    Two directories hold them: ``<workspace>/jobs`` (the agent's submitted
    jobs) and ``<workspace>/mason`` (serve jobs). Only ``*.sbatch`` and
    ``*.out`` are candidates, so the serve endpoint record is never
    touched. A file whose embedded job id is still in the queue is kept:
    SLURM is writing its ``.out``. *job_id_of* maps a file name to its
    job id or ``None``.
    """
    gone: list[Path] = []
    live: list[Path] = []
    for directory in (root / "jobs", mason_dir(root)):
        if not directory.is_dir():
            continue
        for pattern in _JOB_FILE_PATTERNS:
            for path in sorted(directory.glob(pattern)):
                job_id = job_id_of(path.name)
                (live if job_id is not None and job_id in active else gone).append(path)
    return sorted(gone), sorted(live)


def purge_inventory(
    root: Path,
    *,
    all_sessions: bool,
    active: frozenset[str],
    job_id_of: JobIdOf,
    dry_run: bool = True,
) -> Inventory:
    """List everything ``slab purge`` deletes from *root*, with counts and bytes.

    With *dry_run* nothing is touched. Without it the same categories are
    deleted in order, and the inventory returned is what actually went:
    the run store reports the rows and blobs it removed, and the sweep
    reports the scratch it removed. *active* is the set of job ids the
    scheduler still holds, the serve record's job included.
    """
    root = Path(root)
    groups = transcript_groups(root, include_orphans=True)
    kept: list[Kept] = []
    if groups and not all_sessions:
        newest = transcript_groups(root)
        if newest:
            conversation, siblings = newest[-1]
            groups = [g for g in groups if g[0] != conversation]
            for path in (conversation, *siblings):
                kept.append(
                    Kept(
                        kind="transcript",
                        item=_relative(root, path),
                        reason="the newest conversation (--all-sessions removes it too)",
                    )
                )
            for path in session_sidecars(root, conversation):
                kept.append(
                    Kept(
                        kind="sidecar",
                        item=_relative(root, path),
                        reason="the newest conversation (--all-sessions removes it too)",
                    )
                )
    transcripts = [path for conversation, siblings in groups for path in (conversation, *siblings)]
    sidecars = [path for conversation, _ in groups for path in session_sidecars(root, conversation)]
    unrecognised = unrecognised_session_files(root)
    if not all_sessions:
        for path in unrecognised:
            kept.append(
                Kept(
                    kind="unrecognised",
                    item=_relative(root, path),
                    reason="no transcript claims it (--all-sessions removes it)",
                )
            )
        unrecognised = []
    locks = stale_locks(root)
    gone_jobs, live_jobs = job_files(root, active, job_id_of)
    for path in live_jobs:
        kept.append(
            Kept(kind="job file", item=_relative(root, path), reason="its job is in the queue")
        )

    def files(name: str, paths: list[Path]) -> Category:
        # Sized before anything is unlinked, so the deleted inventory
        # reports what went, not what is left.
        return Category(name=name, items=[_relative(root, p) for p in paths], bytes=_size(paths))

    with Workspace(root) as ws:
        records = [
            r.path for r in stale_records(root, runs=ws.runs, keep_newest=not all_sessions)
        ]
        names = {run.id: run.name for run in ws.runs.list_runs(state="expired")}
        if not dry_run:
            # A session that started since the probe holds its lock now.
            still_stale = set(stale_locks(root))
            locks = [path for path in locks if path in still_stale]
        categories = [
            files("stale locks", locks),
            files("transcripts", transcripts),
            files("sidecars", sidecars),
            files("unrecognised", unrecognised),
            files("harness records", records),
            files("job files", gone_jobs),
        ]
        if not dry_run:
            for path in [*locks, *transcripts, *sidecars, *unrecognised, *records, *gone_jobs]:
                path.unlink(missing_ok=True)
        report = ws.purge_expired(dry_run=dry_run)
        scratch = sweep_scratch(ws, dry_run=dry_run)
    for entry in scratch.kept:
        kept.append(Kept(kind="scratch", item=entry["path"], reason=entry["reason"]))
    categories.extend(
        [
            Category(
                name="expired runs",
                items=[f"{run_id}  {names.get(run_id, '')}".rstrip() for run_id in report.deleted],
            ),
            Category(name="blobs", items=list(report.dropped), bytes=report.freed_bytes),
            Category(
                name="scratch",
                items=[entry["path"] for entry in scratch.removed],
                bytes=scratch.freed_bytes,
            ),
        ]
    )
    return Inventory(
        root=str(root),
        all_sessions=all_sessions,
        dry_run=dry_run,
        categories=categories,
        kept=kept,
    )
