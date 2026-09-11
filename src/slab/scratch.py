"""Ownership of slab-managed scratch directories, and the inventory of them.

Every scratch directory that :func:`slab.backends._scratch_dir` makes
carries a marker file naming the process that made it, the host it ran
on, and the run it belonged to. The marker is what a later sweep reads:
ownership is recorded here, never inferred from a directory's age. A
directory with no marker, or whose marker names no run, is owned by a
process alone, and it is a leftover once that process is gone.

This module lists and judges nothing beyond that. :func:`leftovers` is
the inventory, and the decision to remove one belongs to
``foundation.retention.sweep_scratch``, which knows the run store.
"""

from __future__ import annotations

import os
import socket
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

OWNER_MARKER = ".slab-owner"
"""The file a slab-managed scratch directory carries to name its owner."""

RUN_ENV = "SLAB_RUN_ID"
"""The variable a run exports so every scratch made inside it is stamped."""

SESSION_ENV = "SLAB_SESSION"


def this_host() -> str:
    """The hostname a marker records, the same one the run store stamps.

    Examples:
        >>> isinstance(this_host(), str) and this_host() != ""
        True
    """
    return socket.gethostname()


def process_alive(pid: int) -> bool:
    """Whether a process with *pid* exists on this host.

    Signal 0 probes without delivering: a process that belongs to another
    user answers with a permission error, which still means it exists.

    A dead child of this process is reaped first. A background launch
    abandons its ``Popen`` object, so a child that an OOM kill or a
    ``SIGKILL`` ended stays a zombie until something waits for it, and a
    zombie still answers signal 0. One non-blocking ``waitpid`` on *pid*
    collects it when it is a dead child of this process, and reports it
    dead; a pid that is not a child answers ``ChildProcessError``, which
    is ignored. This is the whole mechanism: no list of abandoned
    ``Popen`` objects is kept, because ``waitpid`` on the recorded pid
    reaches the same child without one.

    Examples:
        >>> process_alive(os.getpid())
        True
        >>> process_alive(0)
        False
    """
    if pid <= 0:
        return False
    with suppress(ChildProcessError):
        reaped, _ = os.waitpid(pid, os.WNOHANG)
        if reaped == pid:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Owner(BaseModel):
    """What a scratch directory's marker says about who made it.

    ``run_id`` and ``session`` are the run and the session that were
    current when the directory was made, from ``$SLAB_RUN_ID`` and
    ``$SLAB_SESSION``; either is ``None`` outside one.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    pid: int
    host: str
    created_at: str
    prefix: str = ""
    run_id: str | None = None
    session: str | None = None


def mark_owner(directory: Path, *, prefix: str = "") -> Path:
    """Write the marker into *directory* for this process; return its path.

    Examples:
        >>> import tempfile
        >>> made = Path(tempfile.mkdtemp())
        >>> marker = mark_owner(made, prefix="slab-qe-")
        >>> Owner.model_validate_json(marker.read_text()).pid == os.getpid()
        True
    """
    owner = Owner(
        pid=os.getpid(),
        host=this_host(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        prefix=prefix,
        run_id=os.environ.get(RUN_ENV) or None,
        session=os.environ.get(SESSION_ENV) or None,
    )
    marker = Path(directory) / OWNER_MARKER
    marker.write_text(owner.model_dump_json() + "\n", encoding="utf-8")
    return marker


def read_owner(directory: Path) -> Owner | None:
    """The marker in *directory*, or ``None`` when it is missing or unreadable."""
    marker = Path(directory) / OWNER_MARKER
    try:
        return Owner.model_validate_json(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class Leftover(BaseModel):
    """One ``slab-*`` directory under the scratch root, as the inventory sees it.

    Fields:
        path: The directory.
        size_bytes: Its size, every file summed.
        owner: The marker's content, or ``None`` when there is no readable marker.
        unowned: No run claims it: there is no marker, or the marker names no run.
        alive: Whether the owner's process exists on this host. ``None``
            when there is no marker, or the marker names another host,
            because nothing about that host can be seen from here.
    """

    model_config = ConfigDict(frozen=True)

    path: Path
    size_bytes: int
    owner: Owner | None
    unowned: bool
    alive: bool | None


def _size_of(directory: Path) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(directory):
        for name in filenames:
            with suppress(OSError):
                total += (Path(dirpath) / name).stat().st_size
    return total


def leftovers(root: str | os.PathLike[str]) -> list[Leftover]:
    """Every ``slab-*`` directory under *root* with its marker read. Deletes nothing.

    Only ``slab-*`` entries are listed, so a scratch root shared with
    other tools is never read beyond them.

    Examples:
        >>> import tempfile
        >>> root = Path(tempfile.mkdtemp())
        >>> mine = root / "slab-qe-abc"
        >>> mine.mkdir()
        >>> _ = mark_owner(mine, prefix="slab-qe-")
        >>> (root / "slab-lammps-xyz").mkdir()
        >>> (root / "other-tool").mkdir()
        >>> [(l.path.name, l.unowned, l.alive) for l in leftovers(root)]
        [('slab-lammps-xyz', True, None), ('slab-qe-abc', True, True)]
    """
    base = Path(root).expanduser()
    if not base.is_dir():
        return []
    host = this_host()
    found: list[Leftover] = []
    for path in sorted(base.glob("slab-*")):
        if not path.is_dir():
            continue
        owner = read_owner(path)
        alive: bool | None = None
        if owner is not None and owner.host == host:
            alive = process_alive(owner.pid)
        found.append(
            Leftover(
                path=path,
                size_bytes=_size_of(path),
                owner=owner,
                unowned=owner is None or owner.run_id is None,
                alive=alive,
            )
        )
    return found


def scratch_root() -> Path | None:
    """The configured ``[paths] scratch`` root, or ``None`` when unset.

    Only this root is ever swept. The platform temp directory, which
    a scratch made without the setting lands in, is never listed.
    """
    from slab.config import config_value

    root = config_value("paths.scratch")
    return None if root is None else Path(str(root)).expanduser()
