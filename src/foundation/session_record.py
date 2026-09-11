"""The record an external harness leaves: what it loaded and what it claims.

The resident agent keeps a transcript of every model call. A harness that
drives the workspace over MCP keeps its own transcript elsewhere, in a
format SLAB never sees, so the workspace has to keep the two things a
later reader needs from such a session: which skills it loaded, and what
results it reported against which runs. That is this record, one JSON
lines file per session under ``<workspace>/sessions/``. The benchmark
scorer reads it the way it reads a resident transcript, and the runs the
session launched carry the same session id, so ``list_runs`` and
``promote_session`` see the session whole.

Events are dicts with a ``type``: ``session`` (the header, written once),
``skill`` (a skill loaded, with its digest), and ``results`` (the
structured hand-back: result name to value and unit, plus the run ids
that produced them). The vocabulary is small on purpose; anything a
harness wants to remember about its own reasoning belongs in its own
transcript or in the notebook.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from foundation.errors import AmbiguousSessionError, FoundationError, SessionNotFoundError
from foundation.lifecycle import ExecutionStatus

if TYPE_CHECKING:
    from foundation.store import RunStore

RECORDS_DIR = "sessions"


class ResultsError(FoundationError):
    """A results hand-back that the scorer could not read, naming the rule."""


def new_session_id(client: str = "mcp") -> str:
    """A session id for a harness session: the client, a stamp, and the pid.

    Examples:
        >>> new_session_id("mcp").startswith("mcp-")
        True
    """
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{client}-{stamp}-{os.getpid()}"


def records_dir(root: Path) -> Path:
    return Path(root) / RECORDS_DIR


def validate_results(results: Any, run_ids: Any) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Check a hand-back's shape and return it normalized, or raise ResultsError.

    Each result is ``{"value": number, "unit": str}``; ``run_ids`` is a
    non-empty list of strings. The scorer judges the value against a band,
    so a string where a number belongs is refused here, where the client
    can still correct it.

    Examples:
        >>> validate_results({"a0": {"value": 3.6, "unit": "Å"}}, ["ab12"])
        ({'a0': {'value': 3.6, 'unit': 'Å'}}, ['ab12'])
        >>> validate_results({"a0": {"value": "3.6", "unit": "Å"}}, ["ab12"])
        Traceback (most recent call last):
        ...
        foundation.session_record.ResultsError: result 'a0' has no numeric value
    """
    if not isinstance(results, dict) or not results:
        raise ResultsError("results must map at least one result name to {value, unit}")
    clean: dict[str, dict[str, Any]] = {}
    for name, entry in results.items():
        if not isinstance(entry, dict):
            raise ResultsError(f"result {name!r} must be an object with value and unit")
        value = entry.get("value")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ResultsError(f"result {name!r} has no numeric value")
        unit = entry.get("unit")
        if not isinstance(unit, str) or not unit.strip():
            raise ResultsError(f"result {name!r} has no unit")
        clean[str(name)] = {"value": value, "unit": unit}
    if not isinstance(run_ids, list) or not run_ids or not all(
        isinstance(r, str) and r.strip() for r in run_ids
    ):
        raise ResultsError("run_ids must be a non-empty list of run ids")
    return clean, [str(r) for r in run_ids]


class SessionRecord:
    """One harness session's record: its id, its file, and its events."""

    def __init__(self, root: Path, session_id: str, *, client: str | None = None) -> None:
        self.root = Path(root)
        self.session_id = session_id
        self.client = client
        self.path = records_dir(self.root) / f"{session_id}.jsonl"

    def record(self, event: dict[str, Any]) -> Path:
        """Append one event; the header is written first when the file is new."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stamped = {"at": datetime.now(UTC).isoformat(), **event}
        with open(self.path, "a", encoding="utf-8") as handle:
            if self.path.stat().st_size == 0:
                header = {
                    "at": stamped["at"],
                    "type": "session",
                    "session": self.session_id,
                    "client": self.client,
                }
                handle.write(json.dumps(header, ensure_ascii=False) + "\n")
            handle.write(json.dumps(stamped, ensure_ascii=False) + "\n")
        return self.path

    def events(self) -> list[dict[str, Any]]:
        """Every event in order; a damaged line is skipped, not refused."""
        if not self.path.exists():
            return []
        found: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                found.append(event)
        return found

    def header(self) -> dict[str, Any]:
        return next((e for e in self.events() if e.get("type") == "session"), {})

    def skills(self) -> dict[str, str]:
        """Skill name to digest for every skill the session loaded."""
        return {
            str(e.get("name")): str(e.get("digest") or "unknown")
            for e in self.events()
            if e.get("type") == "skill"
        }

    def results(self) -> dict[str, Any] | None:
        """The last results event, or None when the session reported none."""
        found = [e for e in self.events() if e.get("type") == "results"]
        return found[-1] if found else None


def session_records(root: Path) -> list[SessionRecord]:
    """Every harness session record in the workspace, oldest first."""
    directory = records_dir(root)
    if not directory.is_dir():
        return []
    return [
        SessionRecord(root, path.stem)
        for path in sorted(directory.glob("*.jsonl"))
        if path.is_file()
    ]


def find_session_record(root: Path, session: str) -> SessionRecord:
    """The record for a session id, or a unique prefix of one."""
    records = session_records(root)
    exact = [r for r in records if r.session_id == session]
    if exact:
        return exact[0]
    matches = [r for r in records if r.session_id.startswith(session)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise SessionNotFoundError(f"no harness session record matches {session!r} under {root}")
    raise AmbiguousSessionError(session, [r.session_id for r in matches])


def stale_records(
    root: Path, *, runs: RunStore, keep_newest: bool = True
) -> list[SessionRecord]:
    """The harness records ``slab purge`` may delete, oldest first.

    A record is stale when its session id stamps no run that is still
    ``running``: the harness that wrote it has nothing in flight. With
    *keep_newest* the newest record stays, whatever its runs, the way the
    newest Mason conversation stays resumable.
    """
    records = session_records(root)
    if keep_newest and records:
        records = records[:-1]
    live = {
        run.session for run in runs.list_runs(status=ExecutionStatus.RUNNING) if run.session
    }
    return [record for record in records if record.session_id not in live]
