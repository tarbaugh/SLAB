"""Shared operations behind the CLI and the MCP server — one code path, two skins.

Everything here returns plain JSON-able dicts (or domain objects the callers
format), so the CLI renders text, the MCP server returns structure, and the
behavior cannot drift between them.
"""

from __future__ import annotations

import io
import json
import os
import re
import runpy
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from slab.errors import NestedRunError, SlabError
from slab.models import Run
from slab.retention import DEFAULT_POLICY, RetentionPolicy
from slab.runtime import Workspace

DEFAULT_ROOT = ".slab"
_DURATION = re.compile(r"^(\d+(?:\.\d+)?)\s*([smhd])$")
_DAYS_PER_UNIT = {"s": 1 / 86_400, "m": 1 / 1_440, "h": 1 / 24, "d": 1.0}
_EFFECTIVELY_NOW = 1e-9  # ~90µs in days: "--older-than 0d" means "everything, now"


def resolve_root(explicit: str | os.PathLike[str] | None) -> Path:
    """Workspace root: explicit flag, else $SLAB_WORKSPACE, else ``./.slab``.

    Examples:
        >>> resolve_root("/tmp/ws")
        PosixPath('/tmp/ws')
        >>> import os
        >>> os.environ.pop("SLAB_WORKSPACE", None) and None
        >>> resolve_root(None)
        PosixPath('.slab')
    """
    if explicit is not None:
        return Path(explicit)
    from_env = os.environ.get("SLAB_WORKSPACE")
    return Path(from_env) if from_env else Path(DEFAULT_ROOT)


def parse_duration_days(text: str) -> float:
    """Parse ``"30d"``, ``"12h"``, ``"45m"``, ``"90s"`` into days.

    Zero means "expire everything unpromoted, now" and maps to an epsilon
    (retention TTLs must be positive).

    Examples:
        >>> parse_duration_days("30d")
        30.0
        >>> parse_duration_days("12h")
        0.5
        >>> parse_duration_days("0d")
        1e-09
    """
    match = _DURATION.fullmatch(text.strip().lower())
    if match is None:
        raise ValueError(
            f"cannot parse duration {text!r}: use <number><unit> with unit s/m/h/d, e.g. 30d"
        )
    value = float(match.group(1)) * _DAYS_PER_UNIT[match.group(2)]
    return value if value > 0 else _EFFECTIVELY_NOW


def load_policy(root: Path, explicit_path: str | os.PathLike[str] | None = None) -> RetentionPolicy:
    """Load the retention policy: explicit file, else ``<root>/policy.json``, else defaults.

    Examples:
        >>> import tempfile
        >>> load_policy(Path(tempfile.mkdtemp())) is DEFAULT_POLICY
        True
    """
    path = Path(explicit_path) if explicit_path is not None else root / "policy.json"
    if explicit_path is None and not path.exists():
        return DEFAULT_POLICY
    with open(path, encoding="utf-8") as handle:
        return RetentionPolicy.model_validate(json.load(handle))


def ttl_override_policy(days: float) -> RetentionPolicy:
    """A policy whose only effect is 'expire quarantined/verified older than *days*'.

    Examples:
        >>> ttl_override_policy(7).verified.ttl_days
        7.0
    """
    return RetentionPolicy.model_validate(
        {"quarantined": {"ttl_days": days}, "verified": {"ttl_days": days}}
    )


def run_summary(run: Run) -> dict[str, Any]:
    """Compact JSON-able view of a run (used by list/promote/launch results)."""
    return {
        "id": run.id,
        "name": run.name,
        "state": run.state.value,
        "status": run.status.value,
        "intent": run.intent,
        "error": run.error,
        "created_at": run.created_at.isoformat(),
        "state_entered_at": run.state_entered_at.isoformat(),
    }


def run_details(ws: Workspace, run_id: str) -> dict[str, Any]:
    """Everything about one run: fields, checks, tasks, artifacts, history.

    Artifact entries carry ``bytes_available`` — whether the content is still
    in the artifact store or has been hash-and-discarded.
    """
    run = ws.runs.get(run_id)
    checks = ws.runs.list_check_results(run.id)
    tasks = ws.runs.list_tasks(run.id)
    artifacts = ws.runs.list_artifacts(run.id)
    history = ws.runs.history(run.id)
    return {
        "run": run_summary(run)
        | {
            "meta": run.meta,
            "started_at": None if run.started_at is None else run.started_at.isoformat(),
            "finished_at": None if run.finished_at is None else run.finished_at.isoformat(),
        },
        "checks": [
            {
                "name": c.name,
                "kind": c.kind,
                "passed": c.passed,
                "message": c.message,
            }
            for c in checks
        ],
        "tasks": [
            {
                "seq": t.seq,
                "name": t.name,
                "status": t.status.value,
                "cache_hit": t.cache_hit,
                "error": t.error,
                "duration_s": (
                    None
                    if t.finished_at is None
                    else round((t.finished_at - t.started_at).total_seconds(), 3)
                ),
                "recipe": t.recipe,
                "inputs": t.inputs,
                "outputs": t.outputs,
            }
            for t in tasks
        ],
        "artifacts": [
            {
                "name": a.name,
                "role": a.role.value,
                "hash": a.hash,
                "size_bytes": a.size_bytes,
                "bytes_available": ws.artifacts.has(a.hash),
            }
            for a in artifacts
        ],
        "history": [
            {
                "from": t.from_state.value,
                "to": t.to_state.value,
                "actor": t.actor,
                "reason": t.reason,
                "forced": t.forced,
                "at": t.at.isoformat(),
            }
            for t in history
        ],
    }


def launch_script(
    root: Path,
    script: str | os.PathLike[str],
    *,
    name: str | None = None,
    intent: str | None = None,
    argv: tuple[str, ...] = (),
    capture_output: bool = False,
) -> dict[str, Any]:
    """Execute a workflow script inside a fresh run context; return the outcome.

    This is ``slab run`` and the MCP ``launch_workflow`` tool: the script is
    plain Python with ``@task`` calls and ``@check`` declarations — the runner
    supplies the workspace and the run context, so scripts carry zero
    ceremony. Scripts that manage their own ``Workspace.start_run`` should be
    executed with plain ``python`` instead (nesting is refused with a hint).

    The result dict carries ``run_id``, final ``state``/``status``, check
    counts, ``error`` (if the script raised), and — with
    ``capture_output=True`` — everything the script printed (used by the MCP
    server, whose stdout is the protocol channel).
    """
    script_path = Path(script).resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"no such workflow script: {script_path}")

    buffer = io.StringIO()
    old_argv = sys.argv
    sys.argv = [str(script_path), *argv]
    sys.path.insert(0, str(script_path.parent))
    error: str | None = None
    with Workspace(root) as ws:
        try:
            with ws.start_run(name=name or script_path.stem, intent=intent) as active:
                run_id = active.id
                if capture_output:
                    with redirect_stdout(buffer), redirect_stderr(buffer):
                        runpy.run_path(str(script_path), run_name="__main__")
                else:
                    runpy.run_path(str(script_path), run_name="__main__")
        except NestedRunError:
            raise SlabError(
                f"{script_path.name} manages its own runs (it calls start_run); "
                f"execute it with plain 'python {script_path.name}' instead of 'slab run'"
            ) from None
        except Exception:
            error = traceback.format_exc(limit=8)
        finally:
            sys.argv = old_argv
            sys.path.remove(str(script_path.parent))

        run = ws.runs.get(run_id)
        checks = ws.runs.list_check_results(run_id)
        result: dict[str, Any] = run_summary(run) | {
            "run_id": run.id,
            "checks_passed": sum(1 for c in checks if c.passed),
            "checks_total": len(checks),
            "tasks_recorded": len(ws.runs.list_tasks(run_id)),
        }
    if error is not None:
        result["traceback"] = error
    if capture_output:
        result["output"] = buffer.getvalue()
    return result
