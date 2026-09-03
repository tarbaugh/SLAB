"""The whole-stack preflight behind ``slab doctor``.

One command that means "ready to launch a campaign". Each row probes the
real campaign path — the configuration, the workspace, the memory store,
the engines, the scheduler, the mp snapshot, the gracemaker trainer, the
model endpoint, the sandbox, and the freshness of the rendered job — and
the command exits nonzero only on an ``x`` row. An ``=`` row is a fact,
not a failure: a laptop with no scheduler is healthy, and the doctor must
say so rather than fail it.

This lives in ``slab_stack`` because it is the one package allowed to
import all three layers.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from foundation import _ops
from foundation import memory as memory_store
from foundation.errors import FoundationError
from foundation.runtime import Workspace
from mason import doctor as mason_doctor
from mason.errors import MasonError
from slab._ops import engines_overview
from slab.errors import SlabError

if TYPE_CHECKING:
    from mason.config import AgentConfig
    from slab.config import HpcConfig, SlabConfig

Emit = Callable[[str], None]

_ERRORS = (MasonError, FoundationError, SlabError, OSError, ValueError)


def _config_rows() -> tuple[list[tuple[str, str]], Any, Any]:
    """Both owners validate their tables; returns (rows, slab_cfg, agent)."""
    rows: list[tuple[str, str]] = []
    slab_cfg = agent = None
    try:
        from slab.config import load_config_with_origins

        slab_cfg, merge = load_config_with_origins()
        if merge.files:
            layers = ", ".join(f"{layer} {path}" for layer, path in merge.files)
            rows.append(("+", f"config: {layers}"))
        else:
            rows.append(("=", "config: no files; built-in defaults apply everywhere"))
    except _ERRORS as e:
        rows.append(("x", f"config (slab tables): {e}"))
    try:
        from mason.config import load_config

        agent = load_config().agent
        rows.append(("+", "config [agent]: validates"))
    except _ERRORS as e:
        rows.append(("x", f"config [agent]: {e}"))
    return rows, slab_cfg, agent


def _workspace_row(workspace: Path | None) -> tuple[str, str]:
    try:
        root = _ops.resolve_root(workspace)
    except _ERRORS as e:
        return ("x", f"workspace: {e}")
    if not Path(root).exists():
        return ("=", f"workspace: none yet at {root} (created on first use)")
    try:
        with Workspace(root) as ws:
            mode, wanted = ws.runs.journal_mode, ws.runs.journal_mode_wanted
    except _ERRORS as e:
        return ("x", f"workspace: {e}")
    if mode != wanted:
        # A campaign started before an upgrade holds the database in the
        # old mode; the store keeps that mode rather than refuse, and the
        # switch lands on the next open with nothing else holding it.
        return (
            "=",
            f"workspace: {root} opens with {mode} journaling; {wanted} is wanted here, "
            f"and the switch lands on the next open with no other process holding it",
        )
    return ("+", f"workspace: {root} opens ({mode} journaling)")


def _memory_row() -> tuple[str, str]:
    directory = memory_store.memory_dir()
    probe = directory / ".doctor-probe"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe.write_text("", encoding="utf-8")
        probe.unlink()
    except OSError as e:
        return ("x", f"memory: {directory} is not writable ({e})")
    return ("+", f"memory: {directory} is writable")


def _engines_rows() -> tuple[list[tuple[str, str]], list[str]]:
    """The engines overview as rows, plus the checkpoint ids --deep probes."""
    rows: list[tuple[str, str]] = []
    checkpoint_ids: list[str] = []
    try:
        overview = engines_overview(None)
    except _ERRORS as e:
        return [("x", f"engines: {e}")], []
    rows.append(("+", f"engines built-in: {', '.join(overview['builtin'])}"))
    registry = overview["registry"]
    if registry is None:
        rows.append(("=", "engine registry: none configured"))
    else:
        rows.append(
            ("+", f"engine registry: {len(registry['engines'])} engine(s) at {registry['path']}")
        )
    rootstock = overview["rootstock"]
    if rootstock is None:
        rows.append(("=", "rootstock: not installed — no served MLIP checkpoints here"))
    elif rootstock.get("error"):
        rows.append(("=", f"rootstock: {rootstock['error']}"))
    else:
        checkpoint_ids = [
            checkpoint
            for ids in rootstock["checkpoints"].values()
            for checkpoint in ids
        ]
        rows.append(
            ("+", f"rootstock: {rootstock['root']} declares "
                  f"{len(checkpoint_ids)} checkpoint(s)")
        )
    families = overview.get("pseudo_families") or []
    if overview.get("pseudo_families_error"):
        rows.append(("x", f"pseudo families: {overview['pseudo_families_error']}"))
    else:
        rows.append(("=", f"pseudo families: {len(families)} installed"))
    return rows, checkpoint_ids


def _hpc_row(slab_cfg: SlabConfig | None) -> tuple[str, str]:
    if shutil.which("sbatch") is None:
        return ("=", "scheduler: no sbatch on PATH (fine off-cluster)")
    hpc: HpcConfig | None = getattr(slab_cfg, "hpc", None)
    if hpc is None or not hpc.partitions:
        return ("=", "scheduler: sbatch found, but no [hpc] partitions declared")
    return ("+", f"scheduler: sbatch found; partitions {', '.join(sorted(hpc.partitions))}")


def _mp_rows(slab_cfg: SlabConfig | None) -> tuple[list[tuple[str, str]], Path | None]:
    """The mp snapshot as rows, plus its root for ``--deep``.

    Opening the database and counting materials IS the health check: a
    configured root whose sqlite is missing or unreadable is a failing
    row, and a laptop with no snapshot is a healthy fact.
    """
    builders = getattr(slab_cfg, "builders", None)
    root_value = getattr(getattr(builders, "mp", None), "root", None)
    if not root_value:
        return [("=", "mp snapshot: not configured ([builders.mp] has no root)")], None
    from slab.mp import snapshot_info

    try:
        info = snapshot_info(root_value)
    except _ERRORS as e:
        return [("x", f"mp snapshot: {e}")], None
    release = f"release {info['release']}" if info["release"] else "release unknown"
    return (
        [("+", f"mp snapshot: {release}, {info['materials']} materials at {info['root']}")],
        Path(str(info["root"])),
    )


def _gracemaker_row(slab_cfg: SlabConfig | None) -> tuple[str, str]:
    """The gracemaker trainer as one row.

    Probing the tensorpotential version through the configured setup shell
    IS the health check: it exercises the module loads and the environment
    activation end to end, and a configured trainer whose environment
    cannot answer is a failing row. A laptop with no trainer is a healthy
    fact.
    """
    builders = getattr(slab_cfg, "builders", None)
    trainer = getattr(builders, "gracemaker", None)
    if trainer is None or (not trainer.command and not trainer.setup):
        return ("=", "gracemaker: not configured ([builders.gracemaker] is empty)")
    from slab.gracemaker import gracemaker_command, gracemaker_version

    try:
        version = gracemaker_version()
    except _ERRORS as e:  # the probe swallows its own errors; config can still refuse
        return ("x", f"gracemaker: {e}")
    command = gracemaker_command()
    if version is None:
        return (
            "x",
            f"gracemaker: configured, but the environment behind {command!r} "
            "does not answer a tensorpotential version probe — check the "
            "[builders.gracemaker] setup lines",
        )
    return ("+", f"gracemaker: tensorpotential {version} via {command}")


def _mp_deep_rows(root: Path | None, sample: int = 10) -> list[tuple[str, str]]:
    """``PRAGMA quick_check`` plus a sample of ``cif_path`` rows resolved and
    stat'ed — the probe that catches a corrupt database or a truncated
    transfer of the ``cifs/`` tree."""
    if root is None:
        return []
    import contextlib
    import sqlite3

    from slab.mp import connect, structure_path

    try:
        with contextlib.closing(connect(root)) as connection:
            verdict = connection.execute("PRAGMA quick_check").fetchone()[0]
            ids = [
                row[0]
                for row in connection.execute(
                    "SELECT material_id FROM materials WHERE cif_path IS NOT NULL "
                    "ORDER BY material_id LIMIT ?",
                    (sample,),
                )
            ]
    except (*_ERRORS, sqlite3.Error) as e:
        return [("x", f"deep mp: {e}")]
    rows: list[tuple[str, str]] = []
    if verdict == "ok":
        rows.append(("+", "deep mp: metadata.sqlite quick_check ok"))
    else:
        rows.append(("x", f"deep mp: quick_check says {verdict}"))
    if not ids:
        rows.append(("=", "deep mp: no CIF paths recorded to sample"))
        return rows
    missing = 0
    first_error: str | None = None
    for material_id in ids:
        try:
            structure_path(material_id, root=root)
        except _ERRORS as e:
            missing += 1
            first_error = first_error or str(e)
    if missing:
        rows.append(
            ("x", f"deep mp: {missing}/{len(ids)} sampled CIFs unresolvable — {first_error}")
        )
    else:
        rows.append(("+", f"deep mp: {len(ids)} sampled CIFs resolve and exist"))
    return rows


def _endpoint_rows(
    agent: AgentConfig, workspace: Path | None, cluster: str, emit: Emit
) -> int:
    emit("endpoint:")
    try:
        root = _ops.resolve_root(workspace)
        return mason_doctor.run(
            agent, root, cluster=cluster, emit=lambda line: emit(f"  {line}")
        )
    except _ERRORS as e:
        emit(f"  [x] {e}")
        return 1


def _sandbox_rows(agent: AgentConfig, workspace: Path | None) -> list[tuple[str, str]]:
    if not agent.sandbox.image:
        return [("=", "sandbox: not configured ([agent.sandbox] has no image)")]
    from mason.sandbox import preflight

    try:
        root = _ops.resolve_root(workspace)
        return [(mark, f"sandbox: {message}") for mark, message in preflight(agent, root)]
    except _ERRORS as e:
        return [("x", f"sandbox: {e}")]


def _freshness_row(agent: AgentConfig, workspace: Path | None) -> tuple[str, str]:
    """Re-render with the recorded arguments and diff against the files.

    A render is a function of the code, the configuration, and its
    arguments, so an exact re-render either reproduces the on-disk job or
    proves it stale. No stamp heuristics are involved.
    """
    from mason.sandbox import (
        read_render_record,
        render_sandbox_script,
        sandbox_toml,
        snapshot_engines,
    )
    from slab.config import load_config as load_slab_config

    project = Path.cwd()
    out_dir = project / "sandbox"
    record = read_render_record(out_dir)
    script_path = out_dir / "mason-sandbox.sbatch"
    if record is None or not script_path.is_file():
        return ("=", "rendered job: none here (render or launch writes sandbox/)")
    try:
        root = _ops.resolve_root(workspace)
        slab_cfg = load_slab_config(project)
        snapshots = snapshot_engines(slab_cfg)
        toml_path = out_dir / "slab.toml"
        toml_text, _warnings = sandbox_toml(slab_cfg, agent, root.resolve(), snapshots)
        engine_tasks = record.get("engine_tasks")
        entry_agent = record.get("agent")
        script, _binds, context = render_sandbox_script(
            agent,
            hpc=slab_cfg.hpc,
            slab_cfg=slab_cfg,
            workspace_root=root.resolve(),
            project=project,
            goal=str(record["goal"]),
            toml_path=toml_path,
            partition=record.get("partition"),
            time_limit=record.get("time_limit"),
            snapshots=snapshots,
            engine_tasks=int(str(engine_tasks)) if engine_tasks is not None else None,
            entry_agent=str(entry_agent) if entry_agent else None,
        )
    except _ERRORS as e:
        return ("x", f"rendered job: no longer renders ({e}) — stale; re-render")
    fresh = (
        script_path.read_text(encoding="utf-8") == script.rstrip("\n") + "\n"
        and toml_path.read_text(encoding="utf-8") == toml_text
        and (out_dir / "context.md").read_text(encoding="utf-8")
        == context.rstrip("\n") + "\n"
    )
    if fresh:
        return ("+", "rendered job: sandbox/ matches a fresh render")
    return (
        "x",
        "rendered job: sandbox/ differs from a fresh render (code or config "
        "moved since) — re-render, or 'slab mason sandbox launch'",
    )


# One-atom Cu through the same get_calculator path a campaign uses. A
# subprocess, not a thread: a worker stuck on missing weights must be
# killable at the timeout, and a thread is not.
_DEEP_PROBE = (
    "import sys\n"
    "from ase.build import bulk\n"
    "from slab.backends import get_calculator\n"
    "atoms = bulk('Cu', 'fcc', a=3.6)\n"
    "atoms.calc = get_calculator(sys.argv[1])\n"
    "print(f'{atoms.get_potential_energy():.6f}')\n"
)


def _deep_rows(checkpoint_ids: list[str], timeout_s: float) -> list[tuple[str, str]]:
    """One real single-point per declared checkpoint — the only probe that
    catches a checkpoint whose weights were never cached on this machine."""
    if not checkpoint_ids:
        return [("=", "deep: no declared checkpoints to probe")]
    import subprocess
    import sys

    rows: list[tuple[str, str]] = []
    for checkpoint in checkpoint_ids:
        try:
            result = subprocess.run(
                [sys.executable, "-c", _DEEP_PROBE, checkpoint],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            rows.append(("x", f"deep {checkpoint}: no answer within {timeout_s:.0f}s"))
            continue
        except OSError as e:
            rows.append(("x", f"deep {checkpoint}: {e}"))
            continue
        if result.returncode == 0:
            energy = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else "?"
            rows.append(("+", f"deep {checkpoint}: single-point answers ({energy} eV)"))
        else:
            detail = (result.stderr.strip().splitlines() or ["died with no stderr"])[-1]
            rows.append(("x", f"deep {checkpoint}: {detail}"))
    return rows


def run(
    workspace: Path | None,
    *,
    offline: bool,
    deep: bool,
    deep_timeout_s: float = 300.0,
    emit: Emit,
) -> int:
    """Emit every row; return the count of failing (``x``) rows."""
    rows, slab_cfg, agent = _config_rows()
    rows.append(_workspace_row(workspace))
    rows.append(_memory_row())
    engine_rows, checkpoint_ids = _engines_rows()
    rows.extend(engine_rows)
    rows.append(_hpc_row(slab_cfg))
    mp_rows, mp_root_path = _mp_rows(slab_cfg)
    rows.extend(mp_rows)
    rows.append(_gracemaker_row(slab_cfg))
    failures = 0
    for mark, message in rows:
        emit(f"[{mark}] {message}")
        failures += 1 if mark == "x" else 0
    if agent is not None:
        if offline:
            emit("[=] endpoint: skipped (--offline)")
        else:
            failures += _endpoint_rows(agent, workspace, _cluster(slab_cfg), emit)
        tail = [*_sandbox_rows(agent, workspace), _freshness_row(agent, workspace)]
        if deep:
            tail.extend(_deep_rows(checkpoint_ids, deep_timeout_s))
            tail.extend(_mp_deep_rows(mp_root_path))
        for mark, message in tail:
            emit(f"[{mark}] {message}")
            failures += 1 if mark in ("x", "-") else 0
    return failures


def _cluster(slab_cfg: SlabConfig | None) -> str:
    hpc = getattr(slab_cfg, "hpc", None)
    return (hpc.cluster or "") if hpc is not None else ""
