"""Ready-made traced tasks. First (and for the MVP, only): structure relaxation.

These are ordinary ``@task`` functions — call them like functions, inside or
outside a run context. They demonstrate the pattern for wrapping any ASE
calculator as a SLAB task; no physics lives here.

Importing this module pulls in ASE (and numpy); the rest of ``slab`` stays
import-light, which is why these tasks are not re-exported from the package
root — use ``from slab.tasks import relax``.
"""

from __future__ import annotations

import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.optimize import BFGS

from slab.backends import close_calculator, describe_engine, get_calculator
from slab.errors import ArtifactExistsError
from slab.models import ArtifactRole
from slab.runtime import current_run
from slab.tracing import task


# cache_extra folds the resolved engine's identity (source + the registry's
# declared version) into the cache key: a maintainer bumping qe 7.3 -> 7.4 in
# the cluster registry honestly invalidates cached qe results.
@task(
    engines=("ase", "mace-torch", "rootstock"),
    cache_extra=lambda arguments: describe_engine(
        arguments["engine"], arguments.get("calculator_options")
    ),
)
def relax(
    atoms: Atoms,
    *,
    engine: str = "mace",
    fmax: float = 0.05,
    steps: int = 200,
    calculator_options: dict[str, Any] | None = None,
    label: str | None = None,
) -> tuple[Atoms, dict[str, Any]]:
    """Relax atomic positions with BFGS under the chosen engine.

    Positions only — no cell relaxation in the MVP. The input ``atoms`` is
    never mutated; pass it *without* an attached calculator (tracing hashes the
    input, and a live calculator does not serialize).

    Returns ``(relaxed, info)``: the relaxed structure carrying its final
    energy/forces on a ``SinglePointCalculator``, and an ``info`` dict with
    ``engine``, ``converged``, ``fmax`` (the achieved residual),
    ``fmax_target``, ``steps``, ``energy``, ``energy_unit`` (``"eV"``), and
    ``n_atoms`` — everything a ``@check`` needs.

    Inside a run context, the full optimization trajectory is additionally
    stored as an *intermediate* artifact named ``{label or 'relax'}.traj`` —
    inspectable while the run is alive, hash-and-discarded once retention
    tiers kick in.

    On failure, the evidence survives instead of vanishing with the scratch
    directory: the exception is re-raised carrying a diagnostic note
    (completed steps; the last trajectory frame's energy and residual force)
    and — inside a run — the partial trajectory is kept as
    ``{label or 'relax'}-failed.traj``. The tracer stores the note and a
    trimmed traceback on the failed task record
    (:func:`slab.errors.failure_record`), so an agent inspecting the run can
    decide a *specific* correction instead of retrying blind.

    Caching boundary for cluster-served MLIPs: ``engine="rootstock"`` results
    are cached under the checkpoint id and options (traced inputs) plus the
    rootstock *client* version — the served environment's internals on the
    cluster (an env rebuild, in-place weight edits) are invisible to the
    client, whose contract with rootstock is that checkpoint ids are stable
    identities. Registry engines carry their full spec in the key instead,
    so any registry edit invalidates honestly.

    Args:
        atoms: Structure to relax (calculator-free).
        engine: A built-in engine name, a cluster-registry name, or a
            rootstock checkpoint id served silently — e.g.
            ``engine="mace-mp-0-medium"`` with
            ``calculator_options={"cluster": "delta"}`` runs the MACE model
            from the cluster's rootstock install with no further ceremony.
        fmax: Convergence target — optimization stops when the largest
            per-atom force magnitude drops below this (eV/Å).
        steps: Maximum optimizer steps.
        calculator_options: Forwarded to the engine factory
            (e.g. ``{"model": "medium"}`` for mace).
        label: Names the trajectory artifact; auto-suffixed on collision.

    Examples:
        >>> from ase.build import bulk
        >>> atoms = bulk("Cu", "fcc", a=3.58) * (2, 1, 1)
        >>> atoms.rattle(stdev=0.03, seed=7)
        >>> relaxed, info = relax(atoms, engine="emt", fmax=0.05)
        >>> info["converged"], info["fmax"] < 0.05, info["energy_unit"]
        (True, True, 'eV')
    """
    # Engine identity is resolved BEFORE the (possibly hours-long) computation
    # and reused afterwards: a registry edited or deleted mid-run can neither
    # mis-stamp the provenance nor fail a completed optimization. The tracer's
    # cache_extra made the same resolution moments earlier for the cache key.
    described = describe_engine(engine, calculator_options)
    system = atoms.copy()
    calculator = get_calculator(engine, **(calculator_options or {}))
    system.calc = calculator

    try:
        with tempfile.TemporaryDirectory(prefix="slab-relax-") as tmp:
            trajectory = Path(tmp) / "relax.traj"
            optimizer = BFGS(system, trajectory=str(trajectory), logfile=None)
            try:
                converged = bool(optimizer.run(fmax=fmax, steps=steps))
                energy = float(system.get_potential_energy())
                forces = system.get_forces()
            except Exception as e:
                # The scratch directory is about to vanish — capture the
                # evidence first: keep the partial trajectory, note the
                # last-known state on the exception.
                _attach_failure_diagnostics(e, optimizer, trajectory, label)
                raise
            achieved_fmax = float(np.sqrt((forces**2).sum(axis=1).max()))

            active = current_run()
            if active is not None:
                _keep_unique(active, f"{label or 'relax'}.traj", trajectory)
    finally:
        # Worker-backed engines (rootstock) hold a subprocess; release it even
        # when the optimization raises.
        close_calculator(calculator)

    info: dict[str, Any] = {
        "engine": engine,
        "engine_source": described["source"],
        "engine_version": described.get("version"),
        "converged": converged,
        "fmax": achieved_fmax,
        "fmax_target": fmax,
        "steps": optimizer.get_number_of_steps(),
        "energy": energy,
        "energy_unit": "eV",
        "n_atoms": len(system),
    }

    # Detach the live calculator (it may hold an unpicklable torch model) but
    # keep the results on the returned structure.
    relaxed = system.copy()
    relaxed.calc = SinglePointCalculator(relaxed, energy=energy, forces=forces)
    return relaxed, info


def _attach_failure_diagnostics(
    e: Exception, optimizer: BFGS, trajectory: Path, label: str | None
) -> None:
    """Best-effort failure evidence: note the last-known state on *e* and keep
    the partial trajectory. Never raises — diagnostics must not mask the
    original failure.

    The note (surfaced by :func:`slab.errors.failure_record`) and the kept
    trajectory are what turn "it crashed" into a decidable situation: did the
    structure fly apart, did the energy oscillate, how far did it get.
    """
    try:
        note = f"relax failed after {optimizer.get_number_of_steps()} completed step(s)"
        last = _last_trajectory_frame(trajectory)
        if last is not None:
            note += (
                f"; trajectory has {last['frames']} frame(s), last frame: "
                f"E={last['energy']:.6f} eV, max|F|={last['fmax']:.4f} eV/Å"
            )
        active = current_run()
        if active is not None and trajectory.exists() and trajectory.stat().st_size > 0:
            kept_as = _keep_unique(active, f"{label or 'relax'}-failed.traj", trajectory)
            note += f"; partial trajectory kept as artifact {kept_as!r}"
        e.add_note(note)
    except Exception as diagnostics_error:
        with suppress(Exception):
            e.add_note(f"(relax failure diagnostics unavailable: {diagnostics_error})")


def _last_trajectory_frame(trajectory: Path) -> dict[str, Any] | None:
    """Stored energy/forces of the last frame BFGS wrote, or None if unreadable.

    Reads values recorded in the trajectory file only — it must never touch
    the live (just-failed) calculator.
    """
    from typing import cast

    from ase.io import read

    try:
        frames = cast("list[Atoms]", read(trajectory, index=":"))
        last = frames[-1]
        frame_forces = last.get_forces()
        return {
            "frames": len(frames),
            "energy": float(last.get_potential_energy()),
            "fmax": float(np.sqrt((frame_forces**2).sum(axis=1).max())),
        }
    except Exception:  # empty, truncated, or calculator-less trajectory
        return None


def _keep_unique(active: Any, name: str, path: Path) -> str:
    """Store *path* as an intermediate artifact, suffixing the name on collision.

    Returns the name actually used.
    """
    stem, dot, suffix = name.rpartition(".")
    for attempt in range(1, 100):
        candidate = name if attempt == 1 else f"{stem}-{attempt}{dot}{suffix}"
        try:
            active.keep(candidate, path, role=ArtifactRole.INTERMEDIATE)
            return candidate
        except ArtifactExistsError:
            continue
    raise ArtifactExistsError(active.id, name)  # pragma: no cover - 99 collisions
