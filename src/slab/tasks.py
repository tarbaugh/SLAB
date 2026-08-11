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
    cache_extra=lambda arguments: describe_engine(arguments["engine"]),
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

    Caching boundary for cluster-served MLIPs: ``engine="rootstock"`` results
    are cached under the checkpoint id and options (traced inputs) plus the
    rootstock *client* version — the served environment's internals on the
    cluster (an env rebuild, in-place weight edits) are invisible to the
    client, whose contract with rootstock is that checkpoint ids are stable
    identities. Registry engines carry their full spec in the key instead,
    so any registry edit invalidates honestly.

    Args:
        atoms: Structure to relax (calculator-free).
        engine: One of :func:`slab.backends.available_engines`.
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
    described = describe_engine(engine)
    system = atoms.copy()
    calculator = get_calculator(engine, **(calculator_options or {}))
    system.calc = calculator

    try:
        with tempfile.TemporaryDirectory(prefix="slab-relax-") as tmp:
            trajectory = Path(tmp) / "relax.traj"
            optimizer = BFGS(system, trajectory=str(trajectory), logfile=None)
            converged = bool(optimizer.run(fmax=fmax, steps=steps))

            energy = float(system.get_potential_energy())
            forces = system.get_forces()
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


def _keep_unique(active: Any, name: str, path: Path) -> None:
    """Store *path* as an intermediate artifact, suffixing the name on collision."""
    stem, dot, suffix = name.rpartition(".")
    for attempt in range(1, 100):
        candidate = name if attempt == 1 else f"{stem}-{attempt}{dot}{suffix}"
        try:
            active.keep(candidate, path, role=ArtifactRole.INTERMEDIATE)
            return
        except ArtifactExistsError:
            continue
    raise ArtifactExistsError(active.id, name)  # pragma: no cover - 99 collisions
