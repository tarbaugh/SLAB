"""Ready-made traced tasks: structure relaxation and single-point evaluation.

These are ordinary ``@task`` functions — call them like functions, inside or
outside a run context. They demonstrate the pattern for wrapping any ASE
calculator as a SLAB task; no physics lives here. Together they cover the
canonical two-fidelity workflow: ``relax`` under a cheap engine, then
``single_point`` on the result under an expensive one.

Importing this module pulls in ASE (and numpy) through :mod:`slab.backends`;
both ``foundation`` and ``slab`` stay import-light, which is why these tasks
are not re-exported from either package root — use
``from foundation.tasks import relax``.
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

from foundation.errors import ArtifactExistsError
from foundation.models import ArtifactRole
from foundation.runtime import current_run
from foundation.tracing import task
from slab.backends import (
    close_calculator,
    collect_engine_outputs,
    collect_failure_evidence,
    describe_engine,
    get_calculator,
)


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
    tiers kick in. File-IO engines contribute their primary output the same
    way: with ``engine="qe"``, the final SCF's ``espresso.pwo`` is kept as
    ``{label or 'relax'}.pwo``; with ``engine="lammps"``, the final force
    evaluation's log (thermo table included) as ``{label or 'relax'}.log``.

    On failure, the evidence survives instead of vanishing with the scratch
    directories: the exception is re-raised carrying diagnostic notes
    (completed steps; the last trajectory frame's energy and residual force;
    for file-IO engines, the engine's own error report — ``pw.x``'s
    ``Error in routine ...`` message, LAMMPS's ``ERROR: ...`` log line) and —
    inside a run — the partial trajectory is kept as
    ``{label or 'relax'}-failed.traj``, alongside the engine's own files as
    ``{label or 'relax'}-failed.{pwi,pwo,crash}`` (qe) or
    ``{label or 'relax'}-failed.{in,log,data}`` (lammps). The tracer stores the
    notes and a trimmed traceback on the failed task record
    (:func:`foundation.errors.failure_record`), so an agent inspecting the run can
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
    if _qe_shaped(engine, calculator_options):
        _guard_qe_kpoints(atoms, calculator_options, task="relax")
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
                # The scratch directories are about to vanish — capture the
                # evidence first: keep the partial trajectory and the engine's
                # own files, note the last-known state on the exception.
                _attach_failure_diagnostics(e, optimizer, trajectory, label, calculator)
                raise
            achieved_fmax = float(np.sqrt((forces**2).sum(axis=1).max()))

            active = current_run()
            if active is not None:
                _keep_unique(active, f"{label or 'relax'}.traj", trajectory)
                # File-IO engines (qe) leave their primary output behind —
                # for pw.x the final SCF's espresso.pwo. Keep it as an
                # intermediate before close_calculator removes the scratch.
                for suffix, produced in collect_engine_outputs(calculator):
                    _keep_unique(active, f"{label or 'relax'}.{suffix}", produced)
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


@task(
    engines=("ase", "mace-torch", "rootstock"),
    cache_extra=lambda arguments: describe_engine(
        arguments["engine"], arguments.get("calculator_options")
    ),
)
def single_point(
    atoms: Atoms,
    *,
    engine: str = "mace",
    calculator_options: dict[str, Any] | None = None,
    label: str | None = None,
) -> tuple[Atoms, dict[str, Any]]:
    """Evaluate energy and forces once under the chosen engine — no optimization.

    The second half of the canonical two-fidelity workflow: relax a structure
    under a cheap engine (a universal MLIP, EMT), then ``single_point`` the
    relaxed geometry under the expensive one (``engine="qe"``) for a number
    worth believing. The input ``atoms`` is never mutated; pass it *without*
    a live calculator (the returned structure from :func:`relax` carries only
    a serializable ``SinglePointCalculator`` and is safe to pass directly).

    Returns ``(evaluated, info)``: the structure carrying its energy/forces on
    a ``SinglePointCalculator``, and an ``info`` dict with ``engine``,
    ``engine_source``, ``engine_version``, ``energy``, ``energy_unit``
    (``"eV"``), ``fmax`` (the largest per-atom force magnitude, eV/Å — the
    natural check that the cheap relaxation was good enough), and ``n_atoms``.
    There is deliberately no ``converged`` key: nothing was optimized, and an
    engine whose own self-consistency fails raises instead of returning.

    Inside a run context, file-IO engines contribute their primary output as
    an *intermediate* artifact: with ``engine="qe"``, the SCF's
    ``espresso.pwo`` is kept as ``{label or 'single-point'}.pwo``; with
    ``engine="lammps"``, the evaluation's log as
    ``{label or 'single-point'}.log``. On failure the engine's own error
    report is attached to the exception as notes and its files are kept as
    ``{label or 'single-point'}-failed.*`` — the same evidence contract as
    :func:`relax`.

    Args:
        atoms: Structure to evaluate (calculator-free).
        engine: A built-in engine name, a cluster-registry name, or a
            rootstock checkpoint id — same resolution as :func:`relax`.
        calculator_options: Forwarded to the engine factory (for ``qe``,
            e.g. ``qe_protocol_options(atoms, protocol="balanced")`` plus
            ``pseudo_family=``).
        label: Names the kept engine-output artifacts.

    Examples:
        >>> from ase.build import bulk
        >>> evaluated, info = single_point(bulk("Cu", "fcc", a=3.58), engine="emt")
        >>> info["energy_unit"], info["n_atoms"], "converged" in info
        ('eV', 1, False)
    """
    resolved_options = calculator_options
    if _qe_shaped(engine, calculator_options):
        _guard_qe_kpoints(atoms, calculator_options, task="single_point")
        resolved_options = _qe_scf_options(calculator_options)
    # Same ordering rule as relax: engine identity resolves BEFORE the
    # computation, so a registry edited mid-run cannot mis-stamp provenance.
    # Identity (and the tracer's cache_extra) see the caller's options; only
    # the factory sees the scf-pinned copy.
    described = describe_engine(engine, calculator_options)
    system = atoms.copy()
    calculator = get_calculator(engine, **(resolved_options or {}))
    system.calc = calculator

    try:
        try:
            # Forces first: ASE then requests energy+forces in one engine
            # execution (for qe, one pw.x run with forces in the input);
            # asking for energy alone first would trigger a second run.
            forces = system.get_forces()
            energy = float(system.get_potential_energy())
        except Exception as e:
            _attach_engine_evidence(e, calculator, label or "single-point")
            raise
        active = current_run()
        if active is not None:
            for suffix, produced in collect_engine_outputs(calculator):
                _keep_unique(active, f"{label or 'single-point'}.{suffix}", produced)
    finally:
        close_calculator(calculator)

    info: dict[str, Any] = {
        "engine": engine,
        "engine_source": described["source"],
        "engine_version": described.get("version"),
        "energy": energy,
        "energy_unit": "eV",
        "fmax": float(np.sqrt((forces**2).sum(axis=1).max())),
        "n_atoms": len(system),
    }

    evaluated = system.copy()
    evaluated.calc = SinglePointCalculator(evaluated, energy=energy, forces=forces)
    return evaluated, info


def _qe_shaped(engine: str, options: dict[str, Any] | None) -> bool:
    """True for the built-in ``qe`` engine AND registry aliases built on it.

    The qe guards are engine-semantic, not name-semantic: an alias declared
    as ``calculator = "slab.backends.qe_calculator"`` runs the same pw.x and
    earns the same refusals (k-points, single_point's scf pin) — keying them
    on the literal name "qe" would make the alias the silent-wrong-answer
    route around them.
    """
    if engine.strip().lower() == "qe":
        return True
    return describe_engine(engine, options).get("calculator") == "slab.backends.qe_calculator"


def _guard_qe_kpoints(atoms: Atoms, options: dict[str, Any] | None, *, task: str) -> None:
    """Refuse a fully periodic ``qe`` run that declares no k-points at all.

    With ``kpts`` absent ASE writes ``K_POINTS gamma``, and a Γ-only SCF on
    a bulk cell converges, prints plausible numbers, and is physically
    meaningless — the silent-wrong-answer mode. An explicit ``kpts=None`` is
    the opt-in for genuine Γ-only runs (molecules in boxes).
    """
    opts = options or {}
    if not bool(atoms.pbc.all()) or "kpts" in opts or "kspacing" in opts:
        return
    raise ValueError(
        f"{task} with engine='qe' on a fully periodic cell declares no "
        f"k-points: ASE would write a Γ-only SCF that converges and looks "
        f"plausible while sampling a single point of the Brillouin zone. "
        f"Expand a protocol (qe_protocol_options derives a mesh from the "
        f"cell) or pass kpts= explicitly — kpts=None opts in to Γ-only"
    )


def _qe_scf_options(options: dict[str, Any] | None) -> dict[str, Any]:
    """``calculator_options`` with ``calculation`` pinned to ``'scf'``, loudly.

    A recycled dict carrying ``calculation='relax'``/``'vc-relax'`` would
    make a task named single_point silently run pw.x's *internal* optimizer
    and attribute the final geometry's energy and near-zero forces to the
    input structure. A conflicting value is refused, absence is pinned; the
    caller's dict is a traced input and is never mutated.
    """
    from copy import deepcopy

    resolved: dict[str, Any] = deepcopy(dict(options or {}))
    input_data = dict(resolved.get("input_data") or {})
    control = input_data.get("control")
    declared = input_data.get("calculation")
    if isinstance(control, dict) and "calculation" in control:
        declared = control["calculation"]
    if declared is not None and str(declared) != "scf":
        raise ValueError(
            f"single_point runs exactly one SCF, but calculator_options "
            f"declares calculation={str(declared)!r} — drop the key, or use "
            f"relax for optimization"
        )
    if isinstance(control, dict):
        input_data["control"] = {**control, "calculation": "scf"}
    else:
        input_data["calculation"] = "scf"
    resolved["input_data"] = input_data
    return resolved


def _attach_failure_diagnostics(
    e: Exception, optimizer: BFGS, trajectory: Path, label: str | None, calculator: Any
) -> None:
    """Best-effort failure evidence: note the last-known state on *e*, keep
    the partial trajectory, and fold in whatever the engine itself left
    behind. Never raises — diagnostics must not mask the original failure.

    The notes (surfaced by :func:`foundation.errors.failure_record`) and the kept
    files are what turn "it crashed" into a decidable situation: did the
    structure fly apart, did the SCF diverge, what did ``pw.x`` actually say.
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
    _attach_engine_evidence(e, calculator, label)


def _attach_engine_evidence(e: Exception, calculator: Any, label: str | None) -> None:
    """File-IO engines tell their failure story in files (``espresso.pwo``,
    ``CRASH``), not in the exception — which is a bare ``CalledProcessError``.
    Note the story on *e* and, inside a run, keep the files before the
    engine's scratch directory vanishes. Never raises.
    """
    try:
        notes, evidence_files = collect_failure_evidence(calculator)
        # Parsed evidence attaches before any storage happens: a failing
        # artifact store (disk full) must not cost the already-extracted
        # engine error message.
        for note in notes:
            e.add_note(note)
        active = current_run()
        if active is not None and evidence_files:
            kept = [
                _keep_unique(active, f"{label or 'relax'}-failed.{suffix}", path)
                for suffix, path in evidence_files
            ]
            e.add_note("engine files kept as artifacts: " + ", ".join(repr(k) for k in kept))
    except Exception as diagnostics_error:
        with suppress(Exception):
            e.add_note(f"(engine failure evidence unavailable: {diagnostics_error})")


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
