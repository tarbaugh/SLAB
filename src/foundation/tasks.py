"""Ready-made traced tasks: structures, relaxation, single points, MLIP training.

These are ordinary ``@task`` functions — call them like functions, inside or
outside a run context. They demonstrate the pattern for wrapping any ASE
calculator as a SLAB task; no physics lives here. Together they cover the
canonical workflow: ``build_structure`` makes the geometry (atomsk), ``relax``
optimizes it under a cheap engine, then ``single_point`` evaluates the result
under an expensive one. The training pair extends it:
``collect_training_data`` assembles the labels those tasks recorded, and
``train_potential`` fits a GRACE potential with gracemaker.

Importing this module pulls in ASE (and numpy) through :mod:`slab.backends`;
both ``foundation`` and ``slab`` stay import-light, which is why these tasks
are not re-exported from either package root — use
``from foundation.tasks import relax``.
"""

from __future__ import annotations

import gzip
import hashlib
import re
import shlex
import shutil
import tarfile
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.filters import FrechetCellFilter
from ase.io import read as ase_read
from ase.io import write as ase_write
from ase.optimize import BFGS, CellAwareBFGS

from foundation.errors import ArtifactExistsError, FoundationError
from foundation.lifecycle import ExecutionStatus
from foundation.models import ArtifactRole
from foundation.runtime import current_run
from foundation.serialize import loads as foundation_loads
from foundation.tracing import task
from slab.atomsk import build_scratch_dir, describe_atomsk, run_atomsk
from slab.backends import (
    close_calculator,
    collect_engine_outputs,
    collect_failure_evidence,
    describe_engine,
    get_calculator,
)
from slab.errors import BuilderError
from slab.gracemaker import describe_gracemaker, run_gracemaker, train_scratch_dir
from slab.mp import describe_mp, mp_root, structure_path


# cache_extra folds the resolved engine's identity (source + the registry's
# declared version) into the cache key: a maintainer bumping qe 7.3 -> 7.4 in
# the cluster registry honestly invalidates cached qe results.
@task(
    engines=("ase", "rootstock"),
    cache_extra=lambda arguments: describe_engine(
        arguments["engine"], arguments.get("calculator_options")
    ),
)
def relax(
    atoms: Atoms,
    *,
    engine: str,
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
            (e.g. ``{"cluster": "delta"}`` for a rootstock checkpoint id).
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


#: Voigt masks for :func:`relax_cell`'s ``symmetry`` argument. Six-tuple in
#: the order ASE's cell filters expect (xx, yy, zz, yz, xz, xy). ``isotropic``
#: uses a full mask paired with the filter's ``hydrostatic_strain=True`` flag,
#: which ties the three normal components to one volumetric degree of freedom.
_CELL_MASKS: dict[str, tuple[bool, bool, bool, bool, bool, bool]] = {
    "isotropic": (True, True, True, False, False, False),
    "orthorhombic": (True, True, True, False, False, False),
    "triclinic": (True, True, True, True, True, True),
}


@task(
    engines=("ase", "rootstock"),
    cache_extra=lambda arguments: describe_engine(
        arguments["engine"], arguments.get("calculator_options")
    ),
)
def relax_cell(
    atoms: Atoms,
    *,
    engine: str,
    symmetry: Literal["isotropic", "orthorhombic", "triclinic"] = "triclinic",
    fmax: float = 0.05,
    smax: float = 0.005,
    steps: int = 200,
    calculator_options: dict[str, Any] | None = None,
    label: str | None = None,
) -> tuple[Atoms, dict[str, Any]]:
    """Relax positions AND cell to zero force and near-zero stress.

    The companion to :func:`relax` for tasks whose *reference* must be the
    engine's own equilibrium cell — the elastic-constants and surface-energy
    skills both name this as their prerequisite. Uses
    :class:`ase.optimize.CellAwareBFGS` on a
    :class:`ase.filters.FrechetCellFilter`-wrapped structure.

    Args:
        atoms: Structure to relax (calculator-free). Never mutated.
        engine: A built-in name, a cluster-registry name, or a rootstock
            checkpoint id — same resolution as :func:`relax`. The engine must
            produce a stress tensor (rootstock-served MLIPs do; ``qe``'s
            ASE calculator does, but for ``qe`` the honest route is pw.x's
            own ``calculation='vc-relax'`` inside :func:`single_point` —
            this task would relax the cell against a fixed k-mesh, which
            gives a k-mesh-dependent equilibrium).
        symmetry: Cell degrees of freedom.
            ``"isotropic"`` allows only volumetric scaling (one dof, cubic
            or fully-constrained shapes); ``"orthorhombic"`` allows the
            three axis lengths to vary independently but keeps angles at
            their input (fits Pnma, Cmcm, ...); ``"triclinic"`` (the
            default) allows all six strain components.
        fmax: Force convergence target (eV/Å) — the same meaning as in
            :func:`relax`, applied to atomic positions.
        smax: Stress convergence target (eV/Å³) — the largest allowed
            magnitude of any (masked) stress component. 0.005 eV/Å³ ≈
            0.8 GPa; tighten it (e.g. 0.001) when the run feeds an elastic
            or surface-energy calculation that needs zero residual stress.
        steps: Maximum optimizer steps.
        calculator_options: Forwarded to the engine factory.
        label: Names the trajectory artifact; auto-suffixed on collision.

    Returns ``(relaxed, info)`` where ``info`` mirrors :func:`relax` and
    additionally carries ``smax`` (the achieved residual), ``smax_target``,
    ``symmetry``, ``cell_lengths``, ``cell_angles``, and ``volume``.

    Failure evidence follows :func:`relax`'s contract: partial trajectory
    kept as ``{label or 'relax_cell'}-failed.traj``, engine files kept with
    the same prefix, and diagnostic notes on the exception.

    Examples:
        >>> from ase.build import bulk
        >>> atoms = bulk("Cu", "fcc", a=3.4)  # compressed vs EMT equilibrium
        >>> relaxed, info = relax_cell(atoms, engine="emt", symmetry="isotropic")
        >>> info["converged"], info["smax"] < 0.005, info["symmetry"]
        (True, True, 'isotropic')
    """
    if _qe_shaped(engine, calculator_options):
        _guard_qe_kpoints(atoms, calculator_options, task="relax_cell")
    if symmetry not in _CELL_MASKS:
        raise ValueError(
            f"unknown symmetry {symmetry!r} — expected one of "
            f"{sorted(_CELL_MASKS)}"
        )
    described = describe_engine(engine, calculator_options)
    system = atoms.copy()
    calculator = get_calculator(engine, **(calculator_options or {}))
    system.calc = calculator

    mask = list(_CELL_MASKS[symmetry])
    hydrostatic = symmetry == "isotropic"

    try:
        with tempfile.TemporaryDirectory(prefix="slab-relax-cell-") as tmp:
            trajectory = Path(tmp) / "relax_cell.traj"
            # exp_cell_factor defaults to len(atoms) since ASE 3.29, and
            # CellAwareBFGS asserts it is exactly 1.0 — any multi-atom cell
            # crashes without the explicit pin.
            filt = FrechetCellFilter(
                system, mask=mask, hydrostatic_strain=hydrostatic, exp_cell_factor=1.0
            )
            # ASE's CellAwareBFGS accepts a filter-wrapped Atoms and a None
            # logfile in practice, but its type stub narrows both. Ignoring
            # here keeps the idiomatic call; everything else stays typed.
            optimizer = CellAwareBFGS(filt, trajectory=str(trajectory), logfile=None)  # type: ignore[arg-type]
            try:
                converged = bool(optimizer.run(fmax=fmax, smax=smax, steps=steps))
                energy = float(system.get_potential_energy())
                forces = system.get_forces()
                stress = system.get_stress()  # voigt: xx, yy, zz, yz, xz, xy
            except Exception as e:
                _attach_failure_diagnostics(
                    e, optimizer, trajectory, label, calculator, task_name="relax_cell"
                )
                raise
            achieved_fmax = float(np.sqrt((forces**2).sum(axis=1).max()))
            # Only the masked stress components are relaxed toward zero; the
            # frozen ones (off-diagonal in "orthorhombic", every non-volumetric
            # component in "isotropic") stay at whatever the equilibrium leaves.
            masked_stress = np.array(stress)[np.array(mask, dtype=bool)]
            achieved_smax = float(np.abs(masked_stress).max()) if masked_stress.size else 0.0

            active = current_run()
            if active is not None:
                _keep_unique(active, f"{label or 'relax_cell'}.traj", trajectory)
                for suffix, produced in collect_engine_outputs(calculator):
                    _keep_unique(active, f"{label or 'relax_cell'}.{suffix}", produced)
    finally:
        close_calculator(calculator)

    cell = system.cell.cellpar()
    info: dict[str, Any] = {
        "engine": engine,
        "engine_source": described["source"],
        "engine_version": described.get("version"),
        "converged": converged,
        "fmax": achieved_fmax,
        "fmax_target": fmax,
        "smax": achieved_smax,
        "smax_target": smax,
        "steps": optimizer.get_number_of_steps(),
        "energy": energy,
        "energy_unit": "eV",
        "n_atoms": len(system),
        "symmetry": symmetry,
        "cell_lengths": (float(cell[0]), float(cell[1]), float(cell[2])),
        "cell_angles": (float(cell[3]), float(cell[4]), float(cell[5])),
        "volume": float(system.get_volume()),
    }

    relaxed = system.copy()
    relaxed.calc = SinglePointCalculator(relaxed, energy=energy, forces=forces)
    return relaxed, info


@task(
    engines=("ase", "rootstock"),
    cache_extra=lambda arguments: describe_engine(
        arguments["engine"], arguments.get("calculator_options")
    ),
)
def single_point(
    atoms: Atoms,
    *,
    engine: str,
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


# cache_extra folds the atomsk install's identity (resolved command, detected
# version) into the cache key: pointing at a different binary, or upgrading
# it, honestly invalidates cached structures.
@task(
    engines=("ase",),
    cache_extra=lambda arguments: describe_atomsk(command=arguments.get("command")),
)
def build_structure(
    args: str | list[str] | tuple[str, ...],
    *,
    inputs: dict[str, Atoms | str] | None = None,
    output: str | None = None,
    command: str | None = None,
    label: str | None = None,
    timeout_s: float = 600.0,
) -> tuple[Atoms, dict[str, Any]]:
    """Build or transform a structure with atomsk, traced.

    Atomsk is a *builder*, not an engine: it creates and transforms
    structures (unit cells, supercells, defects, interfaces, polycrystals)
    and computes no energies. This task runs one atomsk invocation in a
    private scratch directory, reads the structure it wrote back through
    ASE, and returns it ready for :func:`relax` or :func:`single_point`.
    Configure the install under ``[builders.atomsk]`` in ``slab.toml``.

    *args* is atomsk's own argument list — mode, options, and bare file
    names — without the leading ``atomsk``; a single string is split with
    shell rules. Every file name must be bare: the invocation runs in a
    fresh scratch directory, and an argument that names a path elsewhere is
    refused, because a traced build reading undeclared files records a
    cache identity that lies. Files the invocation reads enter through
    ``inputs``, a mapping of bare file name to value: an ``Atoms`` is
    written as a structure file (format from its extension — ``.xsf``
    round-trips cell and species reliably), and a plain string is written
    verbatim (a ``--polycrystal`` parameter file). Every input is traced.

    ``output`` names the produced file to read back. Leave it None when the
    invocation produces exactly one new file; with several (extra output
    formats, polycrystal statistics files), name the one that is the
    result. Inside a run context every produced file is kept as an
    intermediate artifact (``{label or 'build'}.{suffix}``), and the full
    atomsk log as ``{label or 'build'}.log``. On failure the extracted
    ``X!X ERROR`` lines ride on the exception as notes and the log is kept
    as ``{label or 'build'}-failed.log`` — the same evidence contract as
    the engine tasks.

    Returns ``(atoms, info)``: the structure, and an ``info`` dict with
    ``builder`` (``"atomsk"``), ``version``, ``command``, ``args``,
    ``output``, ``produced`` (every new file name), ``n_atoms``,
    ``formula``, and ``pbc``.

    Example::

        supercell, info = build_structure(
            "--create fcc 4.046 Al -duplicate 4 4 4 al.xsf", label="al-444"
        )
        relaxed, opt = relax(supercell, engine="mace-mp-0-medium")

    Args:
        args: Atomsk's argument list (or one string, shell-split).
        inputs: Files to stage into the scratch directory, by bare file
            name: ``Atoms`` values become structure files, string values
            are written verbatim (parameter files).
        output: The produced file holding the result; required only when
            the invocation produces more than one new file.
        command: Overrides the resolved atomsk command for this call
            (else ``[builders.atomsk] command``, else ``atomsk``).
        label: Names the kept artifacts.
        timeout_s: Kill the invocation after this long.
    """
    argv = shlex.split(args) if isinstance(args, str) else [str(token) for token in args]
    # Builder identity resolves BEFORE the invocation, mirroring the engine
    # tasks: the tracer's cache_extra made the same resolution moments ago.
    described = describe_atomsk(command=command)
    name = label or "build"
    scratch = build_scratch_dir()
    try:
        for input_name, staged_value in (inputs or {}).items():
            _guard_staged_name(input_name)
            if isinstance(staged_value, str):
                (scratch / input_name).write_text(staged_value, encoding="utf-8")
            else:
                ase_write(scratch / input_name, staged_value)
        staged = {entry.name for entry in scratch.iterdir()}
        try:
            outcome = run_atomsk(argv, cwd=scratch, command=command, timeout_s=timeout_s)
        except BuilderError as e:
            active = current_run()
            if active is not None and e.log:
                log_path = scratch / "slab-atomsk.log"
                log_path.write_text(e.log, encoding="utf-8")
                kept = _keep_unique(active, f"{name}-failed.log", log_path)
                e.add_note(f"full atomsk log kept as artifact {kept!r}")
            raise
        produced = sorted(
            entry.name for entry in scratch.iterdir() if entry.name not in staged
        )
        result_name = _pick_output(output, produced)
        try:
            structure = _read_built(scratch / result_name)
        except Exception as e:
            e.add_note(
                f"atomsk wrote {result_name!r} but ASE cannot read it back; "
                "write an ASE-readable format too (xsf) and name it with "
                "output="
            )
            raise
        if not isinstance(structure, Atoms):  # a multi-frame file
            structure = structure[-1]
        active = current_run()
        if active is not None:
            log_path = scratch / "slab-atomsk.log"
            log_path.write_text(outcome.log, encoding="utf-8")
            _keep_unique(active, f"{name}.log", log_path)
            for produced_name in produced:
                suffix = Path(produced_name).suffix.lstrip(".").lower() or produced_name
                _keep_unique(active, f"{name}.{suffix}", scratch / produced_name)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    info: dict[str, Any] = {
        "builder": "atomsk",
        "version": described.get("version"),
        "command": described["command"],
        "args": argv,
        "output": result_name,
        "produced": produced,
        "n_atoms": len(structure),
        "formula": structure.get_chemical_formula(),
        "pbc": [bool(flag) for flag in structure.pbc],
    }
    return structure, info


# cache_extra folds the snapshot's identity (release, material count) into the
# cache key: installing a newer snapshot honestly invalidates fetched
# structures, while the same release mounted at a different path still hits.
@task(engines=("ase",), cache_extra=lambda arguments: describe_mp())
def fetch_structure(
    material_id: str, *, label: str | None = None
) -> tuple[Atoms, dict[str, Any]]:
    """Fetch one structure from the offline Materials Project snapshot, traced.

    The snapshot (``[builders.mp] root`` in ``slab.toml``) is a read-only
    local data source, a *builder* like atomsk: it supplies structures,
    never energies. This task resolves the material's archived CIF below
    the snapshot root, reads it through ASE, keeps the CIF with the run,
    and returns the structure ready for :func:`relax` or
    :func:`single_point`.

    Two rules from the snapshot's contract. A result's identity is the
    pair ``(release, material_id)`` — report both, never a formula alone,
    because entries share compositions and releases revise records. And
    absence is absence: a material id the snapshot does not hold raises
    with that statement, and nothing here falls back to an online lookup.

    Search before fetching: reduce candidates against ``metadata.sqlite``
    (``slab.mp.search_materials``, the ``slab mp search`` command, or the
    agent's search tools), then fetch the shortlisted ids one by one.

    Returns ``(atoms, info)``: the structure, and an ``info`` dict with
    ``builder`` (``"mp"``), ``material_id``, ``release``, ``cif_path``
    (relative to the snapshot root), ``source`` (``"cif"``), ``n_atoms``,
    ``formula``, and ``pbc``.

    Example::

        atoms, info = fetch_structure("mp-149")
        relaxed, opt = relax(atoms, engine="mace-mp-0-medium")

    Args:
        material_id: One Materials Project id, e.g. ``"mp-149"``.
        label: Names the kept CIF artifact (default: the material id).
    """
    # Snapshot identity resolves BEFORE the read, mirroring the engine
    # tasks: the tracer's cache_extra made the same resolution moments ago.
    described = describe_mp()
    cif = structure_path(material_id)
    structure = ase_read(cif)
    if not isinstance(structure, Atoms):  # a multi-frame file
        structure = structure[-1]
    active = current_run()
    if active is not None:
        _keep_unique(active, f"{label or material_id}.cif", cif)
    info: dict[str, Any] = {
        "builder": "mp",
        "material_id": material_id,
        "release": described.get("release"),
        "cif_path": str(cif.relative_to(mp_root())),
        "source": "cif",
        "n_atoms": len(structure),
        "formula": structure.get_chemical_formula(),
        "pbc": [bool(flag) for flag in structure.pbc],
    }
    return structure, info


# cache_extra resolves every source to its content hash: a new task in a
# source run honestly misses, while identical content collected again hits.
@task(
    engines=("ase",),
    cache_extra=lambda arguments: {
        "sources": [
            [s["run_id"], s["ref"], s["hash"]]
            for s in _training_sources(
                arguments["run_ids"],
                engine=arguments.get("engine"),
                frames=arguments.get("frames", "final"),
                allow_mixed=arguments.get("allow_mixed", False),
            )
        ]
    },
)
def collect_training_data(
    run_ids: Sequence[str],
    *,
    engine: str | None = None,
    frames: Literal["final", "all"] = "final",
    allow_mixed: bool = False,
    output: str = "training.extxyz",
    label: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Assemble a labeled training dataset from completed runs, traced.

    Reads the recorded results of completed ``relax``, ``relax_cell``, and
    ``single_point`` tasks in the named runs — the exact energies and
    forces those engines computed, engine-agnostic and parse-free — and
    writes one extended-XYZ file ready for :func:`train_potential`.
    ``frames="all"`` additionally includes every labeled frame of the kept
    relaxation trajectories (cheap labels along the optimization path, the
    standard dataset-bootstrapping move; the endpoint frame then appears
    twice, once from the trajectory and once as the task result).

    Labels from different engines in one dataset are this task's
    silent-wrong-answer mode, so mixed sources are refused: pass
    ``engine=`` to select one, or ``allow_mixed=True`` to state that
    mixing is intended. Every other gap is refused loudly too — a run
    contributing no labeled structure, or source bytes the retention
    policy has discarded — never silently thinned.

    The written file lands in the working directory on a cold execution
    only; on a cache hit the recorded dataset artifact is the result.
    ``info["dataset_hash"]`` is the authoritative handle either way —
    re-materialize with ``slab show`` and the artifact store when the file
    is gone.

    Returns ``(path, info)``: the dataset path, and an ``info`` dict with
    ``n_structures``, ``n_duplicates`` (identical task results deduplicated
    across runs), ``n_unlabeled_frames`` (trajectory frames without
    energy+forces, skipped), ``elements``, ``engines``, per-run ``sources``
    counts, ``free_energy_structures`` (sources carrying the
    force-consistent energy, which the extxyz keeps alongside ``energy``),
    ``output``, and ``dataset_hash``.

    Args:
        run_ids: Runs to collect from (full ids or unique prefixes).
        engine: Keep only labels computed by this engine.
        frames: ``"final"`` (task results only) or ``"all"`` (also every
            labeled trajectory frame).
        allow_mixed: Accept labels from more than one engine.
        output: Path of the extended-XYZ file to write.
        label: Names the kept dataset artifact (default: the output name).
    """
    sources = _training_sources(
        run_ids, engine=engine, frames=frames, allow_mixed=allow_mixed
    )
    structures: list[Atoms] = []
    seen_hashes: set[str] = set()
    per_run: dict[str, int] = {}
    engines_used: set[str] = set()
    n_duplicates = 0
    n_unlabeled = 0
    n_free_energy = 0
    with _training_stores() as (_runs, artifacts):
        for source in sources:
            stored = artifacts.get(source["hash"])
            if source["kind"] == "task":
                if source["hash"] in seen_hashes:
                    n_duplicates += 1
                    continue
                seen_hashes.add(source["hash"])
                value = foundation_loads(stored.read_bytes())
                frames_in = [value] if isinstance(value, Atoms) else []
            else:
                read = ase_read(stored, index=":", format="traj")
                frames_in = list(read) if isinstance(read, list) else [read]
            for frame in frames_in:
                calc = getattr(frame, "calc", None)
                results = getattr(calc, "results", {})
                if "energy" not in results or "forces" not in results:
                    n_unlabeled += 1
                    continue
                if "free_energy" in results:
                    n_free_energy += 1
                structures.append(frame)
                per_run[source["run_id"]] = per_run.get(source["run_id"], 0) + 1
            if source["engine"]:
                engines_used.add(source["engine"])
    if not structures:
        raise FoundationError(
            "no labeled structures survived collection: every candidate "
            "frame lacked energy+forces"
        )
    out_path = Path(output)
    ase_write(out_path, structures, format="extxyz")
    dataset_hash: str | None = None
    active = current_run()
    if active is not None:
        kept = _keep_unique(active, label or out_path.name, out_path)
        dataset_hash = active.runs.get_artifact(active.id, kept).hash
    elements = sorted({symbol for atoms in structures for symbol in atoms.symbols})
    info: dict[str, Any] = {
        "n_structures": len(structures),
        "n_duplicates": n_duplicates,
        "n_unlabeled_frames": n_unlabeled,
        "elements": elements,
        "engines": sorted(engines_used),
        "sources": per_run,
        "free_energy_structures": n_free_energy,
        "output": str(out_path),
        "dataset_hash": dataset_hash,
    }
    return str(out_path), info


# cache_extra folds the trainer's identity AND the dataset file's content
# hash into the cache key: the serializer would otherwise hash the dataset
# by its path string, and changed bytes at the same path must miss.
@task(
    engines=("ase",),
    cache_extra=lambda arguments: {
        **describe_gracemaker(command=arguments.get("command")),
        **(
            {"dataset_sha256": _file_sha256(arguments["dataset"])}
            if arguments.get("dataset") is not None
            else {}
        ),
    },
)
def train_potential(
    input_yaml: str,
    *,
    dataset: str | None = None,
    label: str | None = None,
    export_fs: bool = False,
    command: str | None = None,
    timeout_s: float = 86400.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train a GRACE machine-learned potential with gracemaker, traced.

    Gracemaker (``[builders.gracemaker]`` in ``slab.toml``) is the only
    MLIP-training route SLAB has. It runs as a subprocess in its own
    python environment, in slab-managed scratch — on a cluster, submit
    the workflow that calls this task to a GPU partition through the
    scheduler; do not run a real fit on a login node.

    *input_yaml* is the **text** of gracemaker's own input file, verbatim
    — write the YAML yourself (sections ``cutoff``, ``seed``, ``data``,
    ``potential``, ``fit``, ``backend``) and pass the text, never a path:
    the text is what enters the cache identity. The *dataset* file is
    staged next to it, and its ``data: filename:`` entry must reference
    the dataset by bare basename. (That reference is checked textually,
    not by parsing the YAML, so a commented-out ``filename:`` line can
    satisfy it — the fit then fails loudly inside gracemaker.) With
    ``dataset=None`` the YAML must name data by a path readable from the
    compute node; that file is then outside the cache identity, like a
    foundation-model preset named under ``potential:`` (preset names are
    treated as stable identities, the checkpoint-id rule).

    The fit's evidence is kept with the run: the training log, the model
    architecture, the final train/test metrics, and the exported
    ``saved_model`` as one tar.gz (plus ``FS_model.yaml`` when
    ``export_fs=True``). The exports are also copied into a
    ``{label}/`` directory in the working directory — on a cold
    execution only. On a cache hit nothing is re-copied: the artifact
    hashes in the returned record are the authoritative handles;
    re-materialize from them with ``slab show`` and the artifact store.
    A converged-but-bad fit is not a failure here — judge the returned
    metrics with a ``@check``.

    Returns ``(model, info)``: *model* holds the handles (``seed_dir``,
    ``output_dir``, ``saved_model``, ``fs_model``, artifact name→hash
    map); *info* holds the trainer identity and the final
    ``train_metrics``/``test_metrics``.

    Args:
        input_yaml: The full input.yaml text, agent-authored.
        dataset: Path to the training data (extxyz or ``.pkl.gz``) to
            stage beside the input file.
        label: Names the kept artifacts and the output directory
            (default ``potential``).
        export_fs: Also export the GRACE/FS ``FS_model.yaml`` (only
            meaningful for FS-preset fits).
        command: Override the configured gracemaker command.
        timeout_s: Hard kill for one gracemaker invocation (default 24 h;
            the batch job's own time limit is the outer guard).
    """
    name = label or "potential"
    described = describe_gracemaker(command=command)
    if "\n" not in input_yaml and Path(input_yaml).exists():
        raise BuilderError(
            f"input_yaml looks like a path ({input_yaml!r}); pass the YAML "
            "text itself — the text is what enters the cache identity"
        )
    dataset_path: Path | None = None
    if dataset is not None:
        dataset_path = Path(dataset)
        if not dataset_path.is_file():
            raise BuilderError(f"dataset {dataset!r} does not exist or is not a file")
        if dataset_path.name not in input_yaml:
            raise BuilderError(
                f"the input.yaml text never mentions the dataset's basename "
                f"{dataset_path.name!r}; reference it under data: filename: "
                "(the dataset is staged beside input.yaml under that name)"
            )
    out_dir = Path(name)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise BuilderError(
            f"output directory {out_dir} exists and is not empty; move it "
            "aside or pass a different label= (checked before the fit so a "
            "day of training is not spent to hit this)"
        )
    active = current_run()
    scratch = train_scratch_dir()
    try:
        (scratch / "input.yaml").write_text(input_yaml, encoding="utf-8")
        if dataset_path is not None:
            shutil.copy2(dataset_path, scratch / dataset_path.name)
        try:
            run_gracemaker(
                ["input.yaml"], cwd=scratch, command=command, timeout_s=timeout_s
            )
        except BuilderError as e:
            if active is not None:
                _keep_training_failure(active, scratch, name, e)
            raise
        seed_dir = _the_seed_dir(scratch)
        artifact_hashes: dict[str, str] = {}
        if active is not None:
            for fname, kept_as in (
                ("log.txt", f"{name}.log"),
                ("model.yaml", f"{name}-model.yaml"),
                ("train_metrics.yaml", f"{name}-train-metrics.yaml"),
                ("test_metrics.yaml", f"{name}-test-metrics.yaml"),
            ):
                source = seed_dir / fname
                if source.is_file():
                    kept = _keep_unique(active, kept_as, source)
                    artifact_hashes[kept] = active.runs.get_artifact(active.id, kept).hash
        export_args = ["-r", "-s", *(["-sf"] if export_fs else []), "input.yaml"]
        try:
            run_gracemaker(
                export_args, cwd=scratch, command=command, timeout_s=timeout_s
            )
        except BuilderError as e:
            e.add_note(
                "the fit itself completed; only the export invocation "
                f"({' '.join(export_args)}) failed, and the fit's log and "
                "metrics are already kept as artifacts"
            )
            raise
        saved_model = seed_dir / "saved_model"
        if not saved_model.is_dir():
            raise BuilderError(
                f"gracemaker's export reported success but {seed_dir.name}/"
                "saved_model/ does not exist; read the kept training log"
            )
        fs_model = seed_dir / "FS_model.yaml"
        if export_fs and not fs_model.is_file():
            raise BuilderError(
                "export_fs=True but gracemaker wrote no FS_model.yaml — the "
                "-sf export applies to FS-preset fits only"
            )
        if active is not None:
            tarball = scratch / f"{name}-saved-model.tar.gz"
            _deterministic_tar(saved_model, tarball, arcroot="saved_model")
            kept = _keep_unique(active, tarball.name, tarball)
            artifact_hashes[kept] = active.runs.get_artifact(active.id, kept).hash
            if export_fs:
                kept = _keep_unique(active, f"{name}-FS-model.yaml", fs_model)
                artifact_hashes[kept] = active.runs.get_artifact(active.id, kept).hash
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(saved_model, out_dir / "saved_model")
        for fname in ("model.yaml", "train_metrics.yaml", "test_metrics.yaml"):
            source = seed_dir / fname
            if source.is_file():
                shutil.copy2(source, out_dir / fname)
        if export_fs:
            shutil.copy2(fs_model, out_dir / "FS_model.yaml")
        train_metrics = _final_metrics(seed_dir / "train_metrics.yaml")
        test_metrics = _final_metrics(seed_dir / "test_metrics.yaml")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    model: dict[str, Any] = {
        "builder": "gracemaker",
        "seed_dir": f"seed/{seed_dir.name}",
        "output_dir": str(out_dir),
        "saved_model": str(out_dir / "saved_model"),
        "fs_model": str(out_dir / "FS_model.yaml") if export_fs else None,
        "artifacts": artifact_hashes,
    }
    info: dict[str, Any] = {
        "builder": "gracemaker",
        "command": described["command"],
        "version": described["version"],
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "artifacts": artifact_hashes,
        "output_dir": str(out_dir),
    }
    return model, info


# Extensions atomsk writes that ASE reads only under an explicit format name.
_BUILDER_FORMATS = {
    ".lmp": "lammps-data",
    ".pw": "espresso-in",
    ".pos": "vasp",
}


def _read_built(path: Path) -> Any:
    """Read a builder's output file, naming the format where ASE cannot infer it."""
    explicit = _BUILDER_FORMATS.get(path.suffix.lower())
    if explicit is not None:
        return ase_read(path, format=explicit)
    return ase_read(path)


def _guard_staged_name(name: str) -> None:
    """Refuse an ``inputs`` key that is not a bare file name."""
    if not name or "/" in name or "\\" in name or name.startswith(("~", ".")):
        raise BuilderError(
            f"inputs name {name!r} must be a bare file name (it is written "
            "into the scratch directory the build runs in)"
        )


def _pick_output(output: str | None, produced: list[str]) -> str:
    """The produced file to read back, or a refusal that names the candidates."""
    if output is not None:
        if output not in produced:
            raise BuilderError(
                f"output={output!r} was not produced; atomsk wrote: "
                f"{', '.join(produced) or 'nothing'}"
            )
        return output
    if len(produced) == 1:
        return produced[0]
    if not produced:
        raise BuilderError(
            "atomsk terminated without writing any file; check that the "
            "argument list names an output file"
        )
    raise BuilderError(
        f"atomsk produced {len(produced)} files ({', '.join(produced)}); "
        "pass output= to name the one holding the result"
    )


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


#: Tasks whose recorded return[0] is a labeled structure (an Atoms carrying a
#: SinglePointCalculator with the engine's exact energy and forces).
_LABEL_SOURCES = frozenset({"relax", "relax_cell", "single_point"})


@contextmanager
def _training_stores() -> Iterator[tuple[Any, Any]]:
    """``(runs, artifacts)`` for the active run, else the resolved workspace.

    Outside a run the task is just a function; it then reads the same
    workspace the CLI would (flag > ``$SLAB_WORKSPACE`` > config >
    ``.slab``), opened for this call and closed after it.
    """
    active = current_run()
    if active is not None:
        yield active.runs, active.artifacts
        return
    from foundation._ops import resolve_root
    from foundation.runtime import Workspace

    workspace = Workspace(resolve_root(None))
    try:
        yield workspace.runs, workspace.artifacts
    finally:
        workspace.close()


def _training_sources(
    run_ids: Sequence[str],
    *,
    engine: str | None,
    frames: str,
    allow_mixed: bool,
) -> list[dict[str, Any]]:
    """Resolve every labeled source to its content hash, refusing gaps loudly.

    Shared by ``collect_training_data``'s cache identity and its body, so
    the cache key and the work read exactly the same selection.
    """
    with _training_stores() as (runs, artifacts):
        return _resolve_training_sources(
            runs, artifacts, run_ids, engine=engine, frames=frames, allow_mixed=allow_mixed
        )


def _resolve_training_sources(
    runs: Any,
    artifacts: Any,
    run_ids: Sequence[str],
    *,
    engine: str | None,
    frames: str,
    allow_mixed: bool,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    engines_seen: set[str] = set()
    for raw_id in run_ids:
        run_id = runs.get(raw_id).id
        records = runs.list_tasks(run_id)
        eligible = [
            record
            for record in records
            if record.name in _LABEL_SOURCES
            and record.status == ExecutionStatus.COMPLETED
            and record.outputs.get("return[0]") is not None
        ]
        run_engines = {
            str(record.recipe.get("params", {}).get("engine"))
            for record in eligible
            if record.recipe.get("params", {}).get("engine") is not None
        }
        if engine is not None:
            eligible = [
                record
                for record in eligible
                if record.recipe.get("params", {}).get("engine") == engine
            ]
        run_sources: list[dict[str, Any]] = [
            {
                "run_id": run_id,
                "kind": "task",
                "ref": f"{record.name}#{record.seq}",
                "hash": record.outputs["return[0]"],
                "engine": record.recipe.get("params", {}).get("engine"),
            }
            for record in eligible
        ]
        if frames == "all":
            if len(run_engines) > 1 and (engine is not None or not allow_mixed):
                raise FoundationError(
                    f"run {run_id} holds tasks under more than one engine "
                    f"({', '.join(sorted(run_engines))}), so its trajectory "
                    "frames cannot be attributed to one; collect it with "
                    'frames="final", or allow_mixed=True without engine='
                )
            if engine is None or run_engines == {engine}:
                run_engine = next(iter(run_engines)) if len(run_engines) == 1 else None
                run_sources.extend(
                    {
                        "run_id": run_id,
                        "kind": "traj",
                        "ref": ref.name,
                        "hash": ref.hash,
                        "engine": run_engine,
                    }
                    for ref in runs.list_artifacts(run_id)
                    if ref.name.endswith(".traj")
                    and not ref.name.endswith("-failed.traj")
                )
        if not run_sources:
            recorded = sorted({record.name for record in records}) or ["no tasks"]
            constraint = f" under engine {engine!r}" if engine is not None else ""
            raise FoundationError(
                f"run {run_id} contributes no labeled structures{constraint}: "
                f"it recorded {', '.join(recorded)}; eligible sources are "
                "completed relax, relax_cell, and single_point tasks"
            )
        for source in run_sources:
            if not artifacts.has(source["hash"]):
                raise FoundationError(
                    f"the bytes of {source['ref']} in run {run_id} "
                    f"({source['hash'][:12]}…) were discarded by retention; "
                    "re-run the source or collect from runs whose bytes are "
                    "still held"
                )
            if source["engine"]:
                engines_seen.add(str(source["engine"]))
        sources.extend(run_sources)
    if engine is None and len(engines_seen) > 1 and not allow_mixed:
        raise FoundationError(
            "the selected runs mix labels from engines "
            f"{', '.join(sorted(engines_seen))}; a training set under mixed "
            "engines is usually an error — pass engine= to pick one, or "
            "allow_mixed=True to state that mixing is intended"
        )
    return sources


def _file_sha256(path: str | Path) -> str:
    """Streamed sha256 of a file; a missing dataset refuses before any work."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as e:
        raise BuilderError(f"cannot read dataset {str(path)!r}: {e}") from e
    return digest.hexdigest()


def _the_seed_dir(scratch: Path) -> Path:
    """The one ``seed/<N>/`` tree gracemaker worked in, or a refusal."""
    seed_root = scratch / "seed"
    candidates = sorted(p for p in seed_root.glob("*") if p.is_dir()) if seed_root.is_dir() else []
    if not candidates:
        raise BuilderError(
            "gracemaker reported success but wrote no seed/<N>/ working "
            "tree; read its console log for what actually happened"
        )
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise BuilderError(
            f"gracemaker left more than one seed directory (seed/{{{names}}}); "
            "one fit per task call — train each seed in its own call"
        )
    return candidates[0]


def _deterministic_tar(source: Path, dest: Path, *, arcroot: str) -> None:
    """Tar a directory reproducibly: sorted members, zeroed times and owners.

    Determinism is dedup hygiene for the artifact store, not cache
    identity — task outputs never enter cache keys.
    """
    with (
        open(dest, "wb") as raw,
        # filename="" keeps the destination's name out of the gzip header;
        # GzipFile would otherwise embed fileobj.name, breaking determinism.
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz,
        tarfile.open(fileobj=gz, mode="w") as tar,
    ):
        for path in sorted(source.rglob("*")):
            info = tar.gettarinfo(path, arcname=f"{arcroot}/{path.relative_to(source)}")
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            if path.is_file():
                with open(path, "rb") as handle:
                    tar.addfile(info, handle)
            else:
                tar.addfile(info)


_METRIC_LINE = re.compile(
    r"^\s*([A-Za-z_][\w/.-]*)\s*:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*,?\s*$"
)


def _final_metrics(path: Path, tail_lines: int = 20) -> dict[str, Any]:
    """The final epoch's metrics from a gracemaker metrics file.

    Gracemaker writes one YAML list item per epoch, each a flow mapping
    that is also strict JSON (observed against tensorpotential 0.6.0):
    ``- {"rmse/depa": ..., "epoch": N}``. The last parseable row is the
    final state of the fit. A minimal parse, never PyYAML (not a
    dependency): JSON rows first, a flat ``key: number`` scan second, and
    the raw tail last, so the evidence always reaches the run record.
    """
    import json

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return {"unreadable": str(e)}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        if line.startswith("- {") and line.endswith("}"):
            try:
                row = json.loads(line[2:])
            except ValueError:
                continue
            if isinstance(row, dict):
                return row
    metrics: dict[str, Any] = {}
    for line in lines:
        match = _METRIC_LINE.match(line)
        if match:
            metrics[match.group(1)] = float(match.group(2))
    if metrics:
        return metrics
    return {"raw_tail": lines[-tail_lines:]}


def _keep_training_failure(active: Any, scratch: Path, name: str, e: BuilderError) -> None:
    """Keep a failed fit's bounded evidence with the run. Never raises.

    The seed tree's log and partial metrics, up to 20 checkpoint files
    (MB-scale weights — the multi-GB bulk is TensorFlow temp data, which
    dies with scratch), and the captured console output.
    """
    kept: list[str] = []
    with suppress(Exception):
        for seed_dir in sorted((scratch / "seed").glob("*")):
            for fname, kept_as in (
                ("log.txt", f"{name}-failed.log"),
                ("train_metrics.yaml", f"{name}-failed-train-metrics.yaml"),
                ("test_metrics.yaml", f"{name}-failed-test-metrics.yaml"),
            ):
                source = seed_dir / fname
                if source.is_file():
                    kept.append(_keep_unique(active, kept_as, source))
            checkpoints = seed_dir / "checkpoints"
            if checkpoints.is_dir():
                files = sorted(p for p in checkpoints.rglob("*") if p.is_file())
                for ckpt in files[:20]:
                    kept.append(
                        _keep_unique(active, f"{name}-failed-checkpoint-{ckpt.name}", ckpt)
                    )
        if e.log:
            console = scratch / "slab-gracemaker-console.log"
            console.write_text(e.log, encoding="utf-8")
            kept.append(_keep_unique(active, f"{name}-failed-console.log", console))
    if kept:
        e.add_note(
            "training evidence kept as artifacts: "
            + ", ".join(repr(k) for k in kept)
        )


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
    e: Exception,
    optimizer: BFGS,
    trajectory: Path,
    label: str | None,
    calculator: Any,
    *,
    task_name: str = "relax",
) -> None:
    """Best-effort failure evidence: note the last-known state on *e*, keep
    the partial trajectory, and fold in whatever the engine itself left
    behind. Never raises — diagnostics must not mask the original failure.

    The notes (surfaced by :func:`foundation.errors.failure_record`) and the kept
    files are what turn "it crashed" into a decidable situation: did the
    structure fly apart, did the SCF diverge, what did ``pw.x`` actually say.
    """
    try:
        note = (
            f"{task_name} failed after "
            f"{optimizer.get_number_of_steps()} completed step(s)"
        )
        last = _last_trajectory_frame(trajectory)
        if last is not None:
            note += (
                f"; trajectory has {last['frames']} frame(s), last frame: "
                f"E={last['energy']:.6f} eV, max|F|={last['fmax']:.4f} eV/Å"
            )
        active = current_run()
        if active is not None and trajectory.exists() and trajectory.stat().st_size > 0:
            kept_as = _keep_unique(active, f"{label or task_name}-failed.traj", trajectory)
            note += f"; partial trajectory kept as artifact {kept_as!r}"
        e.add_note(note)
    except Exception as diagnostics_error:
        with suppress(Exception):
            e.add_note(
                f"({task_name} failure diagnostics unavailable: {diagnostics_error})"
            )
    _attach_engine_evidence(e, calculator, label, task_name=task_name)


def _attach_engine_evidence(
    e: Exception, calculator: Any, label: str | None, *, task_name: str = "relax"
) -> None:
    """File-IO engines tell their failure story in files (``espresso.pwo``,
    ``CRASH``), not in the exception — which is a bare ``CalledProcessError``.
    Note the story on *e* and, inside a run, keep the files before the
    engine's scratch directory vanishes. Never raises.
    """
    try:
        notes, evidence_files = collect_failure_evidence(calculator, e)
        # Parsed evidence attaches before any storage happens: a failing
        # artifact store (disk full) must not cost the already-extracted
        # engine error message.
        for note in notes:
            e.add_note(note)
        active = current_run()
        if active is not None and evidence_files:
            kept = [
                _keep_unique(active, f"{label or task_name}-failed.{suffix}", path)
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
