"""Tests for the relax task and engine backends (EMT keeps them fast and dep-free)."""

from pathlib import Path

import pytest
from ase.build import bulk

from foundation import ArtifactRole, ExecutionStatus, Workspace, dumps, loads
from foundation.tasks import relax
from slab import EngineNotAvailableError
from slab.backends import available_engines, get_calculator


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    with Workspace(tmp_path / "ws") as workspace:
        yield workspace


def _rattled_cu(seed: int = 42, stdev: float = 0.05):  # -> Atoms
    atoms = bulk("Cu", "fcc", a=3.58) * (2, 2, 2)
    atoms.rattle(stdev=stdev, seed=seed)
    return atoms


# -- backends --------------------------------------------------------------------------


def test_emt_and_lj_calculators_build() -> None:
    assert type(get_calculator("emt")).__name__ == "EMT"
    assert type(get_calculator("LJ ".strip().upper())).__name__ == "LennardJones"


def test_unknown_engine_lists_options() -> None:
    with pytest.raises(EngineNotAvailableError, match="emt, lammps, lj, qe, rootstock"):
        get_calculator("definitely-unknown-engine")


def test_engine_list_is_sorted_and_stable() -> None:
    assert available_engines() == ("emt", "lammps", "lj", "qe", "rootstock")


# -- relax, untraced -------------------------------------------------------------------


def test_relax_untraced_converges() -> None:
    atoms = _rattled_cu()
    relaxed, info = relax(atoms, engine="emt", fmax=0.05)
    assert info["converged"] is True
    assert info["fmax"] < 0.05
    assert info["energy_unit"] == "eV"
    assert info["engine_source"] == "builtin"
    assert info["engine_version"] is None
    assert info["n_atoms"] == len(atoms) == len(relaxed)
    assert info["steps"] > 0
    # input untouched, output carries results on a serializable calculator
    assert atoms.get_positions() == pytest.approx(_rattled_cu().get_positions())
    assert relaxed.get_potential_energy() == pytest.approx(info["energy"])
    assert relaxed.calc.__class__.__name__ == "SinglePointCalculator"


def test_relaxed_atoms_survive_serialization() -> None:
    relaxed, info = relax(_rattled_cu(), engine="emt")
    clone = loads(dumps(relaxed))
    assert clone.get_positions() == pytest.approx(relaxed.get_positions())
    assert clone.get_potential_energy() == pytest.approx(info["energy"])


# -- relax, traced ---------------------------------------------------------------------


def test_traced_relax_records_task_and_trajectory(ws: Workspace) -> None:
    with ws.start_run(name="cu-relax") as run:
        _relaxed, _info = relax(_rattled_cu(), engine="emt", label="variant-0")

    (record,) = ws.runs.list_tasks(run.id)
    assert record.name == "relax"
    assert record.status is ExecutionStatus.COMPLETED
    assert record.recipe["engines"]["ase"] is not None  # pinned engine version
    assert record.recipe["params"]["engine"] == "emt"
    assert set(record.inputs) == {
        "atoms",
        "engine",
        "fmax",
        "steps",
        "calculator_options",
        "label",
    }

    (traj,) = ws.runs.list_artifacts(run.id)
    assert traj.name == "variant-0.traj"
    assert traj.role is ArtifactRole.INTERMEDIATE
    assert traj.size_bytes > 0
    assert ws.artifacts.has(traj.hash)


def test_traced_relax_caches_identical_calls(ws: Workspace) -> None:
    with ws.start_run() as first:
        a1, i1 = relax(_rattled_cu(), engine="emt")
    with ws.start_run() as second:
        a2, i2 = relax(_rattled_cu(), engine="emt")

    assert ws.runs.list_tasks(first.id)[0].cache_hit is False
    assert ws.runs.list_tasks(second.id)[0].cache_hit is True
    assert i2 == i1
    assert a2.get_positions() == pytest.approx(a1.get_positions())
    # the cached run never executed relax, so it stored no new trajectory
    assert ws.runs.list_artifacts(second.id) == []


def test_trajectory_names_auto_suffix_on_collision(ws: Workspace) -> None:
    with ws.start_run() as run:
        relax(_rattled_cu(seed=1), engine="emt")
        relax(_rattled_cu(seed=2), engine="emt")
    names = [a.name for a in ws.runs.list_artifacts(run.id)]
    assert names == ["relax.traj", "relax-2.traj"]


def test_relax_gc_story(ws: Workspace) -> None:
    """Promoted relax run: structure kept (terminal), trajectory dropped (intermediate)."""
    with ws.start_run(name="keeper") as run:
        relaxed, _info = relax(_rattled_cu(), engine="emt", label="cu")
        run.keep("relaxed", relaxed)
    ws.runs.transition(run.id, "promoted", force=True)

    traj = ws.runs.get_artifact(run.id, "cu.traj")
    kept = ws.runs.get_artifact(run.id, "relaxed")
    report = ws.gc()
    assert traj.hash in report.dropped
    assert kept.hash in report.kept
    assert not ws.artifacts.has(traj.hash)
    restored = loads(ws.artifacts.get(kept.hash).read_bytes())
    assert restored.get_positions() == pytest.approx(relaxed.get_positions())


def test_relax_closes_worker_backed_calculators(monkeypatch: pytest.MonkeyPatch) -> None:
    """Worker-backed engines (rootstock) hold a subprocess; relax must release it."""
    from ase.calculators.emt import EMT

    closed: list[bool] = []

    class WorkerBackedEMT(EMT):
        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr("foundation.tasks.get_calculator", lambda engine, **kw: WorkerBackedEMT())
    relax(_rattled_cu(), engine="emt", fmax=0.05)
    assert closed == [True]


def test_relax_closes_calculator_even_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []

    class ExplodingCalculator:
        def close(self) -> None:
            closed.append(True)

        def get_potential_energy(self, *a: object, **k: object) -> float:
            raise RuntimeError("SCF diverged")

    monkeypatch.setattr(
        "foundation.tasks.get_calculator", lambda engine, **kw: ExplodingCalculator()
    )
    with pytest.raises(Exception):  # noqa: B017 - any failure path must still close
        relax(_rattled_cu(), engine="emt")
    assert closed == [True]


# -- relax failure diagnostics ---------------------------------------------------------


def _emt_that_fails_after(n_calls: int):  # -> Calculator class instance
    """An EMT that behaves normally for *n_calls* force evaluations, then dies —
    a mid-optimization engine failure (SCF divergence, worker crash)."""
    from ase.calculators.emt import EMT

    class DoomedEMT(EMT):
        calls = 0

        def calculate(self, *args: object, **kwargs: object) -> None:
            type(self).calls += 1
            if type(self).calls > n_calls:
                raise RuntimeError("SCF diverged")
            super().calculate(*args, **kwargs)

    return DoomedEMT()


def test_failed_relax_keeps_partial_trajectory_and_notes_state(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core failure-evidence contract: a mid-optimization crash leaves an
    agent the partial trajectory and a last-known-state note, not just a
    one-line error and an empty run."""
    monkeypatch.setattr(
        "foundation.tasks.get_calculator", lambda engine, **kw: _emt_that_fails_after(3)
    )
    with ws.start_run(name="doomed") as run, pytest.raises(
        RuntimeError, match="SCF diverged"
    ) as excinfo:
        relax(_rattled_cu(), engine="emt", label="cu")

    (note,) = excinfo.value.__notes__
    assert "relax failed after" in note
    assert "last frame: E=" in note
    assert "partial trajectory kept as artifact 'cu-failed.traj'" in note

    (traj,) = ws.runs.list_artifacts(run.id)
    assert traj.name == "cu-failed.traj"
    assert traj.role is ArtifactRole.INTERMEDIATE
    assert ws.artifacts.has(traj.hash)
    # the kept bytes really are a readable trajectory with stored energetics
    from ase.io import read

    frames = read(ws.artifacts.get(traj.hash), index=":", format="traj")
    assert len(frames) >= 1
    assert frames[-1].get_potential_energy() < 0

    # and the tracer folded the note into the failed task's evidence record
    (record,) = ws.runs.list_tasks(run.id)
    assert record.status is ExecutionStatus.FAILED
    assert record.failure["notes"] == [note]


def test_failed_relax_outside_a_run_still_notes_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Untraced calls get the note too; there is just no artifact store to keep
    the trajectory in."""
    monkeypatch.setattr(
        "foundation.tasks.get_calculator", lambda engine, **kw: _emt_that_fails_after(2)
    )
    with pytest.raises(RuntimeError, match="SCF diverged") as excinfo:
        relax(_rattled_cu(), engine="emt")
    (note,) = excinfo.value.__notes__
    assert "relax failed after" in note
    assert "kept as artifact" not in note


def test_failure_on_first_evaluation_is_still_clean(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dying before any frame exists must not crash the diagnostics path."""
    monkeypatch.setattr(
        "foundation.tasks.get_calculator", lambda engine, **kw: _emt_that_fails_after(0)
    )
    with ws.start_run() as run, pytest.raises(RuntimeError, match="SCF diverged") as excinfo:
        relax(_rattled_cu(), engine="emt")
    (note,) = excinfo.value.__notes__
    assert "relax failed after 0 completed step(s)" in note
    (record,) = ws.runs.list_tasks(run.id)
    assert record.failure["type"] == "RuntimeError"


def test_diagnostics_failure_never_masks_the_original_error(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "foundation.tasks.get_calculator", lambda engine, **kw: _emt_that_fails_after(2)
    )

    def keep_explodes(active: object, name: str, path: object) -> str:
        raise OSError("disk full")

    monkeypatch.setattr("foundation.tasks._keep_unique", keep_explodes)
    with ws.start_run(), pytest.raises(RuntimeError, match="SCF diverged") as excinfo:
        relax(_rattled_cu(), engine="emt")
    (note,) = excinfo.value.__notes__
    assert "diagnostics unavailable" in note
    assert "disk full" in note


def test_registry_version_bump_invalidates_relax_cache(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: relax through a registry engine; bumping the registry's
    declared version must recompute, never serve the old engine's result."""
    import json

    registry_path = tmp_path / "engines.json"

    def declare(version: str) -> None:
        registry_path.write_text(
            json.dumps(
                {
                    "cluster": "testcluster",
                    "engines": {
                        "emt-cluster": {
                            "calculator": "ase.calculators.emt.EMT",
                            "version": version,
                        }
                    },
                }
            )
        )

    monkeypatch.setenv("SLAB_ENGINES", str(registry_path))
    declare("1.0")
    with ws.start_run() as first:
        _r1, info1 = relax(_rattled_cu(), engine="emt-cluster")
    assert info1["engine_source"] == "registry:testcluster"
    assert info1["engine_version"] == "1.0"

    with ws.start_run() as second:
        relax(_rattled_cu(), engine="emt-cluster")
    assert ws.runs.list_tasks(second.id)[0].cache_hit is True  # same version: hit

    declare("2.0")
    with ws.start_run() as third:
        _r3, info3 = relax(_rattled_cu(), engine="emt-cluster")
    assert ws.runs.list_tasks(third.id)[0].cache_hit is False  # bumped: recompute
    assert info3["engine_version"] == "2.0"
    assert ws.runs.list_tasks(first.id)[0].recipe["extra"]["version"] == "1.0"


def test_registry_spec_edit_invalidates_relax_cache(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not just version: ANY spec edit (env, options) changes the cache identity."""
    import json

    registry_path = tmp_path / "engines.json"

    def declare(env: dict) -> None:
        registry_path.write_text(
            json.dumps(
                {
                    "engines": {
                        "emt-cluster": {
                            "calculator": "ase.calculators.emt.EMT",
                            "version": "1.0",
                            "env": env,
                        }
                    }
                }
            )
        )

    monkeypatch.setenv("SLAB_ENGINES", str(registry_path))
    declare({})
    with ws.start_run():
        relax(_rattled_cu(), engine="emt-cluster")
    declare({"OMP_NUM_THREADS": "4"})  # maintainer changed the entry, same version
    with ws.start_run() as second:
        relax(_rattled_cu(), engine="emt-cluster")
    assert ws.runs.list_tasks(second.id)[0].cache_hit is False


def test_relax_provenance_resolved_before_computation(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registry deleted mid-optimization must neither crash a completed relax
    nor change what gets stamped: identity is captured before the run."""
    import json

    registry_path = tmp_path / "engines.json"
    registry_path.write_text(
        json.dumps(
            {
                "cluster": "delta",
                "engines": {
                    "emt-cluster": {"calculator": "ase.calculators.emt.EMT", "version": "1.0"}
                },
            }
        )
    )
    monkeypatch.setenv("SLAB_ENGINES", str(registry_path))

    real_get_calculator = __import__("slab.backends", fromlist=["get_calculator"]).get_calculator

    def build_then_delete_registry(engine: str, **kw: object) -> object:
        calc = real_get_calculator(engine, **kw)
        registry_path.unlink()  # the maintainer pulls the registry mid-run
        return calc

    monkeypatch.setattr("foundation.tasks.get_calculator", build_then_delete_registry)
    _relaxed, info = relax(_rattled_cu(), engine="emt-cluster", fmax=0.05)
    assert info["converged"] is True
    assert info["engine_version"] == "1.0"  # stamped from the pre-run snapshot
    assert info["engine_source"] == "registry:delta"




# -- relax_cell ------------------------------------------------------------------------


def test_relax_cell_isotropic_converges_a_compressed_reference() -> None:
    from foundation.tasks import relax_cell

    # EMT's Cu equilibrium is a=3.61 A (Cu bulk primitive edge = a/sqrt(2)),
    # so a=3.4 is compressed and the isotropic mode should scale to equilibrium.
    atoms = bulk("Cu", "fcc", a=3.4)
    relaxed, info = relax_cell(atoms, engine="emt", symmetry="isotropic", smax=0.005)
    assert info["converged"] is True
    assert info["smax"] < 0.005
    assert info["symmetry"] == "isotropic"
    assert info["energy_unit"] == "eV"
    assert info["engine_source"] == "builtin"
    a, b, c = info["cell_lengths"]
    assert a == pytest.approx(b) == pytest.approx(c)
    assert info["volume"] > atoms.get_volume()  # relaxation expanded the cell
    # The input structure was not mutated.
    assert atoms.cell.array == pytest.approx(bulk("Cu", "fcc", a=3.4).cell.array)
    assert relaxed.calc.__class__.__name__ == "SinglePointCalculator"


def test_relax_cell_accepts_a_multi_atom_cell() -> None:
    """ASE >= 3.29 defaults FrechetCellFilter's exp_cell_factor to the atom
    count, and CellAwareBFGS asserts it is 1.0 — so every cell with more
    than one atom crashed until the task pinned the factor. The primitive
    fixtures elsewhere have one atom and never saw it."""
    from foundation.tasks import relax_cell

    atoms = bulk("Cu", "fcc", a=3.4, cubic=True)  # 4 atoms
    _, info = relax_cell(atoms, engine="emt", symmetry="isotropic", smax=0.005)
    assert info["converged"] is True
    assert info["n_atoms"] == 4


def test_relax_cell_orthorhombic_holds_shear_at_zero() -> None:
    from foundation.tasks import relax_cell

    atoms = bulk("Cu", "fcc", a=3.4)
    _, info = relax_cell(atoms, engine="emt", symmetry="orthorhombic")
    alpha, beta, gamma = info["cell_angles"]
    assert alpha == pytest.approx(60.0, abs=1e-6)
    assert beta == pytest.approx(60.0, abs=1e-6)
    assert gamma == pytest.approx(60.0, abs=1e-6)


def test_relax_cell_refuses_unknown_symmetry() -> None:
    from foundation.tasks import relax_cell

    with pytest.raises(ValueError, match="unknown symmetry 'hexagonal'"):
        relax_cell(bulk("Cu", "fcc", a=3.6), engine="emt", symmetry="hexagonal")  # type: ignore[arg-type]


def test_relax_cell_records_the_task_and_trajectory(ws: Workspace) -> None:
    from foundation.tasks import relax_cell

    atoms = bulk("Cu", "fcc", a=3.4)
    with ws.start_run(name="cu-cell", intent="Cu EMT cell relax") as run:
        _, info = relax_cell(atoms, engine="emt", symmetry="isotropic", label="cu-cell")
    assert info["converged"] is True
    (record,) = ws.runs.list_tasks(run.id)
    assert record.name == "relax_cell"
    assert record.recipe["params"]["engine"] == "emt"
    assert record.recipe["params"]["symmetry"] == "isotropic"
    (traj,) = ws.runs.list_artifacts(run.id)
    assert traj.name == "cu-cell.traj"
    assert traj.role is ArtifactRole.INTERMEDIATE
