"""Tests for the relax task and engine backends (EMT keeps them fast and dep-free)."""

from pathlib import Path

import pytest
from ase.build import bulk

from slab import (
    ArtifactRole,
    EngineNotAvailableError,
    ExecutionStatus,
    Workspace,
    dumps,
    loads,
)
from slab.backends import available_engines, get_calculator
from slab.tasks import relax


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
    with pytest.raises(EngineNotAvailableError, match="emt, lammps, lj, mace, qe, rootstock"):
        get_calculator("definitely-unknown-engine")


def test_engine_list_is_sorted_and_stable() -> None:
    assert available_engines() == ("emt", "lammps", "lj", "mace", "qe", "rootstock")


def test_mace_missing_gives_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def hide_mace(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("mace"):
            raise ImportError("No module named 'mace'")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", hide_mace)
    with pytest.raises(EngineNotAvailableError, match=r"pip install 'slab\[mace\]'"):
        get_calculator("mace")


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

    monkeypatch.setattr("slab.tasks.get_calculator", lambda engine, **kw: WorkerBackedEMT())
    relax(_rattled_cu(), engine="emt", fmax=0.05)
    assert closed == [True]


def test_relax_closes_calculator_even_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[bool] = []

    class ExplodingCalculator:
        def close(self) -> None:
            closed.append(True)

        def get_potential_energy(self, *a: object, **k: object) -> float:
            raise RuntimeError("SCF diverged")

    monkeypatch.setattr("slab.tasks.get_calculator", lambda engine, **kw: ExplodingCalculator())
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
        "slab.tasks.get_calculator", lambda engine, **kw: _emt_that_fails_after(3)
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
        "slab.tasks.get_calculator", lambda engine, **kw: _emt_that_fails_after(2)
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
        "slab.tasks.get_calculator", lambda engine, **kw: _emt_that_fails_after(0)
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
        "slab.tasks.get_calculator", lambda engine, **kw: _emt_that_fails_after(2)
    )

    def keep_explodes(active: object, name: str, path: object) -> str:
        raise OSError("disk full")

    monkeypatch.setattr("slab.tasks._keep_unique", keep_explodes)
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

    monkeypatch.setattr("slab.tasks.get_calculator", build_then_delete_registry)
    _relaxed, info = relax(_rattled_cu(), engine="emt-cluster", fmax=0.05)
    assert info["converged"] is True
    assert info["engine_version"] == "1.0"  # stamped from the pre-run snapshot
    assert info["engine_source"] == "registry:delta"


def test_mace_extra_importable_when_installed() -> None:
    """With slab[mace] installed, the import path the factory uses must resolve.

    (Building the calculator itself downloads a model checkpoint, so the real
    end-to-end exercise lives in examples/demo.py, not the unit suite.)
    """
    pytest.importorskip("mace.calculators", reason="mace-torch not installed")
    from mace.calculators import mace_mp

    assert callable(mace_mp)


def test_mace_identity_names_model_and_dist_version() -> None:
    """engine='mace' cache identity must say which MLIP actually runs: the
    resolved checkpoint (slab's own 'small' default included) and the
    mace-torch version — changing either must invalidate honestly."""
    from importlib.metadata import version

    from slab.backends import describe_engine

    identity = describe_engine("mace")
    assert identity["model"] == "small"
    assert identity["version"] == version("mace-torch")
    assert describe_engine("mace", {"model": "medium"})["model"] == "medium"
