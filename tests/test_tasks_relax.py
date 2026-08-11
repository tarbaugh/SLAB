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
    with pytest.raises(EngineNotAvailableError, match="emt, lj, mace"):
        get_calculator("vasp")


def test_engine_list_is_sorted_and_stable() -> None:
    assert available_engines() == ("emt", "lj", "mace")


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


def test_mace_extra_importable_when_installed() -> None:
    """With slab[mace] installed, the import path the factory uses must resolve.

    (Building the calculator itself downloads a model checkpoint, so the real
    end-to-end exercise lives in examples/demo.py, not the unit suite.)
    """
    pytest.importorskip("mace.calculators", reason="mace-torch not installed")
    from mace.calculators import mace_mp

    assert callable(mace_mp)
