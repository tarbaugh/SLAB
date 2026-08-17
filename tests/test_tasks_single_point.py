"""Tests for the single_point task (EMT keeps them fast and dep-free).

The engine-specific halves — a fake pw.x replay and failure evidence — live
with their engines in test_backends_qe.py; here is the task contract itself:
one evaluation, honest info (no 'converged'), artifact and caching behavior,
and the two-fidelity chain relax -> single_point linking by hash.
"""

from pathlib import Path

import pytest
from ase.build import bulk

from slab import ExecutionStatus, Workspace
from slab.tasks import relax, single_point


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    with Workspace(tmp_path / "ws") as workspace:
        yield workspace


def _rattled_cu(seed: int = 42, stdev: float = 0.05):  # -> Atoms
    atoms = bulk("Cu", "fcc", a=3.58) * (2, 2, 2)
    atoms.rattle(stdev=stdev, seed=seed)
    return atoms


def test_single_point_untraced_evaluates() -> None:
    from ase.calculators.emt import EMT

    atoms = _rattled_cu()
    evaluated, info = single_point(atoms, engine="emt")

    reference = _rattled_cu()
    reference.calc = EMT()
    assert info["energy"] == pytest.approx(reference.get_potential_energy())
    assert info["energy_unit"] == "eV"
    assert info["n_atoms"] == len(atoms)
    assert info["fmax"] > 0  # a rattled structure has residual forces
    assert "converged" not in info  # nothing was optimized; no false promise
    assert evaluated.get_potential_energy() == pytest.approx(info["energy"])
    assert evaluated.get_forces().shape == (len(atoms), 3)


def test_single_point_does_not_mutate_input() -> None:
    atoms = _rattled_cu()
    before = atoms.get_positions().copy()
    single_point(atoms, engine="emt")
    assert atoms.calc is None
    assert atoms.get_positions() == pytest.approx(before)


def test_single_point_accepts_relax_output_directly() -> None:
    """The two-fidelity chain: relax's returned structure (which carries a
    serializable SinglePointCalculator) is a valid single_point input as-is."""
    relaxed, cheap = relax(_rattled_cu(), engine="emt", fmax=0.05)
    _evaluated, expensive = single_point(relaxed, engine="emt")
    assert expensive["energy"] == pytest.approx(cheap["energy"])
    assert expensive["fmax"] == pytest.approx(cheap["fmax"])


def test_traced_chain_links_by_hash(ws: Workspace) -> None:
    with ws.start_run(name="two-fidelity") as run:
        relaxed, _cheap = relax(_rattled_cu(), engine="emt", fmax=0.05, label="cheap")
        _final, _dft = single_point(relaxed, engine="emt", label="expensive")

    first, second = ws.runs.list_tasks(run.id)
    assert first.name == "relax"
    assert second.name == "single_point"
    assert second.status is ExecutionStatus.COMPLETED
    assert second.recipe["params"]["engine"] == "emt"
    # DAG by equality: single_point consumed exactly what relax produced.
    assert second.inputs["atoms"] == first.outputs["return[0]"]
    assert set(second.inputs) == {"atoms", "engine", "calculator_options", "label"}


def test_traced_single_point_caches_identical_calls(ws: Workspace) -> None:
    with ws.start_run() as first:
        _, i1 = single_point(_rattled_cu(), engine="emt")
    with ws.start_run() as second:
        _, i2 = single_point(_rattled_cu(), engine="emt")

    assert ws.runs.list_tasks(first.id)[0].cache_hit is False
    assert ws.runs.list_tasks(second.id)[0].cache_hit is True
    assert i2 == i1


def test_single_point_closes_worker_backed_calculators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ase.calculators.emt import EMT

    closed: list[bool] = []

    class WorkerBackedEMT(EMT):
        def close(self) -> None:
            closed.append(True)

    monkeypatch.setattr("slab.tasks.get_calculator", lambda engine, **kw: WorkerBackedEMT())
    single_point(_rattled_cu(), engine="emt")
    assert closed == [True]


def test_single_point_closes_calculator_even_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []

    class ExplodingCalculator:
        def close(self) -> None:
            closed.append(True)

        def get_forces(self, *a: object, **k: object) -> None:
            raise RuntimeError("SCF diverged")

    monkeypatch.setattr("slab.tasks.get_calculator", lambda engine, **kw: ExplodingCalculator())
    with pytest.raises(RuntimeError, match="SCF diverged"):
        single_point(_rattled_cu(), engine="emt")
    assert closed == [True]


def test_failed_single_point_records_failure_without_artifacts(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ExplodingCalculator:
        def get_forces(self, *a: object, **k: object) -> None:
            raise RuntimeError("SCF diverged")

    monkeypatch.setattr("slab.tasks.get_calculator", lambda engine, **kw: ExplodingCalculator())
    with (
        pytest.raises(RuntimeError, match="SCF diverged"),
        ws.start_run(name="doomed") as run,
    ):
        single_point(_rattled_cu(), engine="emt")

    (record,) = ws.runs.list_tasks(run.id)
    assert record.status is ExecutionStatus.FAILED
    assert ws.runs.list_artifacts(run.id) == []  # in-memory engine left nothing


# -- qe-specific guards ----------------------------------------------------------------


def test_qe_bulk_without_kpoints_is_refused() -> None:
    """Γ-only on a bulk cell converges and looks plausible — refuse it loudly."""
    with pytest.raises(ValueError, match="k-points"):
        single_point(bulk("Si", "diamond", a=5.43), engine="qe", calculator_options={})


def test_relax_qe_bulk_without_kpoints_is_refused() -> None:
    with pytest.raises(ValueError, match="k-points"):
        relax(bulk("Si", "diamond", a=5.43), engine="qe", calculator_options={})


def test_scf_pin_and_conflict_refusal() -> None:
    from slab.tasks import _qe_scf_options

    pinned = _qe_scf_options(None)
    assert pinned["input_data"]["calculation"] == "scf"

    nested = {"input_data": {"control": {"tprnfor": True}, "system": {"ecutwfc": 30.0}}}
    before = {"input_data": {"control": {"tprnfor": True}, "system": {"ecutwfc": 30.0}}}
    out = _qe_scf_options(nested)
    assert out["input_data"]["control"]["calculation"] == "scf"
    assert nested == before  # the caller's dict is a traced input: never mutated

    with pytest.raises(ValueError, match=r"calculation='relax'"):
        _qe_scf_options({"input_data": {"control": {"calculation": "relax"}}})
    with pytest.raises(ValueError, match="vc-relax"):
        _qe_scf_options({"input_data": {"calculation": "vc-relax"}})  # flat form too
    explicit = _qe_scf_options({"input_data": {"control": {"calculation": "scf"}}})
    assert explicit["input_data"]["control"]["calculation"] == "scf"
