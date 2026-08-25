"""Built-in skill content: every shipped script runs, fits, and refuses well.

Scripts execute through ``runpy`` with a patched ``argv`` — in-process, so
they stay under coverage and cannot collide as same-named modules.
"""

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write as ase_write

from mason.skills import discover_skills

SKILLS = Path(__file__).parent.parent / "src" / "mason" / "skills"
FIT_EOS = SKILLS / "equation-of-state" / "scripts" / "fit_eos.py"
CONV = SKILLS / "convergence-study" / "scripts" / "convergence_table.py"
RDF = SKILLS / "radial-distribution" / "scripts" / "rdf.py"
MSD = SKILLS / "msd-diffusion" / "scripts" / "msd.py"


def _run(
    script: Path, *args: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> tuple[object, str]:
    """Execute a script as __main__; return (exit code, captured stdout)."""
    monkeypatch.setattr(sys, "argv", [str(script), *args])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(script), run_name="__main__")
    return excinfo.value.code, capsys.readouterr().out


def _emt_eos_points() -> list[dict[str, float]]:
    base = bulk("Cu", "fcc", a=3.58)
    points = []
    for scale in (0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06):
        atoms = base.copy()
        atoms.set_cell(base.cell * scale, scale_atoms=True)
        atoms.calc = EMT()
        points.append({"volume": atoms.get_volume(), "energy": atoms.get_potential_energy()})
    return points


# -- fit_eos ------------------------------------------------------------------


def test_fit_eos_recovers_emt_copper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "eos.json"
    data.write_text(json.dumps(_emt_eos_points()))
    code, out = _run(
        FIT_EOS, str(data), "--json", "--natoms", "1", monkeypatch=monkeypatch, capsys=capsys
    )
    assert code == 0
    fit = json.loads(out)
    # EMT copper: a0 near 3.59 A -> V0/atom near 11.6 A^3; B near 140 GPa.
    assert 10.5 < fit["v0_per_atom_A3"] < 12.5
    assert 80.0 < fit["b_GPa"] < 220.0
    assert fit["points"] == 7


def test_fit_eos_refuses_thin_or_malformed_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    thin = tmp_path / "thin.json"
    thin.write_text(json.dumps(_emt_eos_points()[:3]))
    code, _ = _run(FIT_EOS, str(thin), monkeypatch=monkeypatch, capsys=capsys)
    assert isinstance(code, str) and "not enough" in code

    junk = tmp_path / "junk.json"
    junk.write_text('{"volume": 1}')
    code, _ = _run(FIT_EOS, str(junk), monkeypatch=monkeypatch, capsys=capsys)
    assert isinstance(code, str) and "list" in code

    code, _ = _run(FIT_EOS, str(tmp_path / "gone.json"), monkeypatch=monkeypatch, capsys=capsys)
    assert isinstance(code, str) and "no such file" in code


# -- convergence_table --------------------------------------------------------


def test_convergence_table_names_the_cheapest_converged_rung(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ladder = [
        {"value": 30, "energy": -10.0500},
        {"value": 40, "energy": -10.0100},
        {"value": 50, "energy": -10.0008},
        {"value": 60, "energy": -10.0002},
        {"value": 70, "energy": -10.0000},
    ]
    data = tmp_path / "conv.json"
    data.write_text(json.dumps(ladder))
    code, out = _run(
        CONV, str(data), "--natoms", "1", "--json", monkeypatch=monkeypatch, capsys=capsys
    )
    assert code == 0
    verdict = json.loads(out)
    assert verdict["converged_at"] == "50"
    assert verdict["unit"] == "meV/atom"

    code, out = _run(CONV, str(data), "--natoms", "1", monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0
    assert "converged at 50" in out
    assert "<- reference" in out


def test_convergence_table_says_when_the_ladder_is_too_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "conv.json"
    data.write_text(json.dumps([{"value": 30, "energy": -9.9}, {"value": 40, "energy": -10.0}]))
    code, out = _run(CONV, str(data), monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0
    assert "not converged" in out
    assert "extend the ladder" in out

    data.write_text(json.dumps([{"value": 30, "energy": -9.9}]))
    code, _ = _run(CONV, str(data), monkeypatch=monkeypatch, capsys=capsys)
    assert isinstance(code, str) and "at least 2 rungs" in code


# -- rdf ----------------------------------------------------------------------


def test_rdf_finds_the_fcc_nearest_neighbor_peak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    atoms = bulk("Cu", "fcc", a=3.615, cubic=True) * (3, 3, 3)
    data = tmp_path / "cu.xyz"
    ase_write(data, atoms)
    code, out = _run(RDF, str(data), "--json", monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0
    result = json.loads(out)
    # fcc nearest neighbors sit at a/sqrt(2) = 2.556 A.
    assert 2.45 < result["first_peak_r_A"] < 2.66
    assert result["first_peak_g"] > 3.0
    assert result["frames"] == 1


def test_rdf_refuses_rmax_beyond_the_minimum_image_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    atoms = bulk("Cu", "fcc", a=3.615, cubic=True) * (2, 2, 2)
    data = tmp_path / "cu.xyz"
    ase_write(data, atoms)
    code, _ = _run(RDF, str(data), "--rmax", "50", monkeypatch=monkeypatch, capsys=capsys)
    assert isinstance(code, str) and "minimum-image convention" in code


def test_rdf_species_filter_refuses_absent_elements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    atoms = bulk("Cu", "fcc", a=3.615, cubic=True)
    data = tmp_path / "cu.xyz"
    ase_write(data, atoms)
    code, _ = _run(
        RDF, str(data), "--species", "Li", "Li", monkeypatch=monkeypatch, capsys=capsys
    )
    assert isinstance(code, str) and "no Li-Li pairs" in code and "Cu" in code


# -- msd ----------------------------------------------------------------------


def test_msd_recovers_a_random_walk_diffusion_coefficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from ase import Atoms

    rng = np.random.default_rng(7)
    n_atoms, n_frames, sigma = 40, 240, 0.05
    steps = rng.normal(0.0, sigma, size=(n_frames, n_atoms, 3))
    steps[0] = 0.0
    positions = 50.0 + steps.cumsum(axis=0)
    frames = [
        Atoms("Ar" * n_atoms, positions=positions[i], cell=[100.0] * 3, pbc=False)
        for i in range(n_frames)
    ]
    data = tmp_path / "walk.xyz"
    ase_write(data, frames)
    code, out = _run(
        MSD, str(data), "--dt-fs", "1.0", "--json", monkeypatch=monkeypatch, capsys=capsys
    )
    assert code == 0
    result = json.loads(out)
    expected = sigma**2 / 2.0  # MSD = 3 sigma^2 n = 6 D t with dt = 1 fs
    assert 0.6 * expected < result["d_A2_per_fs"] < 1.6 * expected
    assert result["atoms_averaged"] == n_atoms


def test_msd_refuses_short_input_and_bad_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from ase import Atoms

    frames = [Atoms("Ar", positions=[[0, 0, 0]], cell=[10.0] * 3) for _ in range(5)]
    data = tmp_path / "short.xyz"
    ase_write(data, frames)
    code, _ = _run(MSD, str(data), "--dt-fs", "1.0", monkeypatch=monkeypatch, capsys=capsys)
    assert isinstance(code, str) and "not enough for an MSD" in code
    code, _ = _run(MSD, str(data), "--dt-fs", "-1", monkeypatch=monkeypatch, capsys=capsys)
    assert isinstance(code, str) and "--dt-fs" in code


# -- the template and the whole loop ------------------------------------------


def test_the_eos_template_runs_as_a_verified_traced_run_and_fits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The skill's own promise, executed: template -> verified run -> fit."""
    from foundation._ops import launch_script

    monkeypatch.chdir(tmp_path)
    template = SKILLS / "equation-of-state" / "assets" / "eos_scan.py"
    result = launch_script(
        tmp_path / ".slab",
        template,
        name="eos-shakeout",
        intent="skill template shakeout (EMT, laptop settings)",
        capture_output=True,
    )
    assert result["state"] == "verified"
    assert result["checks_passed"] == result["checks_total"] == 2
    assert (tmp_path / "eos.json").is_file()

    code, out = _run(
        FIT_EOS, str(tmp_path / "eos.json"), "--json", "--natoms", "1",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    fit = json.loads(out)
    assert 80.0 < fit["b_GPa"] < 220.0


def test_every_builtin_skill_validates_and_maps_to_its_specialists(tmp_path: Path) -> None:
    skills = discover_skills(tmp_path)
    assert {
        "equation-of-state",
        "convergence-study",
        "radial-distribution",
        "msd-diffusion",
        "surface-energy",
    } <= set(skills)
    assert skills["convergence-study"].agents == frozenset({"dft-expert"})
    assert skills["surface-energy"].agents == frozenset({"dft-expert"})
    assert skills["radial-distribution"].agents == frozenset({"md-expert", "analysis-expert"})
    assert skills["msd-diffusion"].agents == frozenset({"md-expert", "analysis-expert"})
    # The scriptless skill is deliberate: scripts are optional in the format.
    assert not (skills["surface-energy"].root / "scripts").exists()
