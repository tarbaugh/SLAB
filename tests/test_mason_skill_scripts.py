"""Built-in skill content: every shipped script runs, fits, and refuses well.

Scripts execute through ``runpy`` with a patched ``argv`` — in-process, so
they stay under coverage and cannot collide as same-named modules.
"""

import json
import runpy
import sys
from pathlib import Path
from typing import Any

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
PAIR_STYLE = SKILLS / "lammps-potentials" / "scripts" / "pair_style_for.py"
DATA = Path(__file__).parent / "data"


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
    assert 2.0 < fit["b_prime"] < 8.0
    assert fit["rms_residual_meV_per_atom"] < 1.0
    assert fit["warnings"] == []
    low, high = fit["volume_range_of_v0"]
    assert 0.80 < low < 0.90 and 1.10 < high < 1.25  # linear 0.94-1.06 is wide in volume

    code, out = _run(FIT_EOS, str(data), "--natoms", "1", monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0 and "B' =" in out and "scanned V/V0" in out


def test_fit_eos_warns_in_json_when_the_minimum_is_at_an_edge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Points that all sit on one side of the minimum: the fitted V0 falls
    outside the inner points, and the warning rides in the JSON too."""
    points = [p for p in _emt_eos_points() if p["volume"] > 11.0][:4]
    assert len(points) == 4
    data = tmp_path / "edge.json"
    data.write_text(json.dumps(points))
    code, out = _run(
        FIT_EOS, str(data), "--json", "--natoms", "1", monkeypatch=monkeypatch, capsys=capsys
    )
    assert code == 0
    fit = json.loads(out)
    assert any("outside the inner points" in w for w in fit["warnings"])


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
    assert "confirmed by 2 later rung(s)" in out
    assert "<- reference" in out


def test_convergence_table_reads_forces_and_pressures_and_wants_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Forces converge later than energies here: the energy verdict is 40,
    the force verdict 50, and a pressure ladder needs its own key."""
    ladder = [
        {"value": 30, "energy": -10.0100, "fmax": 0.0300},
        {"value": 40, "energy": -10.0005, "fmax": 0.0120},
        {"value": 50, "energy": -10.0003, "fmax": 0.0040},
        {"value": 60, "energy": -10.0001, "fmax": 0.0010},
        {"value": 70, "energy": -10.0000, "fmax": 0.0000},
    ]
    data = tmp_path / "conv.json"
    data.write_text(json.dumps(ladder))
    code, out = _run(CONV, str(data), "--json", monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0 and json.loads(out)["converged_at"] == "40"

    code, out = _run(
        CONV, str(data), "--quantity", "force", "--json", monkeypatch=monkeypatch, capsys=capsys
    )
    assert code == 0
    verdict = json.loads(out)
    assert verdict["converged_at"] == "50" and verdict["unit"] == "meV/A"
    assert verdict["threshold"] == 5.0

    code, _ = _run(
        CONV, str(data), "--quantity", "pressure", monkeypatch=monkeypatch, capsys=capsys
    )
    assert isinstance(code, str) and "no 'pressure'" in code

    # Rung 60 is within threshold but only the reference follows it: not confirmed.
    short = [
        {"value": 30, "energy": -10.0100},
        {"value": 40, "energy": -10.0050},
        {"value": 60, "energy": -10.0001},
        {"value": 70, "energy": -10.0000},
    ]
    data.write_text(json.dumps(short))
    code, out = _run(CONV, str(data), monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0 and "not converged" in out and "2 later rungs" in out


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
    # fcc nearest neighbors sit at a/sqrt(2) = 2.556 A, twelve of them.
    assert 2.45 < result["first_peak_r_A"] < 2.66
    assert result["first_peak_g"] > 3.0
    assert result["frames"] == 1
    assert abs(result["coordination_number"] - 12.0) < 1e-6
    assert 2.55 < result["first_minimum_r_A"] < 3.6  # the gap right after the shell
    assert result["first_peak_g_se"] is None  # one frame gives no block error


def test_rdf_normalises_same_species_pairs_to_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An ideal gas of 32 atoms: g must tend to 1, not (N-1)/N = 0.97."""
    from ase import Atoms

    rng = np.random.default_rng(3)
    frames = [
        Atoms("Ar32", positions=rng.uniform(0.0, 20.0, size=(32, 3)), cell=[20.0] * 3, pbc=True)
        for _ in range(60)
    ]
    data = tmp_path / "gas.xyz"
    ase_write(data, frames)
    code, out = _run(
        RDF, str(data), "--bins", "40", "--json", monkeypatch=monkeypatch, capsys=capsys
    )
    assert code == 0
    result = json.loads(out)
    assert abs(result["tail_mean_g"] - 1.0) < 0.03
    assert result["first_peak_g_se"] is not None
    assert not any("not 1" in w for w in result["warnings"])

    # The highest peak is not the first: a crystal with a taller second shell
    # still reports the nearest-neighbour peak first.
    code, out = _run(RDF, str(data), monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0 and "tail: g averages" in out


def test_rdf_large_cells_take_the_neighbour_list_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    atoms = bulk("Al", "fcc", a=4.046, cubic=True) * (8, 8, 8)  # 2048 atoms
    data = tmp_path / "big.xyz"
    ase_write(data, atoms)
    code, out = _run(
        RDF, str(data), "--rmax", "6.0", "--bins", "120", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    result = json.loads(out)
    assert abs(result["first_peak_r_A"] - 4.046 / np.sqrt(2)) < 0.06
    assert abs(result["coordination_number"] - 12.0) < 1e-6


def test_rdf_refuses_an_npt_frame_below_the_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    big = bulk("Cu", "fcc", a=3.615, cubic=True) * (3, 3, 3)
    small = big.copy()
    small.set_cell(big.cell * 0.5, scale_atoms=True)
    data = tmp_path / "npt.xyz"
    ase_write(data, [big, small])
    code, _ = _run(RDF, str(data), "--rmax", "5.0", monkeypatch=monkeypatch, capsys=capsys)
    assert isinstance(code, str) and "at frame 1" in code


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
    assert 0.8 * expected < result["d_A2_per_fs"] < 1.2 * expected
    assert result["atoms_averaged"] == n_atoms
    assert 0.9 < result["beta"] < 1.1
    assert result["d_se_cm2_per_s"] is not None and len(result["d_blocks_cm2_per_s"]) == 5
    assert result["d_se_cm2_per_s"] < 0.3 * result["d_cm2_per_s"]
    for axis in "xyz":
        assert 0.7 * expected < result["d_per_axis_A2_per_fs"][axis] < 1.3 * expected
    assert result["warnings"] == []

    # Two axes: the same D, from the x and y components alone.
    code, out = _run(
        MSD, str(data), "--dt-fs", "1.0", "--axes", "xy", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    planar = json.loads(out)
    assert 0.8 * expected < planar["d_A2_per_fs"] < 1.2 * expected

    # Yeh-Hummer adds a positive correction that scales with 1/L.
    code, out = _run(
        MSD, str(data), "--dt-fs", "1.0", "--yeh-hummer", "1e-3", "--temperature", "1000",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0 and "Yeh-Hummer D_inf" in out


def test_msd_flags_a_plateau_and_a_changing_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from ase import Atoms

    rng = np.random.default_rng(11)
    n_frames = 120
    # Atoms rattling about fixed sites: the MSD plateaus and beta is near 0.
    sites = rng.uniform(0.0, 10.0, size=(20, 3))
    frames = []
    for i in range(n_frames):
        cell = [10.0 + (0.01 if i > n_frames // 2 else 0.0)] * 3  # an NPT-like jump
        rattled = sites + rng.normal(0.0, 0.05, sites.shape)
        frames.append(Atoms("Ar20", positions=rattled, cell=cell))
    data = tmp_path / "solid.xyz"
    ase_write(data, frames)
    code, out = _run(
        MSD, str(data), "--dt-fs", "1.0", "--json", monkeypatch=monkeypatch, capsys=capsys
    )
    assert code == 0
    result = json.loads(out)
    assert result["beta"] < 0.3
    assert any("plateaued" in w for w in result["warnings"])
    assert any("cell changes between frames" in w for w in result["warnings"])


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
    code, _ = _run(
        MSD, str(data), "--dt-fs", "1", "--fit-to", "0.8", monkeypatch=monkeypatch, capsys=capsys
    )
    assert isinstance(code, str) and "0.5" in code
    code, _ = _run(
        MSD, str(data), "--dt-fs", "1", "--axes", "xq", monkeypatch=monkeypatch, capsys=capsys
    )
    assert isinstance(code, str) and "--axes" in code


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
        "lammps-potentials",
        "equation-of-state",
        "convergence-study",
        "radial-distribution",
        "msd-diffusion",
        "surface-energy",
        "elastic-constants",
        "melt-quench",
        "thermal-response",
        "kinetic-fits",
        "nemd-transport",
        "interface-adhesion",
        "nucleation-cnt",
        "two-phase-melting",
    } <= set(skills)
    assert skills["convergence-study"].agents == frozenset({"dft-expert"})
    assert skills["surface-energy"].agents == frozenset({"dft-expert"})
    assert skills["radial-distribution"].agents == frozenset({"md-expert", "analysis-expert"})
    assert skills["msd-diffusion"].agents == frozenset({"md-expert", "analysis-expert"})
    assert skills["elastic-constants"].agents == frozenset({"dft-expert", "analysis-expert"})
    assert skills["interface-adhesion"].agents == frozenset({"dft-expert", "analysis-expert"})
    assert skills["melt-quench"].agents == frozenset({"md-expert"})
    assert skills["two-phase-melting"].agents == frozenset({"md-expert"})
    assert skills["mlip-training"].agents == frozenset({"dft-expert", "md-expert"})
    assert skills["lammps-potentials"].agents == frozenset({"md-expert"})
    for analysis in ("thermal-response", "kinetic-fits", "nemd-transport", "nucleation-cnt"):
        assert skills[analysis].agents == frozenset({"md-expert", "analysis-expert"})
    # The scriptless skills are deliberate: scripts are optional in the format.
    assert not (skills["surface-energy"].root / "scripts").exists()
    assert not (skills["mlip-training"].root / "scripts").exists()
    assert (skills["two-phase-melting"].root / "scripts" / "interface_velocity.py").is_file()


# -- fit_rates ----------------------------------------------------------------

FIT_RATES = SKILLS / "kinetic-fits" / "scripts" / "fit_rates.py"
FIT_NEMD = SKILLS / "nemd-transport" / "scripts" / "fit_nemd.py"
FIT_ELASTIC = SKILLS / "elastic-constants" / "scripts" / "fit_elastic.py"
QUENCH_REPORT = SKILLS / "melt-quench" / "scripts" / "quench_report.py"
FIT_RAMP = SKILLS / "thermal-response" / "scripts" / "fit_thermal_ramp.py"
ADHESION = SKILLS / "interface-adhesion" / "scripts" / "adhesion.py"
CNT = SKILLS / "nucleation-cnt" / "scripts" / "cnt.py"

_KB_EV = 8.617333262e-5


def test_fit_rates_recovers_an_arrhenius_law_with_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "rates.json"
    rows = [
        {"T": t, "value": 1e-4 * np.exp(-0.5 / (_KB_EV * t))} for t in (500, 600, 700, 800, 900)
    ]
    data.write_text(json.dumps(rows))
    code, out = _run(
        FIT_RATES, str(data), "--mode", "arrhenius", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    fit = json.loads(out)
    assert abs(fit["ea_eV"] - 0.5) < 1e-6
    assert abs(fit["prefactor"] - 1e-4) / 1e-4 < 1e-4
    assert fit["r2"] > 0.999999
    assert fit["ea_se_eV"] is not None and fit["ea_se_eV"] < 1e-6  # exact data
    assert fit["ea_lnA_correlation"] > 0.9  # compensation: Ea up, ln A up
    assert fit["curvature"]["verdict"] == "arrhenius"
    assert abs(fit["t_ref_K"] - 1.0 / np.mean(1.0 / np.array([500, 600, 700, 800, 900]))) < 1e-9

    # Scattered replicas with errors: a weighted fit with a real error bar,
    # and the diffusion prefactor as an attempt frequency.
    rng = np.random.default_rng(5)
    rows = [
        {"T": t, "value": 1e-4 * np.exp(-0.5 / (_KB_EV * t)) * (1 + rng.normal(0, 0.1)),
         "err": 1e-5 * np.exp(-0.5 / (_KB_EV * t))}
        for t in (500, 500, 600, 600, 700, 700, 800, 800, 900, 900)
    ]
    data.write_text(json.dumps(rows))
    code, out = _run(
        FIT_RATES, str(data), "--mode", "arrhenius", "--jump-length", "2.5", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    fit = json.loads(out)
    assert fit["weighted"] and fit["points"] == 10 and fit["temperatures"] == 5
    assert abs(fit["ea_eV"] - 0.5) < 3 * fit["ea_se_eV"] + 0.02
    assert 0.005 < fit["ea_se_eV"] < 0.05
    # nu = 2 d D0 / a^2 with D0 in cm^2/s and a = 2.5 A = 2.5e-8 cm.
    expected_nu = 6 * fit["prefactor"] / (2.5e-8) ** 2
    assert abs(fit["attempt_frequency_per_s"] - expected_nu) / expected_nu < 1e-9

    code, out = _run(
        FIT_RATES, str(data), "--mode", "arrhenius", "--jump-length", "2.5",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0 and "+/-" in out and "attempt frequency" in out


def test_fit_rates_sees_curvature_and_fits_vft_and_myega(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "growth.json"
    rows = [{"T": t, "value": 12.0 * np.exp(-1500.0 / (t - 250.0))} for t in range(400, 900, 100)]
    data.write_text(json.dumps(rows))
    code, out = _run(
        FIT_RATES, str(data), "--mode", "vft", "--json", monkeypatch=monkeypatch, capsys=capsys
    )
    assert code == 0
    fit = json.loads(out)
    assert abs(fit["t0_K"] - 250.0) < 5.0
    assert abs(fit["b_K"] - 1500.0) / 1500.0 < 0.05
    assert abs(fit["prefactor"] - 12.0) / 12.0 < 0.15
    assert abs(fit["strength_D"] - 6.0) < 0.3
    assert fit["warnings"] == []

    # The same VFT data is curved on an Arrhenius plot, and the fit says so.
    code, out = _run(
        FIT_RATES, str(data), "--mode", "arrhenius", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    fit = json.loads(out)
    assert fit["curvature"]["verdict"] == "curved"
    assert any("curved" in w for w in fit["warnings"])

    # MYEGA recovers its own parameters.
    rows = [
        {"T": t, "value": 5.0 * np.exp(-(800.0 / t) * np.exp(600.0 / t))}
        for t in range(500, 1100, 100)
    ]
    data.write_text(json.dumps(rows))
    code, out = _run(
        FIT_RATES, str(data), "--mode", "myega", "--json", monkeypatch=monkeypatch, capsys=capsys
    )
    assert code == 0
    fit = json.loads(out)
    assert abs(fit["c_K"] - 600.0) < 15.0
    assert abs(fit["k_K"] - 800.0) / 800.0 < 0.05
    assert abs(fit["prefactor"] - 5.0) / 5.0 < 0.1


def test_fit_rates_locates_the_melting_crossing_with_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "velocity.json"
    rng = np.random.default_rng(2)
    rows = [
        {"T": t, "value": 0.004 * (823.0 - t) + rng.normal(0, 0.02)}
        for t in (700, 780, 800, 820, 850, 870, 940)
    ]
    data.write_text(json.dumps(rows))
    code, out = _run(
        FIT_RATES, str(data), "--mode", "crossing", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    fit = json.loads(out)
    assert len(fit["crossings"]) == 1
    assert fit["window_points"] == 5  # 780 to 870 within 60 K of the crossing
    assert abs(fit["t_m_K"] - 823.0) < 10.0
    assert fit["t_m_se_K"] is not None and 0.5 < fit["t_m_se_K"] < 15.0
    assert abs(fit["kinetic_coefficient_per_K"] - 0.004) / 0.004 < 0.3

    # Too few points near the crossing: the two-point fallback, and it says so.
    rows = [{"T": t, "value": 0.004 * (823.0 - t)} for t in (600, 800, 1000)]
    data.write_text(json.dumps(rows))
    code, out = _run(
        FIT_RATES, str(data), "--mode", "crossing", "--window", "50", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    fit = json.loads(out)
    assert abs(fit["t_m_K"] - 823.0) < 0.5 and fit["t_m_se_K"] is None
    assert any("no error bar" in w for w in fit["warnings"])


def test_fit_rates_refuses_what_it_cannot_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "bad.json"
    data.write_text(json.dumps([{"T": 500, "value": -1.0}, {"T": 600, "value": 2.0},
                                {"T": 700, "value": 3.0}]))
    code, _ = _run(
        FIT_RATES, str(data), "--mode", "arrhenius", monkeypatch=monkeypatch, capsys=capsys
    )
    assert isinstance(code, str) and "positive" in code

    data.write_text(json.dumps([{"T": 500, "value": 1.0}, {"T": 600, "value": 2.0}]))
    code, _ = _run(
        FIT_RATES, str(data), "--mode", "crossing", monkeypatch=monkeypatch, capsys=capsys
    )
    assert isinstance(code, str) and "never change sign" in code

    # Replicas at one temperature are kept, but two distinct temperatures are still needed.
    data.write_text(json.dumps([{"T": 500, "value": 1.0}, {"T": 500, "value": 2.0}]))
    code, _ = _run(
        FIT_RATES, str(data), "--mode", "arrhenius", monkeypatch=monkeypatch, capsys=capsys
    )
    assert isinstance(code, str) and "distinct temperatures" in code

    # Three points without errors: a fit, but no error bar, and it says so.
    rows = [{"T": t, "value": 1e-4 * np.exp(-0.5 / (_KB_EV * t))} for t in (500, 700, 900)]
    data.write_text(json.dumps(rows))
    code, out = _run(
        FIT_RATES, str(data), "--mode", "arrhenius", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    fit = json.loads(out)
    assert fit["ea_se_eV"] is None and any("no error bar" in w for w in fit["warnings"])


# -- fit_nemd -----------------------------------------------------------------


def test_fit_nemd_kappa_from_json_and_text_profiles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    xs = np.linspace(0.0, 100.0, 21)
    rows = [{"x": float(x), "T": 300.0 + 0.2 * float(x)} for x in xs]
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(rows))
    # gradient 0.2 K/A = 2e9 K/m under flux 3e9 W/m^2 -> k = 1.5 W/(m K)
    code, out = _run(
        FIT_NEMD, "kappa", str(profile), "--flux", "3e9", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    fit = json.loads(out)
    assert abs(fit["k_W_mK"] - 1.5) < 1e-6
    assert fit["k_se_W_mK"] < 1e-9 and abs(fit["mean_T_K"] - 310.0) < 1e-9
    assert fit["warnings"] == []

    text = tmp_path / "profile.dat"
    text.write_text("# x T\n" + "\n".join(f"{x} {300.0 + 0.2 * x}" for x in xs))
    code, out = _run(
        FIT_NEMD, "kappa", str(text), "--flux", "3e9", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    assert abs(json.loads(out)["k_W_mK"] - 1.5) < 1e-6


def test_fit_nemd_folds_a_sawtooth_and_windows_the_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A periodic Muller-Plathe profile: up on the first half, down on the
    # second, with curved ends where the exchange slabs sit.
    xs = np.linspace(0.0, 100.0, 41)
    rows = []
    for x in xs:
        base = 300.0 + 0.2 * x if x <= 50.0 else 300.0 + 0.2 * (100.0 - x)
        distance_to_slab = min(x, abs(x - 50.0), 100.0 - x)
        bend = 3.0 if distance_to_slab < 6.0 else 0.0
        rows.append({"x": float(x), "T": base - bend})
    profile = tmp_path / "sawtooth.json"
    profile.write_text(json.dumps(rows))

    # Unfolded, the fit is meaningless and the script says so.
    code, out = _run(
        FIT_NEMD, "kappa", str(profile), "--flux", "3e9", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0 and any("not linear" in w for w in json.loads(out)["warnings"])

    code, out = _run(
        FIT_NEMD, "kappa", str(profile), "--flux", "3e9", "--fold", "--drop-ends", "3", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    fit = json.loads(out)
    assert fit["folded"] and abs(fit["k_W_mK"] - 1.5) < 1e-6
    assert fit["half_mismatch"] < 1e-6 and fit["warnings"] == []
    assert fit["halves"][0]["gradient_K_m"] > 0 > fit["halves"][1]["gradient_K_m"]

    # A window by position gives the same answer as dropping bins.
    code, out = _run(
        FIT_NEMD, "kappa", str(profile), "--flux", "3e9", "--xmin", "8", "--xmax", "42",
        "--json", monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0 and abs(json.loads(out)["k_W_mK"] - 1.5) < 1e-6

    # Time blocks give a mean and a standard error.
    block2 = tmp_path / "block2.json"
    block2.write_text(json.dumps([{"x": r["x"], "T": r["T"] * 1.02} for r in rows]))
    code, out = _run(
        FIT_NEMD, "kappa", str(profile), str(block2), "--flux", "3e9", "--fold",
        "--drop-ends", "3", "--json", monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    fit = json.loads(out)
    assert len(fit["blocks"]) == 2 and fit["k_W_mK_block_se"] > 0

    code, out = _run(
        FIT_NEMD, "kappa", str(profile), "--flux", "3e9", "--fold", "--drop-ends", "3",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0 and "half-profile mismatch" in out and "<T> =" in out


def test_fit_nemd_tbr_measures_the_interface_jump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [{"x": float(x), "T": 400.0 - 0.1 * float(x)} for x in np.linspace(0.0, 48.0, 13)]
    rows += [
        {"x": float(x), "T": 365.0 - 0.1 * (float(x) - 52.0)} for x in np.linspace(52.0, 100.0, 13)
    ]
    # bend the two bins next to the interface, as real profiles do
    rows[11]["T"] -= 2.0
    rows[12]["T"] -= 4.0
    rows[13]["T"] += 4.0
    rows[14]["T"] += 2.0
    profile = tmp_path / "tbr.json"
    profile.write_text(json.dumps(rows))
    # both branches extrapolate to the interface with a 30 K jump; R = 30/3e9 = 1e-8
    code, out = _run(
        FIT_NEMD, "tbr", str(profile), "--flux", "3e9", "--interface", "50",
        "--exclude-interface", "9", "--json", monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    fit = json.loads(out)
    assert abs(fit["tbr_m2K_W"] - 1e-8) / 1e-8 < 0.05
    assert abs(fit["conductance_MW_m2K"] - 100.0) < 5.0
    assert abs(fit["left"]["k_W_mK"] - 3.0) < 1e-6
    assert fit["left"]["points"] == 11 and fit["warnings"] == []

    # Without the exclusion the bent bins enter the fit, and the script says so.
    code, out = _run(
        FIT_NEMD, "tbr", str(profile), "--flux", "3e9", "--interface", "50", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0 and any("no points were excluded" in w for w in json.loads(out)["warnings"])


def test_fit_nemd_refuses_bad_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps([{"x": float(x), "T": 300.0 + x} for x in range(10)]))
    code, _ = _run(
        FIT_NEMD, "tbr", str(profile), "--flux", "1e9", "--interface", "50",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert isinstance(code, str) and "outside the profile" in code

    code, _ = _run(
        FIT_NEMD, "tbr", str(profile), "--flux", "1e9", "--interface", "1.5",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert isinstance(code, str) and "at least 3" in code

    code, _ = _run(
        FIT_NEMD, "kappa", str(profile), "--flux", "1e9", "--drop-ends", "5",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert isinstance(code, str) and "removes every point" in code

    bad = tmp_path / "bad.dat"
    bad.write_text("1 2 3\n")
    code, _ = _run(
        FIT_NEMD, "kappa", str(bad), "--flux", "1e9", monkeypatch=monkeypatch, capsys=capsys
    )
    assert isinstance(code, str) and "two columns" in code


# -- fit_elastic --------------------------------------------------------------

_GPA = 160.21766208


def _ortho_ladders() -> dict[str, object]:
    cij = {"C11": 100.0, "C22": 120.0, "C33": 140.0, "C44": 30.0, "C55": 40.0,
           "C66": 50.0, "C12": 40.0, "C13": 30.0, "C23": 20.0}
    combos = {
        "e1": cij["C11"], "e2": cij["C22"], "e3": cij["C33"],
        "e4": cij["C44"], "e5": cij["C55"], "e6": cij["C66"],
        "e12": cij["C11"] + cij["C22"] + 2 * cij["C12"],
        "e13": cij["C11"] + cij["C33"] + 2 * cij["C13"],
        "e23": cij["C22"] + cij["C33"] + 2 * cij["C23"],
    }
    v0 = 200.0
    modes = {
        mode: [
            {"delta": d, "energy": -50.0 + 0.5 * (c / _GPA) * v0 * d * d}
            for d in (-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02)
        ]
        for mode, c in combos.items()
    }
    return {"symmetry": "orthorhombic", "v0": v0, "modes": modes}


def test_fit_elastic_recovers_a_known_orthorhombic_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "elastic.json"
    data.write_text(json.dumps(_ortho_ladders()))
    code, out = _run(FIT_ELASTIC, str(data), "--json", monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0
    fit = json.loads(out)
    for name, expected in (("C11", 100.0), ("C33", 140.0), ("C12", 40.0), ("C23", 20.0)):
        assert abs(fit["cij_GPa"][name] - expected) < 1e-6
    assert abs(fit["b_voigt_GPa"] - 60.0) < 1e-6
    assert fit["b_reuss_GPa"] < fit["b_voigt_GPa"] + 1e-9
    assert 0.0 < fit["poisson"] < 0.5
    assert fit["warnings"] == []
    # Exact quadratics: the quartic and inner-window fits agree to rounding.
    assert fit["relative_uncertainty"] < 1e-6
    assert all(margin > 0.1 for margin in fit["born_margins"].values())

    hexagonal = dict(_ortho_ladders(), symmetry="hexagonal")
    data.write_text(json.dumps(hexagonal))
    code, out = _run(FIT_ELASTIC, str(data), "--json", monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0 and json.loads(out)["symmetry"] == "hexagonal"


def test_fit_elastic_reports_spread_born_margins_and_isotropic_averages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    v0 = 100.0

    def ladder(c_gpa: float, quartic: float = 0.0) -> list[dict[str, float]]:
        return [
            {"delta": d, "energy": 0.5 * (c_gpa / _GPA) * v0 * d * d + quartic * d**4}
            for d in (-0.02, -0.01, -0.005, 0.0, 0.005, 0.01, 0.02)
        ]

    # Isotropic: six ladders with different curvatures average to the mean.
    modes = {"e1": ladder(90.0), "e2": ladder(100.0), "e3": ladder(110.0),
             "e4": ladder(25.0), "e5": ladder(30.0), "e6": ladder(35.0)}
    data = tmp_path / "glass.json"
    data.write_text(json.dumps({"symmetry": "isotropic", "v0": v0, "modes": modes}))
    code, out = _run(FIT_ELASTIC, str(data), "--json", monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0
    fit = json.loads(out)
    assert abs(fit["cij_GPa"]["C11"] - 100.0) < 1e-6
    assert abs(fit["cij_GPa"]["C44"] - 30.0) < 1e-6
    assert fit["warnings"] == []

    # Only e1 and e4: still fits, and says the other directions are missing.
    data.write_text(json.dumps({"symmetry": "isotropic", "v0": v0,
                                "modes": {"e1": modes["e1"], "e4": modes["e4"]}}))
    code, out = _run(FIT_ELASTIC, str(data), "--json", monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0 and any("e2, e3, e5" in w for w in json.loads(out)["warnings"])

    # A strong quartic term makes the three fits disagree: the spread warns.
    noisy = {"e1": ladder(100.0, quartic=40000.0), "e4": ladder(30.0),
             "e12": ladder(2 * 100.0 + 2 * 40.0)}
    data.write_text(json.dumps({"symmetry": "cubic", "v0": v0, "modes": noisy}))
    code, out = _run(FIT_ELASTIC, str(data), "--json", monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0
    fit = json.loads(out)
    assert fit["relative_uncertainty"] > 0.15
    assert any("differ by" in w for w in fit["warnings"])

    # C12 above C11 fails the Born condition C11 - C12 > 0.
    unstable = {"e1": ladder(100.0), "e4": ladder(30.0), "e12": ladder(2 * 100.0 + 2 * 120.0)}
    data.write_text(json.dumps({"symmetry": "cubic", "v0": v0, "modes": unstable}))
    code, out = _run(FIT_ELASTIC, str(data), monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0
    assert "Born stability fails for C11-C12" in out
    assert "not positive definite" in out
    assert "precision:" in out


def test_fit_elastic_refuses_and_warns_well(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _ortho_ladders()
    assert isinstance(payload["modes"], dict)
    del payload["modes"]["e23"]
    data = tmp_path / "elastic.json"
    data.write_text(json.dumps(payload))
    code, _ = _run(FIT_ELASTIC, str(data), monkeypatch=monkeypatch, capsys=capsys)
    assert isinstance(code, str) and "e23" in code

    data.write_text(json.dumps({"symmetry": "hexagonal", "v0": 10.0, "modes": {}}))
    code, _ = _run(FIT_ELASTIC, str(data), monkeypatch=monkeypatch, capsys=capsys)
    assert isinstance(code, str) and "symmetry" in code

    thin = {
        "symmetry": "isotropic", "v0": 10.0,
        "modes": {"e1": [{"delta": 0.0, "energy": 0.0}], "e4": [{"delta": 0.0, "energy": 0.0}]},
    }
    data.write_text(json.dumps(thin))
    code, _ = _run(FIT_ELASTIC, str(data), monkeypatch=monkeypatch, capsys=capsys)
    assert isinstance(code, str) and "at least 4" in code

    # An off-minimum reference leaves a linear term, and the fit says so.
    strained = _ortho_ladders()
    assert isinstance(strained["modes"], dict)
    for row in strained["modes"]["e1"]:
        row["energy"] += 0.5 * row["delta"]
    data.write_text(json.dumps(strained))
    code, out = _run(FIT_ELASTIC, str(data), monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0
    assert "not at its energy minimum" in out


# -- quench_report ------------------------------------------------------------


def test_quench_report_tail_density_and_densification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from ase import Atoms

    frames = []
    for i in range(20):
        a = 12.0 - 0.1 * min(i, 10)  # the cell shrinks, then plateaus
        frames.append(
            Atoms("Cu8", positions=np.random.default_rng(i).uniform(0, a, (8, 3)),
                  cell=[a] * 3, pbc=True)
        )
    data = tmp_path / "quench-100Kps-r1.traj"
    ase_write(data, frames)
    expected = 8 * 63.546 * 1.66053906660 / 11.0**3
    code, out = _run(
        QUENCH_REPORT, str(data), "--rho-c", "9.0", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    report = json.loads(out)["reports"][0]
    assert abs(report["rho_g_cm3"] - expected) / expected < 1e-6
    assert abs(report["delta_v"] - (1.0 - expected / 9.0)) < 1e-6
    assert report["rate_K_per_ps"] == 100.0 and report["replica"] == 1
    assert any("not a declared hold" in w for w in report["warnings"])

    # A declared hold of 10 frames: exact plateau, no drift, no warning.
    code, out = _run(
        QUENCH_REPORT, str(data), "--hold-frames", "10", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    report = json.loads(out)["reports"][0]
    assert report["hold_frames"] == 10 and report["warnings"] == []
    assert report["rho_se_g_cm3"] == 0.0

    # A hold that still drifts is flagged.
    code, out = _run(
        QUENCH_REPORT, str(data), "--hold-frames", "16", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    assert any("has not settled" in w for w in json.loads(out)["reports"][0]["warnings"])


def test_quench_report_groups_replicas_and_fits_the_log_rate_law(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from ase import Atoms

    def trajectory(name: str, a: float) -> Path:
        frames = [
            Atoms("Cu8", positions=np.random.default_rng(i).uniform(0, a, (8, 3)),
                  cell=[a] * 3, pbc=True)
            for i in range(8)
        ]
        path = tmp_path / name
        ase_write(path, frames)
        return path

    # Density 8 m / a^3: a slower quench packs denser here (smaller a).
    files = [
        trajectory("quench-100Kps-r1.traj", 11.0),
        trajectory("quench-100Kps-r2.traj", 11.02),
        trajectory("quench-10Kps-r1.traj", 10.9),
        trajectory("quench-1Kps-r1.traj", 10.8),
    ]
    code, out = _run(
        QUENCH_REPORT, *(str(f) for f in files), "--hold-frames", "8", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    result = json.loads(out)
    rates = {r["rate_K_per_ps"]: r for r in result["rates"]}
    assert rates[100.0]["replicas"] == 2 and rates[100.0]["rho_spread_g_cm3"] > 0
    assert rates[1.0]["replicas"] == 1 and rates[1.0]["rho_spread_g_cm3"] is None
    assert rates[1.0]["rho_g_cm3"] > rates[10.0]["rho_g_cm3"] > rates[100.0]["rho_g_cm3"]
    assert result["log_rate_law"]["slope_per_decade_g_cm3"] < 0

    code, out = _run(
        QUENCH_REPORT, *(str(f) for f in files), "--hold-frames", "8",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0 and "over 2 replica(s)" in out and "log10(rate" in out


def test_quench_report_refuses_thin_or_unbounded_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from ase import Atoms

    short = tmp_path / "short.traj"
    ase_write(short, [Atoms("Cu", positions=[[0, 0, 0]], cell=[10] * 3, pbc=True)] * 2)
    code, _ = _run(QUENCH_REPORT, str(short), monkeypatch=monkeypatch, capsys=capsys)
    assert isinstance(code, str) and "at least 4" in code

    boxless = tmp_path / "boxless.xyz"
    ase_write(boxless, [Atoms("Cu", positions=[[0, 0, 0]])] * 5)
    code, _ = _run(QUENCH_REPORT, str(boxless), monkeypatch=monkeypatch, capsys=capsys)
    assert isinstance(code, str) and "volume" in code

    code, _ = _run(
        QUENCH_REPORT, str(short), "--tail", "1.5", monkeypatch=monkeypatch, capsys=capsys
    )
    assert isinstance(code, str) and "--tail" in code

    five = tmp_path / "five.traj"
    ase_write(five, [Atoms("Cu", positions=[[0, 0, 0]], cell=[10] * 3, pbc=True)] * 5)
    code, _ = _run(
        QUENCH_REPORT, str(five), "--hold-frames", "9", monkeypatch=monkeypatch, capsys=capsys
    )
    assert isinstance(code, str) and "fewer than the 9 hold frames" in code


# -- fit_thermal_ramp ---------------------------------------------------------


def test_fit_thermal_ramp_slopes_and_latent_heat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    crystal = tmp_path / "crystal.json"
    liquid = tmp_path / "liquid.json"
    crystal.write_text(
        json.dumps([{"T": t, "H": -100.0 + 0.01 * t, "V": 500.0 + 0.05 * t, "N": 32,
                     "mass_amu": 32 * 63.546, "L": [7.94 + 0.0004 * t] * 3}
                    for t in range(300, 800, 100)])
    )
    liquid.write_text(
        json.dumps([{"T": t, "H": -80.0 + 0.012 * t, "V": 520.0 + 0.05 * t, "N": 32}
                    for t in range(300, 800, 100)])
    )
    code, out = _run(FIT_RAMP, str(crystal), "--json", monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0
    fit = json.loads(out)
    assert abs(fit["cp_total_eV_K"] - 0.01) < 1e-9
    assert fit["cp_total_se_eV_K"] < 1e-9  # exact line, five rungs
    # 0.01 eV/K over ~525 A^3 -> 3.05e6 J/(m^3 K); CTE = 0.05/(3*525)
    assert abs(fit["cp_vol_J_m3K"] - 3.05e6) / 3.05e6 < 0.01
    assert abs(fit["cte_per_K"] - 0.05 / (3 * 525.0)) / (0.05 / (3 * 525.0)) < 0.01
    # 0.01 eV/K over 32 atoms is 3.63 k_B per atom, 30.2 J/(mol K).
    assert abs(fit["cp_kB_per_atom"] - 0.01 / (32 * _KB_EV)) < 1e-6
    assert abs(fit["cp_J_molK"] - 30.15) < 0.1
    assert abs(fit["cp_J_kgK"] - 474.4) < 1.0
    assert len(fit["cte_per_axis_per_K"]) == 3
    assert fit["warnings"] == []

    code, out = _run(
        FIT_RAMP, str(crystal), "--other", str(liquid), "--at", "500", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    fit = json.loads(out)
    assert abs(fit["dH_eV"] - 21.0) < 1e-9  # (-80+6) - (-100+5)
    assert abs(fit["dH_eV_per_atom"] - 21.0 / 32) < 1e-9
    assert abs(fit["dH_kJ_mol"] - 21.0 / 32 * 96.485) < 0.01

    code, out = _run(FIT_RAMP, str(crystal), monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0 and "k_B/atom" in out and "CTE per axis" in out


def test_fit_thermal_ramp_uses_measured_temperatures_and_sees_hysteresis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ramp = tmp_path / "ramp.json"
    # The measured temperatures run 10 % hot; the slope against them is 0.01/1.1.
    rows = [
        {"T": t, "T_measured": 1.1 * t, "H": -100.0 + 0.01 * 1.1 * t, "V": 500.0, "N": 8,
         "H_se": 0.001, "direction": "up"}
        for t in range(300, 800, 100)
    ]
    rows += [
        {"T": t, "T_measured": 1.1 * t, "H": -100.0 + 0.01 * 1.1 * t + 0.5, "V": 500.0,
         "N": 8, "H_se": 0.001, "direction": "down"}
        for t in (600, 500, 400, 300)
    ]
    ramp.write_text(json.dumps(rows))
    code, out = _run(FIT_RAMP, str(ramp), "--json", monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0
    fit = json.loads(out)
    assert fit["fitted_temperature"] == "measured"
    assert abs(fit["cp_total_eV_K"] - 0.01) < 1e-9
    assert len(fit["hysteresis"]) == 4
    assert any("hysteresis" in w for w in fit["warnings"])

    # Per-axis expansion that differs between axes is flagged.
    rows = [
        {"T": t, "H": 0.01 * t, "V": 500.0 * (1 + 3e-5 * (t - 300)), "N": 8,
         "L": [7.9 * (1 + 5e-5 * (t - 300)), 7.9, 8.0 * (1 + 4e-5 * (t - 300))]}
        for t in range(300, 800, 100)
    ]
    ramp.write_text(json.dumps(rows))
    code, out = _run(FIT_RAMP, str(ramp), "--json", monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0 and any("between axes" in w for w in json.loads(out)["warnings"])


def test_fit_thermal_ramp_refuses_extrapolation_thin_windows_and_unequal_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ramp = tmp_path / "ramp.json"
    ramp.write_text(
        json.dumps([{"T": t, "E": 0.01 * t, "V": 500.0, "N": 8} for t in range(300, 800, 100)])
    )
    code, _ = _run(
        FIT_RAMP, str(ramp), "--other", str(ramp), "--at", "900",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert isinstance(code, str) and "outside" in code

    code, _ = _run(
        FIT_RAMP, str(ramp), "--window", "300", "310", monkeypatch=monkeypatch, capsys=capsys
    )
    assert isinstance(code, str) and "at least 3" in code

    code, _ = _run(FIT_RAMP, str(ramp), "--at", "500", monkeypatch=monkeypatch, capsys=capsys)
    assert isinstance(code, str) and "--other and --at" in code

    bigger = tmp_path / "bigger.json"
    bigger.write_text(
        json.dumps([{"T": t, "E": 0.03 * t, "V": 1500.0, "N": 24} for t in range(300, 800, 100)])
    )
    code, _ = _run(
        FIT_RAMP, str(ramp), "--other", str(bigger), "--at", "500",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert isinstance(code, str) and "8 atoms" in code and "24" in code

    # Rows without N still give a per-cell slope, and say so.
    ramp.write_text(json.dumps([{"T": t, "E": 0.01 * t, "V": 500.0} for t in range(300, 700, 100)]))
    code, out = _run(FIT_RAMP, str(ramp), "--json", monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0
    fit = json.loads(out)
    assert "cp_kB_per_atom" not in fit
    assert any("no atom count" in w for w in fit["warnings"])
    assert any("at least 5" in w for w in fit["warnings"])


# -- adhesion -----------------------------------------------------------------


def test_adhesion_reports_w_adh_gamma_int_and_the_potency_factor(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # W_adh = 10 eV over 100 A^2 = 0.1 eV/A^2 = 1.6022 J/m^2. With surface
    # energies 1.0 + 1.0 the interface energy is 0.3978; a substrate-liquid
    # energy equal to it puts cos(theta) at 0: theta 90 deg, f = 0.5 exactly.
    code, out = _run(
        ADHESION, "--e-interface", "-110", "--e-a", "-60", "--e-b", "-40",
        "--area", "100", "--gamma-a", "1.0", "--gamma-b", "1.0",
        "--gamma-nl", str(2.0 - 1.6021766208), "--gamma-sl", "0.25", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    fit = json.loads(out)
    assert fit["quantity"] == "work_of_adhesion"
    assert abs(fit["w_adh_J_m2"] - 1.6021766208) < 1e-9
    assert abs(fit["gamma_int_J_m2"] - (2.0 - 1.6021766208)) < 1e-9
    assert abs(fit["cos_theta"]) < 1e-9
    assert abs(fit["f_het"] - 0.5) < 1e-9
    assert fit["warnings"] == []

    # Strain energies per area from the free-lattice slab energies, and the
    # work-of-separation name for frozen references.
    code, out = _run(
        ADHESION, "--e-interface", "-110", "--e-a", "-60", "--e-b", "-40",
        "--area", "100", "--e-a-free", "-61", "--e-b-free", "-40",
        "--frozen-references", monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    assert "W_sep =" in out
    assert "strain energy of slab a: 0.1602 J/m^2" in out
    assert "strain energy of slab b: 0.0000 J/m^2" in out


def test_adhesion_clamps_complete_wetting_and_flags_repulsion(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # gamma_NS = 0.3978 is far below gamma_NL - gamma_SL = 2.0 - 0.25.
    code, out = _run(
        ADHESION, "--e-interface", "-110", "--e-a", "-60", "--e-b", "-40",
        "--area", "100", "--gamma-a", "1.0", "--gamma-b", "1.0",
        "--gamma-nl", "2.0", "--gamma-sl", "0.25", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    fit = json.loads(out)
    assert fit["f_het"] == 0.0
    assert any("complete wetting" in w for w in fit["warnings"])

    code, out = _run(
        ADHESION, "--e-interface", "-90", "--e-a", "-60", "--e-b", "-40",
        "--area", "100", "--json", monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    fit = json.loads(out)
    assert fit["w_adh_J_m2"] < 0
    assert any("do not bind" in w for w in fit["warnings"])

    # 100 eV over 100 A^2 is 160 J/m^2: no real interface binds like that.
    code, out = _run(
        ADHESION, "--e-interface", "-200", "--e-a", "-60", "--e-b", "-40",
        "--area", "100", "--json", monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0 and any("outside the range" in w for w in json.loads(out)["warnings"])

    code, _ = _run(
        ADHESION, "--e-interface", "-1", "--e-a", "-1", "--e-b", "-1", "--area", "0",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert isinstance(code, str) and "--area" in code

    # The wetting angle needs the interface energy, so the surface energies.
    code, _ = _run(
        ADHESION, "--e-interface", "-110", "--e-a", "-60", "--e-b", "-40",
        "--area", "100", "--gamma-nl", "0.4", "--gamma-sl", "0.25",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert isinstance(code, str) and "--gamma-a" in code


# -- cnt ----------------------------------------------------------------------


def test_cnt_gamma_inverts_gibbs_thomson_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tm, dhf, gamma = 823.0, 2.4e8, 0.06
    rows = [
        {"T": t, "r_star_nm": 2 * gamma * tm / (dhf * (tm - t)) * 1e9,
         "r_low_nm": 0.9 * 2 * gamma * tm / (dhf * (tm - t)) * 1e9,
         "r_high_nm": 1.1 * 2 * gamma * tm / (dhf * (tm - t)) * 1e9}
        for t in (700.0, 750.0, 800.0)
    ]
    data = tmp_path / "rstar.json"
    data.write_text(json.dumps(rows))
    code, out = _run(
        CNT, "gamma", str(data), "--tm", str(tm), "--dhf", str(dhf), "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    fit = json.loads(out)
    assert abs(fit["gamma_J_m2"] - gamma) < 1e-9
    assert fit["gamma_std_J_m2"] < 1e-12
    assert abs(fit["gamma_at_tm_J_m2"] - gamma) < 1e-9  # a flat gamma(T) extrapolates flat
    assert abs(fit["dgamma_dT_J_m2K"]) < 1e-12
    assert abs(fit["points"][0]["gamma_se_J_m2"] - 0.1 * gamma) < 1e-9  # bracket half-width
    assert fit["warnings"] == []


def test_cnt_gamma_from_cluster_counts_and_its_temperature_law(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import math

    tm, dhf, omega = 823.0, 2.4e8, 30.5
    rho_s = 1.0 / (omega * 1e-30)
    # gamma rising toward Tm at 1e-4 J/m^2/K, as seeding finds.
    rows = []
    for t in (700.0, 750.0, 800.0, 810.0):
        gamma = 0.06 + 1e-4 * (t - tm)
        dgv = dhf * (tm - t) / tm
        r_star = 2 * gamma / dgv
        n_star = (4 / 3) * math.pi * r_star**3 * rho_s
        rows.append({"T": t, "n_star": n_star, "n_low": 0.9 * n_star, "n_high": 1.1 * n_star,
                     "order_parameter": "q6 > 0.33, rc 3.2 A"})
    data = tmp_path / "seeds.json"
    data.write_text(json.dumps(rows))
    code, out = _run(
        CNT, "gamma", str(data), "--tm", str(tm), "--dhf", str(dhf), "--omega", str(omega),
        "--json", monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    fit = json.loads(out)
    assert abs(fit["gamma_at_tm_J_m2"] - 0.06) < 1e-9
    assert abs(fit["dgamma_dT_J_m2K"] - 1e-4) < 1e-12
    assert fit["gamma_at_tm_se_J_m2"] < 1e-9
    assert fit["order_parameter"] == "q6 > 0.33, rc 3.2 A"
    assert fit["warnings"] == []  # all within 20 % undercooling, no small nuclei

    # Without --omega the atom counts cannot be converted.
    code, _ = _run(
        CNT, "gamma", str(data), "--tm", str(tm), "--dhf", str(dhf),
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert isinstance(code, str) and "--omega" in code

    # One deep undercooling with a tiny nucleus and no criterion: three warnings.
    data.write_text(json.dumps([{"T": 500.0, "n_star": 20.0}]))
    code, out = _run(
        CNT, "gamma", str(data), "--tm", str(tm), "--dhf", str(dhf), "--omega", str(omega),
        "--json", monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    warnings = json.loads(out)["warnings"]
    assert any("overestimates" in w for w in warnings)
    assert any("few shells" in w for w in warnings)
    assert any("order_parameter" in w for w in warnings)
    assert any("one undercooling" in w for w in warnings)


def test_cnt_barrier_matches_the_closed_form(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import math

    tm, dhf, gamma, t = 823.0, 2.4e8, 0.06, 700.0
    code, out = _run(
        CNT, "barrier", "--gamma", str(gamma), "--tm", str(tm), "--dhf", str(dhf),
        "--temps", str(t), "--omega", "30.5", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    row = json.loads(out)["rows"][0]
    dgv = dhf * (tm - t) / tm
    assert abs(row["r_star_nm"] - 2 * gamma / dgv * 1e9) < 1e-9
    expected_ev = 16 * math.pi * gamma**3 / (3 * dgv**2) / 1.602176634e-19
    assert abs(row["dG_star_eV"] - expected_ev) / expected_ev < 1e-9
    expected_n = (4 / 3) * math.pi * (2 * gamma / dgv) ** 3 / (30.5e-30)
    assert abs(row["n_star_atoms"] - expected_n) / expected_n < 1e-9

    # gamma(T) from a slope: at 700 K with slope 1e-4 from Tm, gamma is 0.0477.
    code, out = _run(
        CNT, "barrier", "--gamma", str(gamma), "--gamma-slope", "1e-4", "--tm", str(tm),
        "--dhf", str(dhf), "--temps", str(t), "--json", monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    row = json.loads(out)["rows"][0]
    assert abs(row["gamma_J_m2"] - (gamma - 1e-4 * 123.0)) < 1e-9

    # theta = 90 deg is f = 0.5: barrier and cap atom count halve.
    code, out = _run(
        CNT, "barrier", "--gamma", str(gamma), "--tm", str(tm), "--dhf", str(dhf),
        "--temps", str(t), "--omega", "30.5", "--theta", "90", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    result = json.loads(out)
    assert abs(result["f_het"] - 0.5) < 1e-12
    assert abs(result["rows"][0]["dG_star_eV"] - expected_ev / 2) / expected_ev < 1e-9
    assert abs(result["rows"][0]["n_star_atoms"] - expected_n / 2) / expected_n < 1e-9


def test_cnt_rate_and_the_driving_force_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import math

    tm, dhf, gamma, t, omega = 823.0, 2.4e8, 0.06, 700.0, 30.5
    code, out = _run(
        CNT, "rate", "--gamma", str(gamma), "--tm", str(tm), "--dhf", str(dhf),
        "--temps", str(t), "--omega", str(omega), "--attachment", "2e11", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    row = json.loads(out)["rows"][0]
    rho_s = 1 / (omega * 1e-30)
    dgv = dhf * (tm - t) / tm
    n_star = (4 / 3) * math.pi * (2 * gamma / dgv) ** 3 * rho_s
    z = math.sqrt((dgv / rho_s) / (6 * math.pi * 1.380649e-23 * t * n_star))
    assert abs(row["zeldovich"] - z) / z < 1e-9
    barrier_j = 16 * math.pi * gamma**3 / (3 * dgv**2)
    j = rho_s * z * 2e11 * math.exp(-barrier_j / (1.380649e-23 * t))
    assert abs(row["rate_per_m3_s"] - j) / j < 1e-6

    # A driving-force table overrides the linear form, and no undercooling warning fires.
    table = tmp_path / "dmu.json"
    table.write_text(json.dumps([{"T": 500.0, "dgv_J_m3": 5e7}, {"T": 823.0, "dgv_J_m3": 0.0}]))
    code, out = _run(
        CNT, "barrier", "--gamma", str(gamma), "--tm", str(tm), "--dhf", str(dhf),
        "--temps", "600", "--dmu-table", str(table), "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    result = json.loads(out)
    expected_dgv = 5e7 * (823.0 - 600.0) / (823.0 - 500.0)
    assert abs(result["rows"][0]["dGv_J_m3"] - expected_dgv) / expected_dgv < 1e-9
    assert result["warnings"] == []

    # The same temperature under the linear form is 27 % undercooling: a warning.
    code, out = _run(
        CNT, "barrier", "--gamma", str(gamma), "--tm", str(tm), "--dhf", str(dhf),
        "--temps", "600", monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0 and "overestimates the driving force" in out


def test_cnt_refuses_the_wrong_regime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "rstar.json"
    data.write_text(json.dumps([{"T": 900.0, "r_star_nm": 2.0}]))
    code, _ = _run(
        CNT, "gamma", str(data), "--tm", "823", "--dhf", "2.4e8",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert isinstance(code, str) and "not below Tm" in code

    code, _ = _run(
        CNT, "barrier", "--gamma", "0.06", "--tm", "823", "--dhf", "2.4e8",
        "--temps", "600", "--f-het", "1.5", monkeypatch=monkeypatch, capsys=capsys,
    )
    assert isinstance(code, str) and "--f-het" in code

    code, _ = _run(
        CNT, "rate", "--gamma", "0.06", "--tm", "823", "--dhf", "2.4e8",
        "--temps", "700", monkeypatch=monkeypatch, capsys=capsys,
    )
    assert isinstance(code, str) and "--omega" in code


# -- interface_velocity (two-phase-melting) -----------------------------------

INTERFACE = SKILLS / "two-phase-melting" / "scripts" / "interface_velocity.py"


def _coexistence_frames(n_frames: int, growth_per_frame: float, seed: int = 1) -> list[Any]:
    """A 4x4x16 fcc Cu box whose crystal slab (z below z_c) grows each frame;
    the rest is random at the same density, the liquid stand-in."""
    from ase import Atoms

    rng = np.random.default_rng(seed)
    crystal = bulk("Cu", "fcc", a=3.615, cubic=True) * (4, 4, 16)
    cell = crystal.cell.lengths()
    frames = []
    for k in range(n_frames):
        z_c = 0.35 * cell[2] + growth_per_frame * k
        positions = crystal.get_positions().copy()
        liquid = positions[:, 2] >= z_c
        positions[liquid, 0] = rng.uniform(0.0, cell[0], liquid.sum())
        positions[liquid, 1] = rng.uniform(0.0, cell[1], liquid.sum())
        positions[liquid, 2] = rng.uniform(z_c, cell[2], liquid.sum())
        frames.append(Atoms("Cu" * len(crystal), positions=positions, cell=crystal.cell, pbc=True))
    return frames


def test_interface_velocity_separates_crystal_from_liquid_and_fits_the_slope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    frames = _coexistence_frames(n_frames=8, growth_per_frame=1.0)
    data = tmp_path / "coex-r1.traj"
    ase_write(data, frames)
    code, out = _run(
        INTERFACE, str(data), "--dt-fs", "1000", "--interfaces", "1", "--fit-from", "0",
        "--fit-to", "1", "--cutoff", "3.1", "--json", monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    result = json.loads(out)
    replica = result["replicas"][0]
    # The crystal slab starts at 35 % of the box and grows 1 A per frame (1 ps).
    assert 0.25 < replica["fraction_first"] < 0.45
    assert replica["fraction_last"] > replica["fraction_first"]
    assert abs(result["v_A_per_ps"] - 1.0) < 0.3
    assert abs(result["v_m_per_s"] - 100.0) < 30.0
    assert replica["r2"] > 0.9
    assert any("one trajectory" in w for w in result["warnings"])

    # Two interfaces halve the velocity; two replicas give a spread.
    other = tmp_path / "coex-r2.traj"
    ase_write(other, _coexistence_frames(n_frames=8, growth_per_frame=1.0, seed=2))
    code, out = _run(
        INTERFACE, str(data), str(other), "--dt-fs", "1000", "--fit-from", "0", "--fit-to", "1",
        "--cutoff", "3.1", monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    assert "2 interfaces" in out and "+/-" in out


def test_interface_velocity_classifies_a_perfect_crystal_and_refuses_bad_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    crystal = bulk("Cu", "fcc", a=3.615, cubic=True) * (3, 3, 3)
    data = tmp_path / "crystal.traj"
    ase_write(data, [crystal] * 5)
    code, out = _run(
        INTERFACE, str(data), "--dt-fs", "100", "--json", monkeypatch=monkeypatch, capsys=capsys
    )
    assert code == 0
    result = json.loads(out)
    assert result["replicas"][0]["fraction_first"] == 1.0
    assert result["replicas"][0]["q6bar_mean_first"] > 0.45
    assert any("consumed a phase" in w for w in result["warnings"])

    code, _ = _run(
        INTERFACE, str(data), "--dt-fs", "100", "--fit-from", "0.9", "--fit-to", "0.5",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert isinstance(code, str) and "--fit-from" in code
    code, _ = _run(
        INTERFACE, str(data), "--dt-fs", "100", "--cutoff", "0.5",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert isinstance(code, str) and "--cutoff" in code


# -- the MD and strain templates, end to end ----------------------------------


def test_the_strain_template_runs_verified_and_fits_emt_copper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from foundation._ops import launch_script

    monkeypatch.chdir(tmp_path)
    template = SKILLS / "elastic-constants" / "assets" / "strain_scan.py"
    result = launch_script(
        tmp_path / ".slab", template, name="elastic-shakeout",
        intent="skill template shakeout (EMT, cubic)", capture_output=True,
    )
    assert result["state"] == "verified"
    assert result["checks_passed"] == result["checks_total"] == 2

    code, out = _run(
        FIT_ELASTIC, str(tmp_path / "elastic.json"), "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    fit = json.loads(out)
    # EMT copper: C11 ~ 173, C12 ~ 116, C44 ~ 90, B ~ 135 GPa.
    assert 120.0 < fit["cij_GPa"]["C11"] < 230.0
    assert 40.0 < fit["cij_GPa"]["C44"] < 140.0
    assert 90.0 < fit["b_hill_GPa"] < 190.0
    assert 0.2 < fit["poisson"] < 0.45
    assert fit["warnings"] == []


def test_the_quench_template_runs_verified_and_reports_densities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from foundation._ops import launch_script

    monkeypatch.chdir(tmp_path)
    template = SKILLS / "melt-quench" / "assets" / "melt_quench.py"
    result = launch_script(
        tmp_path / ".slab", template, name="quench-shakeout",
        intent="skill template shakeout (EMT melt-quench)", capture_output=True,
    )
    assert result["state"] == "verified"
    assert result["checks_passed"] == result["checks_total"] == 3
    trajectories = sorted(tmp_path.glob("quench-*.traj"))
    assert len(trajectories) == 2
    summary = json.loads((tmp_path / "quench.json").read_text())
    assert summary["hold_frames"] == 10

    code, out = _run(
        QUENCH_REPORT, *(str(t) for t in trajectories), "--hold-frames", "10",
        "--rho-c", "9.12", "--json", monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    result = json.loads(out)
    reports = result["reports"]
    # EMT copper glass/quenched solid lands near the crystal's 9.1 g/cm^3.
    assert all(7.0 < r["rho_g_cm3"] < 9.6 for r in reports)
    assert all(-0.1 < r["delta_v"] < 0.25 for r in reports)
    assert all(r["rho_se_g_cm3"] is not None for r in reports)
    assert len(result["rates"]) == 2 and result["log_rate_law"] is not None


def test_the_ramp_template_runs_verified_and_yields_a_classical_cp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from foundation._ops import launch_script

    monkeypatch.chdir(tmp_path)
    template = SKILLS / "thermal-response" / "assets" / "thermal_ramp.py"
    result = launch_script(
        tmp_path / ".slab", template, name="ramp-shakeout",
        intent="skill template shakeout (EMT NPT ladder)", capture_output=True,
    )
    assert result["state"] == "verified"
    assert result["checks_passed"] == result["checks_total"] == 3

    code, out = _run(
        FIT_RAMP, str(tmp_path / "ramp.json"), "--json", monkeypatch=monkeypatch, capsys=capsys
    )
    assert code == 0
    fit = json.loads(out)
    # Classical solid: c_p near 3 kB/atom. 32 atoms in ~1500 A^3 puts the
    # volumetric value in the couple-of-1e6 J/(m^3 K) range.
    assert 1.5e6 < fit["cp_vol_J_m3K"] < 8.0e6
    assert 2.0 < fit["cp_kB_per_atom"] < 4.5
    assert fit["cte_per_K"] > 0
    assert fit["fitted_temperature"] == "measured" and fit["n_atoms"] == 32
    assert len(fit["cte_per_axis_per_K"]) == 3


# -- check_structure (atomsk-structures) --------------------------------------

CHECK = SKILLS / "atomsk-structures" / "scripts" / "check_structure.py"
DATA = Path(__file__).parent / "data"


def test_check_structure_reports_the_recorded_atomsk_supercell(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Real data: the 2x2x2 fcc Al supercell a real atomsk wrote. Known facts:
    a = 8.092 A, nearest neighbor a0/sqrt(2) = 2.861 A, density 2.70 g/cm^3."""
    code, out = _run(
        CHECK, str(DATA / "atomsk-al-fcc-222.xsf"), "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    report = json.loads(out)
    assert report["n_atoms"] == 32
    assert report["formula"] == "Al32"
    assert report["cell_lengths_A"] == pytest.approx([8.092, 8.092, 8.092])
    assert report["min_distance_A"] == pytest.approx(2.8614, abs=1e-3)
    assert report["density_g_cm3"] == pytest.approx(2.70, abs=0.03)
    assert report["close_pairs"] == 0


def test_check_structure_flags_overlapping_atoms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from ase.build import bulk

    atoms = bulk("Cu", "fcc", a=3.615) * (2, 2, 2)
    atoms += atoms[:1]  # a duplicate atom: the classic bad-merge artifact
    atoms.positions[-1, 0] += 0.3
    bad = tmp_path / "overlap.xsf"
    ase_write(bad, atoms)

    # The bond-relative check fails by default: 0.3 A is 0.11 of a Cu-Cu bond.
    code, out = _run(CHECK, str(bad), "--json", monkeypatch=monkeypatch, capsys=capsys)
    assert isinstance(code, str) and "atoms overlap" in code and "covalent" in code
    report = json.loads(out)  # the report is still printed before the failure
    assert report["min_distance_A"] == pytest.approx(0.3, abs=1e-6)
    assert report["close_pairs"] >= 1
    assert report["closest_pair"] == ["Cu", "Cu"]
    assert report["shortest_expected_bond_A"] == pytest.approx(2.64, abs=0.01)
    assert report["min_distance_fraction"] == pytest.approx(0.3 / 2.64, abs=0.01)

    code, _ = _run(
        CHECK, str(bad), "--fail-below-fraction", "0", monkeypatch=monkeypatch, capsys=capsys
    )
    assert code == 0  # 0 disables the relative check

    code, _ = _run(
        CHECK, str(bad), "--fail-below", "1.0", "--fail-below-fraction", "0",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert isinstance(code, str) and "atoms overlap" in code and "1.0 A" in code


def test_check_structure_compares_the_minimum_with_the_expected_bond(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A sound cell passes the relative check: Al-Al 2.86 A is 1.18 of 2 x 1.21."""
    code, out = _run(
        CHECK, str(DATA / "atomsk-al-fcc-222.xsf"), "--json", "--expect-atoms", "32",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    report = json.loads(out)
    assert report["closest_pair"] == ["Al", "Al"]
    assert report["shortest_expected_bond_A"] == pytest.approx(2.42, abs=0.01)
    assert report["min_distance_fraction"] == pytest.approx(1.18, abs=0.01)
    assert report["expected_atoms"] == 32

    code, out = _run(
        CHECK, str(DATA / "atomsk-al-fcc-222.xsf"), "--expect-atoms", "64",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert isinstance(code, str) and "32 atoms, but 64 were expected" in code
    assert "expected bond: 2.42 A (Al-Al)" in out

    code, _ = _run(
        CHECK, str(DATA / "atomsk-al-fcc-222.xsf"), "--fail-below-fraction", "-1",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 2  # argparse refuses a negative fraction


def test_check_structure_large_cells_take_the_neighborlist_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from ase.build import bulk

    atoms = bulk("Al", "fcc", a=4.046, cubic=True) * (8, 8, 8)  # 2048 atoms
    big = tmp_path / "big.xsf"
    ase_write(big, atoms)
    code, out = _run(CHECK, str(big), "--json", monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0
    report = json.loads(out)
    assert report["n_atoms"] == 2048
    assert report["min_distance_A"] == pytest.approx(2.8614, abs=1e-3)


def test_check_structure_refuses_missing_or_junk_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = _run(CHECK, str(tmp_path / "gone.xsf"), monkeypatch=monkeypatch, capsys=capsys)
    assert isinstance(code, str) and "no such file" in code

    junk = tmp_path / "junk.xsf"
    junk.write_text("not a structure")
    code, _ = _run(CHECK, str(junk), monkeypatch=monkeypatch, capsys=capsys)
    assert isinstance(code, str) and "cannot read" in code


# -- pair_style_for -----------------------------------------------------------


def test_pair_style_for_reads_a_setfl_header_as_eam_alloy(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The Zhou tungsten file that one campaign spent seventy minutes
    rewriting: a setfl file, fine as it is, under the wrong pair_style."""
    path = DATA / "lammps-w-zhou-setfl.eam.alloy"
    code, out = _run(PAIR_STYLE, str(path), monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0
    assert "format: setfl  elements: W" in out
    assert "pair_style eam/alloy\n" in out
    assert f"pair_coeff * * {path} W" in out
    assert "warning" not in out


def test_pair_style_for_reads_a_funcfl_header_as_eam(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = DATA / "lammps-cu-u3-funcfl.eam"
    code, out = _run(PAIR_STYLE, str(path), "--json", monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0
    result = json.loads(out)
    assert result["format"] == "funcfl"
    assert result["pair_style"] == "eam"
    assert result["pair_coeff"] == f"pair_coeff * * {path}"
    assert result["warnings"] == []


def test_pair_style_for_warns_when_the_name_and_the_header_disagree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    misnamed = tmp_path / "W.eam"
    misnamed.write_bytes((DATA / "lammps-w-zhou-setfl.eam.alloy").read_bytes())
    code, out = _run(PAIR_STYLE, str(misnamed), monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0
    assert "pair_style eam/alloy" in out  # the header wins
    assert "warning: header is setfl but the name" in out


def test_pair_style_for_needs_elements_for_ace_and_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ace = tmp_path / "W.yace"
    ace.write_text("elements: [W]\n")
    code, out = _run(PAIR_STYLE, str(ace), monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0
    assert "pair_style pace" in out and "El..." in out and "pass --elements" in out
    code, out = _run(
        PAIR_STYLE, str(ace), "--elements", "W", monkeypatch=monkeypatch, capsys=capsys
    )
    assert code == 0 and f"pair_coeff * * {ace} W" in out and "warning" not in out
    grace = tmp_path / "grace-2l-omat"
    grace.mkdir()
    code, out = _run(
        PAIR_STYLE, str(grace), "--elements", "W", "Re", monkeypatch=monkeypatch, capsys=capsys
    )
    assert code == 0 and "pair_style grace" in out and f"pair_coeff * * {grace} W Re" in out


def test_pair_style_for_refuses_what_it_cannot_classify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    other = tmp_path / "structure.cif"
    other.write_text("data_W\n_cell_length_a 3.17\n")
    code, _ = _run(PAIR_STYLE, str(other), monkeypatch=monkeypatch, capsys=capsys)
    assert code == 2
    missing = tmp_path / "missing.eam"
    code, _ = _run(PAIR_STYLE, str(missing), monkeypatch=monkeypatch, capsys=capsys)
    assert code == 2
