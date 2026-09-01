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
    for analysis in ("thermal-response", "kinetic-fits", "nemd-transport", "nucleation-cnt"):
        assert skills[analysis].agents == frozenset({"md-expert", "analysis-expert"})
    # The scriptless skills are deliberate: scripts are optional in the format.
    assert not (skills["surface-energy"].root / "scripts").exists()
    assert not (skills["two-phase-melting"].root / "scripts").exists()
    assert not (skills["mlip-training"].root / "scripts").exists()


# -- fit_rates ----------------------------------------------------------------

FIT_RATES = SKILLS / "kinetic-fits" / "scripts" / "fit_rates.py"
FIT_NEMD = SKILLS / "nemd-transport" / "scripts" / "fit_nemd.py"
FIT_ELASTIC = SKILLS / "elastic-constants" / "scripts" / "fit_elastic.py"
QUENCH_REPORT = SKILLS / "melt-quench" / "scripts" / "quench_report.py"
FIT_RAMP = SKILLS / "thermal-response" / "scripts" / "fit_thermal_ramp.py"
ADHESION = SKILLS / "interface-adhesion" / "scripts" / "adhesion.py"
CNT = SKILLS / "nucleation-cnt" / "scripts" / "cnt.py"

_KB_EV = 8.617333262e-5


def test_fit_rates_recovers_an_arrhenius_law(
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


def test_fit_rates_recovers_a_vft_law(
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
    assert fit["warnings"] == []


def test_fit_rates_locates_the_melting_crossing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "velocity.json"
    rows = [{"T": t, "value": 0.004 * (823.0 - t)} for t in (700, 760, 820, 880, 940)]
    data.write_text(json.dumps(rows))
    code, out = _run(
        FIT_RATES, str(data), "--mode", "crossing", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    fit = json.loads(out)
    assert len(fit["crossings"]) == 1
    assert abs(fit["crossings"][0]["T_K"] - 823.0) < 0.5


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

    data.write_text(json.dumps([{"T": 500, "value": 1.0}, {"T": 500, "value": 2.0}]))
    code, _ = _run(
        FIT_RATES, str(data), "--mode", "arrhenius", monkeypatch=monkeypatch, capsys=capsys
    )
    assert isinstance(code, str) and "duplicate temperatures" in code


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
    assert fit["warnings"] == []

    text = tmp_path / "profile.dat"
    text.write_text("# x T\n" + "\n".join(f"{x} {300.0 + 0.2 * x}" for x in xs))
    code, out = _run(
        FIT_NEMD, "kappa", str(text), "--flux", "3e9", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    assert abs(json.loads(out)["k_W_mK"] - 1.5) < 1e-6


def test_fit_nemd_tbr_measures_the_interface_jump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = [{"x": float(x), "T": 400.0 - 0.1 * float(x)} for x in np.linspace(0.0, 48.0, 13)]
    rows += [
        {"x": float(x), "T": 365.0 - 0.1 * (float(x) - 52.0)} for x in np.linspace(52.0, 100.0, 13)
    ]
    profile = tmp_path / "tbr.json"
    profile.write_text(json.dumps(rows))
    # both branches extrapolate to the interface with a 30 K jump; R = 30/3e9 = 1e-8
    code, out = _run(
        FIT_NEMD, "tbr", str(profile), "--flux", "3e9", "--interface", "50", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    fit = json.loads(out)
    assert abs(fit["tbr_m2K_W"] - 1e-8) / 1e-8 < 0.05
    assert abs(fit["left"]["k_W_mK"] - 3.0) < 1e-6


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
    data = tmp_path / "quench.traj"
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


# -- fit_thermal_ramp ---------------------------------------------------------


def test_fit_thermal_ramp_slopes_and_latent_heat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    crystal = tmp_path / "crystal.json"
    liquid = tmp_path / "liquid.json"
    crystal.write_text(
        json.dumps([{"T": t, "E": -100.0 + 0.01 * t, "V": 500.0 + 0.05 * t}
                    for t in range(300, 800, 100)])
    )
    liquid.write_text(
        json.dumps([{"T": t, "E": -80.0 + 0.012 * t, "V": 520.0 + 0.05 * t}
                    for t in range(300, 800, 100)])
    )
    code, out = _run(FIT_RAMP, str(crystal), "--json", monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0
    fit = json.loads(out)
    assert abs(fit["cp_total_eV_K"] - 0.01) < 1e-9
    # 0.01 eV/K over ~525 A^3 -> 3.05e6 J/(m^3 K); CTE = 0.05/(3*525)
    assert abs(fit["cp_vol_J_m3K"] - 3.05e6) / 3.05e6 < 0.01
    assert abs(fit["cte_per_K"] - 0.05 / (3 * 525.0)) / (0.05 / (3 * 525.0)) < 0.01

    code, out = _run(
        FIT_RAMP, str(crystal), "--other", str(liquid), "--at", "500", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    fit = json.loads(out)
    assert abs(fit["dH_eV"] - 21.0) < 1e-9  # (-80+6) - (-100+5)


def test_fit_thermal_ramp_refuses_extrapolation_and_thin_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ramp = tmp_path / "ramp.json"
    ramp.write_text(
        json.dumps([{"T": t, "E": 0.01 * t, "V": 500.0} for t in range(300, 800, 100)])
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


# -- adhesion -----------------------------------------------------------------


def test_adhesion_reports_w_adh_and_the_potency_factor(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # W_adh = 10 eV over 100 A^2 = 0.1 eV/A^2 = 1.6022 J/m^2; gamma equal to
    # W_adh puts cos(theta) at 0: theta 90 deg, f = 0.5 exactly.
    code, out = _run(
        ADHESION, "--e-interface", "-110", "--e-a", "-60", "--e-b", "-40",
        "--area", "100", "--gamma", "1.6021766208", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    fit = json.loads(out)
    assert abs(fit["w_adh_J_m2"] - 1.6021766208) < 1e-9
    assert abs(fit["cos_theta"]) < 1e-9
    assert abs(fit["f_het"] - 0.5) < 1e-9
    assert fit["warnings"] == []


def test_adhesion_clamps_complete_wetting_and_flags_repulsion(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = _run(
        ADHESION, "--e-interface", "-110", "--e-a", "-60", "--e-b", "-40",
        "--area", "100", "--gamma", "0.5", "--json",
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

    code, _ = _run(
        ADHESION, "--e-interface", "-1", "--e-a", "-1", "--e-b", "-1", "--area", "0",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert isinstance(code, str) and "--area" in code


# -- cnt ----------------------------------------------------------------------


def test_cnt_gamma_inverts_gibbs_thomson_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tm, dhf, gamma = 823.0, 2.4e8, 0.06
    rows = [
        {"T": t, "r_star_nm": 2 * gamma * tm / (dhf * (tm - t)) * 1e9}
        for t in (600.0, 650.0, 700.0)
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


def test_cnt_barrier_matches_the_closed_form(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import math

    tm, dhf, gamma, t = 823.0, 2.4e8, 0.06, 600.0
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
    assert result["checks_passed"] == result["checks_total"] == 2
    trajectories = sorted(tmp_path.glob("quench-*.traj"))
    assert len(trajectories) == 2
    assert (tmp_path / "quench.json").is_file()

    code, out = _run(
        QUENCH_REPORT, *(str(t) for t in trajectories), "--rho-c", "9.12", "--json",
        monkeypatch=monkeypatch, capsys=capsys,
    )
    assert code == 0
    reports = json.loads(out)["reports"]
    # EMT copper glass/quenched solid lands near the crystal's 9.1 g/cm^3.
    assert all(7.0 < r["rho_g_cm3"] < 9.6 for r in reports)
    assert all(-0.1 < r["delta_v"] < 0.25 for r in reports)


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
    assert fit["cte_per_K"] > 0


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

    code, out = _run(CHECK, str(bad), "--json", monkeypatch=monkeypatch, capsys=capsys)
    assert code == 0
    report = json.loads(out)
    assert report["min_distance_A"] == pytest.approx(0.3, abs=1e-6)
    assert report["close_pairs"] >= 1

    code, _ = _run(
        CHECK, str(bad), "--fail-below", "1.0", monkeypatch=monkeypatch, capsys=capsys
    )
    assert isinstance(code, str) and "atoms overlap" in code


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
