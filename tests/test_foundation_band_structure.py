"""band_structure through a fake pw.x that replays real output.

The fake reads the input it is given and replays the real SCF output for
``calculation='scf'`` and the real bands output for
``calculation='bands'`` (``tests/data/qe-si-bands-*.pwo``, pw.x 7.5). It
also copies each input aside, so a test can read what the task asked
pw.x to do. The real-QE test at the bottom runs when ``$SLAB_TEST_PW`` and
``$SLAB_TEST_PSEUDO_DIR`` point at a pw.x and a pseudopotential directory.
"""

import json
import os
from copy import deepcopy
from pathlib import Path
from subprocess import CalledProcessError

import pytest
from ase.build import bulk

import foundation.tasks as tasks
from foundation import Workspace
from foundation._ops import read_artifact
from foundation.tasks import band_structure
from slab import backends
from slab.backends import resolve_pseudopotentials

DATA = Path(__file__).parent / "data"
SI_SCF = DATA / "qe-si-bands-scf.pwo"
SI_BANDS = DATA / "qe-si-bands-bands.pwo"
AL_SCF = DATA / "qe-al-bands-scf.pwo"
AL_BANDS = DATA / "qe-al-bands-bands.pwo"
SI_JSON = DATA / "qe-si-bands.json"

FENCE = " " + "%" * 40


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    with Workspace(tmp_path / "ws") as workspace:
        yield workspace


def _fake_pw(tmp_path: Path, scf: Path, bands: Path | None, *, fail: str | None = None) -> Path:
    """A pw.x that replays *scf* or *bands* by the input's calculation, and
    copies each input to ``inputs/<calculation>.pwi``. ``fail`` names the
    step that fails the way QE fails: a fenced error block, CRASH, exit 3."""
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    failing = f"""echo "     Program PWSCF v.7.5 starts on 18Sep2026"
    echo "{FENCE}"
    echo "     Error in routine c_bands (1):"
    echo "     too many bands are not converged"
    echo "{FENCE}"
    printf '%s\\n     Error in routine c_bands (1):\\n' "{FENCE}" > CRASH
    exit 3"""
    replay = {"scf": f'cat "{scf}"', "bands": f'cat "{bands}"'}
    if fail is not None:
        replay[fail] = failing
    script = tmp_path / "fake-pw.x"
    script.write_text(
        f"""#!/bin/sh
if grep -q "calculation *= *'bands'" "$2"; then step=bands; else step=scf; fi
cp "$2" "{inputs}/$step.pwi"
if [ "$step" = bands ]; then
    {replay["bands"]}
else
    {replay["scf"]}
fi
"""
    )
    script.chmod(0o755)
    return script


def _options(script: Path, tmp_path: Path) -> dict:
    return {
        "command": str(script),
        "pseudo_dir": str(tmp_path),
        "pseudopotentials": {"Si": "Si.pbesol-n-rrkjus_psl.1.0.0.UPF"},
        "kpts": [7, 7, 7],
        "input_data": {
            "control": {"tprnfor": True, "tstress": True},
            "system": {
                "ecutwfc": 30.0,
                "ecutrho": 240.0,
                "occupations": "smearing",
                "smearing": "cold",
                "degauss": 0.02,
            },
        },
    }


def _si():
    return bulk("Si", "diamond", a=5.43)


@pytest.fixture()
def scratches(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Every scratch directory band_structure makes, to check it is gone."""
    made: list[Path] = []
    original = backends.engine_scratch

    def recording(prefix: str) -> Path:
        made.append(original(prefix))
        return made[-1]

    monkeypatch.setattr(backends, "engine_scratch", recording)
    return made


def test_band_structure_returns_the_verdict_and_keeps_three_artifacts(
    ws: Workspace, tmp_path: Path, scratches: list[Path]
) -> None:
    options = _options(_fake_pw(tmp_path, SI_SCF, SI_BANDS), tmp_path)
    with ws.start_run(name="si-bands", intent="fake pw.x band structure") as run:
        bands, info = band_structure(_si(), calculator_options=options, npoints=60, label="si")

    assert info["is_metal"] is False
    assert info["gap_kind"] == "indirect"
    assert info["vbm_at"]["label"] == "G"
    assert info["gap"] == pytest.approx(info["cbm"] - info["vbm"], abs=1e-4)
    assert info["fermi"] == 6.188
    assert info["engine_version"] == "7.5"
    assert info["scf_energy"] < 0 and info["energy_unit"] == "eV"
    assert (info["path"], info["lattice"], info["npoints"]) == ("GXU,KGLWX", "cF2", 60)
    assert (info["n_bands"], info["n_atoms"], info["n_valence_bands"]) == (8, 2, 4)
    assert info["artifacts"] == ["si-scf.pwo", "si-bands.pwo", "si-bands.json"]

    kept = {a.name: a for a in ws.runs.list_artifacts(run.id)}
    assert set(kept) == {"si-scf.pwo", "si-bands.pwo", "si-bands.json"}
    assert kept["si-bands.json"].role == "terminal"
    assert kept["si-scf.pwo"].role == kept["si-bands.pwo"].role == "intermediate"
    assert ws.artifacts.get(kept["si-scf.pwo"].hash).read_text() == SI_SCF.read_text()
    assert ws.artifacts.get(kept["si-bands.pwo"].hash).read_text() == SI_BANDS.read_text()

    # The result file is the one the real run wrote, number for number.
    written = json.loads(ws.artifacts.get(kept["si-bands.json"].hash).read_text())
    assert written == json.loads(SI_JSON.read_text())
    assert bands == written  # 480 eigenvalues travel inline
    assert written["summary"]["gap"] == info["gap"]

    assert len(scratches) == 1 and not scratches[0].exists()


def test_the_bands_input_reads_the_scf_it_follows(
    ws: Workspace, tmp_path: Path, scratches: list[Path]
) -> None:
    options = _options(_fake_pw(tmp_path, SI_SCF, SI_BANDS), tmp_path)
    with ws.start_run(name="si-bands", intent="inputs"):
        band_structure(_si(), calculator_options=options, npoints=60, label="si")
    scf = (tmp_path / "inputs" / "scf.pwi").read_text()
    step = (tmp_path / "inputs" / "bands.pwi").read_text()

    assert "calculation      = 'scf'" in scf
    assert "K_POINTS automatic" in scf
    assert "calculation      = 'bands'" in step
    assert "verbosity        = 'high'" in step
    assert "nbnd             = 8" in step
    assert "tprnfor          = .false." in step
    assert "tstress          = .false." in step
    assert "K_POINTS crystal_b\n60\n" in step
    for line in ("prefix           = 'pwscf'", "outdir           = './'"):
        assert line in scf and line in step


def test_a_caller_nbands_reaches_the_bands_input(ws: Workspace, tmp_path: Path) -> None:
    options = _options(_fake_pw(tmp_path, SI_SCF, SI_BANDS), tmp_path)
    with ws.start_run(name="si-bands", intent="nbands"):
        band_structure(_si(), calculator_options=options, npoints=60, nbands=12)
    assert "nbnd             = 12" in (tmp_path / "inputs" / "bands.pwi").read_text()


def test_band_structure_refuses_what_it_cannot_run(ws: Workspace, tmp_path: Path) -> None:
    options = _options(_fake_pw(tmp_path, SI_SCF, SI_BANDS), tmp_path)
    with pytest.raises(ValueError, match="needs a Quantum ESPRESSO engine; 'emt'"):
        band_structure(_si(), engine="emt")
    no_mesh = {k: v for k, v in options.items() if k != "kpts"}
    with pytest.raises(ValueError, match="declares no k-points"):
        band_structure(_si(), calculator_options=no_mesh)
    relaxing = deepcopy(options)
    relaxing["input_data"]["control"]["calculation"] = "relax"
    with pytest.raises(
        ValueError,
        match="runs its own SCF and then its own bands step, but calculator_options "
        "declares calculation='relax'",
    ):
        band_structure(_si(), calculator_options=relaxing)
    for key, value in (("nspin", 2), ("noncolin", True), ("lspinorb", True)):
        spinning = deepcopy(options)
        spinning["input_data"]["system"][key] = value
        with pytest.raises(ValueError, match=f"does not support {key}=.* yet, because"):
            band_structure(_si(), calculator_options=spinning)
    with pytest.raises(ValueError, match="own scratch directory"):
        band_structure(_si(), calculator_options={**options, "directory": str(tmp_path)})
    # The occupied count is known after the SCF, and the refusal comes
    # before the bands step runs.
    with (
        ws.start_run(name="too-few", intent="nbands below the occupied bands"),
        pytest.raises(ValueError, match="nbands=3 holds fewer bands than the 4 occupied"),
    ):
        band_structure(_si(), calculator_options=options, npoints=60, nbands=3)
    assert not (tmp_path / "inputs" / "bands.pwi").exists()


@pytest.mark.parametrize("step", ["scf", "bands"])
def test_a_failing_step_keeps_its_evidence_under_the_step_name(
    ws: Workspace, tmp_path: Path, scratches: list[Path], step: str
) -> None:
    options = _options(_fake_pw(tmp_path, SI_SCF, SI_BANDS, fail=step), tmp_path)
    with (
        pytest.raises(CalledProcessError) as excinfo,
        ws.start_run(name="si-fail", intent=f"{step} fails") as run,
    ):
        band_structure(_si(), calculator_options=options, npoints=60, label="si")
    notes = " ".join(excinfo.value.__notes__)
    assert f"band_structure failed in its {step} step" in notes
    assert "Error in routine c_bands (1): too many bands are not converged" in notes
    names = {a.name for a in ws.runs.list_artifacts(run.id)}
    assert f"si-{step}-failed.pwo" in names
    assert f"si-{step}-failed.crash" in names
    assert ("si-scf.pwo" in names) is (step == "bands")
    assert not scratches[0].exists()


def test_the_callers_options_are_unchanged_and_a_repeat_is_a_cache_hit(
    ws: Workspace, tmp_path: Path
) -> None:
    options = _options(_fake_pw(tmp_path, SI_SCF, SI_BANDS), tmp_path)
    before = deepcopy(options)
    with ws.start_run(name="first", intent="first"):
        first, _ = band_structure(_si(), calculator_options=options, npoints=60, label="si")
    assert options == before
    with ws.start_run(name="again", intent="cache hit") as again:
        second, info = band_structure(_si(), calculator_options=options, npoints=60, label="si")
    assert ws.runs.list_tasks(again.id)[0].cache_hit is True
    assert second == first
    read = read_artifact(ws, run_id=again.id, name=info["artifacts"][-1])
    assert json.loads(read.text)["summary"]["gap_kind"] == "indirect"


def test_a_long_listing_stays_in_the_artifact(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tasks, "_BANDS_INLINE_LIMIT", 100)
    options = _options(_fake_pw(tmp_path, SI_SCF, SI_BANDS), tmp_path)
    with ws.start_run(name="long", intent="inline limit"):
        bands, _ = band_structure(_si(), calculator_options=options, npoints=60)
    assert "energies" not in bands
    assert bands["energies_in"] == "bands-bands.json"
    assert bands["summary"]["is_metal"] is False


def test_a_metal_reports_no_gap(ws: Workspace, tmp_path: Path) -> None:
    options = _options(_fake_pw(tmp_path, AL_SCF, AL_BANDS), tmp_path)
    options["pseudopotentials"] = {"Al": "Al.pbesol-n-kjpaw_psl.1.0.0.UPF"}
    with ws.start_run(name="al", intent="metal"):
        _, info = band_structure(
            bulk("Al", "fcc", a=4.04), calculator_options=options, npoints=60, label="al"
        )
    assert info["is_metal"] is True
    assert info["gap"] is None and info["gap_kind"] is None
    assert info["n_bands"] == 6


def test_a_registry_alias_on_the_qe_factory_is_accepted(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _fake_pw(tmp_path, SI_SCF, SI_BANDS)
    registry = tmp_path / "engines.json"
    registry.write_text(
        json.dumps(
            {
                "cluster": "test",
                "engines": {
                    "qe-site": {
                        "calculator": "slab.backends.qe_calculator",
                        "options": {"command": str(script), "pseudo_dir": str(tmp_path)},
                        "version": "7.5",
                    }
                },
            }
        )
    )
    monkeypatch.setenv("SLAB_ENGINES", str(registry))
    located = ("command", "pseudo_dir")
    options = {k: v for k, v in _options(script, tmp_path).items() if k not in located}
    with ws.start_run(name="alias", intent="registry alias"):
        _, info = band_structure(_si(), engine="qe-site", calculator_options=options, npoints=60)
    assert info["engine"] == "qe-site"
    assert info["gap_kind"] == "indirect"


def test_band_structure_is_in_the_task_catalog() -> None:
    from foundation._ops import task_catalog

    entry = next(t for t in task_catalog() if t["name"] == "band_structure")
    assert "gap" in entry["summary"] or "bands" in entry["summary"]


# -- the real thing, when present ------------------------------------------------------


@pytest.mark.skipif(
    not (os.environ.get("SLAB_TEST_PW") and os.environ.get("SLAB_TEST_PSEUDO_DIR")),
    reason="set SLAB_TEST_PW and SLAB_TEST_PSEUDO_DIR to test against a real pw.x",
)
def test_band_structure_qe_real_integration(ws: Workspace) -> None:
    pw = os.environ["SLAB_TEST_PW"]
    pseudo_dir = os.environ["SLAB_TEST_PSEUDO_DIR"]
    atoms = _si()
    with ws.start_run(name="qe-real-bands", intent="real pw.x band structure") as run:
        bands, info = band_structure(
            atoms,
            npoints=20,
            label="si",
            calculator_options={
                "command": pw,
                "pseudo_dir": pseudo_dir,
                "pseudopotentials": resolve_pseudopotentials(atoms, pseudo_dir),
                "kpts": [4, 4, 4],
                "input_data": {"system": {"ecutwfc": 20.0}},
            },
        )
    assert info["is_metal"] is False
    assert info["gap"] > 0
    assert info["vbm_at"]["label"] == "G"
    assert len(bands["energies"]) == 20
    names = {a.name for a in ws.runs.list_artifacts(run.id)}
    assert names == {"si-scf.pwo", "si-bands.pwo", "si-bands.json"}


def test_fixed_occupations_still_read_si_as_an_insulator(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # pw.x prints the highest occupied level of the SCF mesh under fixed
    # occupations, and that level is the valence band maximum itself.
    import slab.bands

    original = slab.bands.scf_levels

    def fixed(text: str) -> dict:
        return {**original(text), "fermi": 6.1597, "fermi_source": "highest occupied level"}

    monkeypatch.setattr(slab.bands, "scf_levels", fixed)
    options = _options(_fake_pw(tmp_path, SI_SCF, SI_BANDS), tmp_path)
    with ws.start_run(name="si-fixed", intent="fixed occupations"):
        _, info = band_structure(_si(), calculator_options=options, npoints=60, label="si")
    assert info["is_metal"] is False
    assert info["gap"] == 0.4639 and info["gap_kind"] == "indirect"
    assert info["n_valence_bands"] == 4


def test_a_conventional_cell_runs_as_its_standardized_primitive_cell(
    ws: Workspace, tmp_path: Path
) -> None:
    # seekpath's path is written for its own primitive cell, so both pw.x
    # steps run on that cell, and the 8-atom cell's mesh is replaced by one
    # at least as dense on the 2-atom cell.
    cubic = bulk("Si", "diamond", a=5.43, cubic=True)
    options = _options(_fake_pw(tmp_path, SI_SCF, SI_BANDS), tmp_path)
    options["kpts"] = [4, 4, 4]
    with ws.start_run(name="si-cubic", intent="conventional cell"):
        bands, info = band_structure(cubic, calculator_options=options, npoints=60, label="si")
    assert options["kpts"] == [4, 4, 4]  # the caller's dict is a traced input
    assert (info["n_atoms_input"], info["n_atoms"], info["cell_changed"]) == (8, 2, True)
    assert info["scf_kpts"] == [7, 7, 7]
    assert len(bands["primitive_cell"]["symbols"]) == 2
    for step in ("scf", "bands"):
        written = (tmp_path / "inputs" / f"{step}.pwi").read_text()
        assert "nat              = 2" in written
    scf = (tmp_path / "inputs" / "scf.pwi").read_text()
    assert "K_POINTS automatic\n7 7 7" in scf
    assert info["gap"] == 0.4639  # the same k-points as the primitive run


def test_an_unchanged_lattice_keeps_the_callers_mesh(ws: Workspace, tmp_path: Path) -> None:
    options = _options(_fake_pw(tmp_path, SI_SCF, SI_BANDS), tmp_path)
    with ws.start_run(name="si-prim", intent="primitive cell"):
        _, info = band_structure(_si(), calculator_options=options, npoints=60, label="si")
    assert info["cell_changed"] is False and "scf_kpts" not in info
    assert "K_POINTS automatic\n7 7 7" in (tmp_path / "inputs" / "scf.pwi").read_text()
