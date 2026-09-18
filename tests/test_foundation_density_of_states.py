"""density_of_states through fake executables that replay real output.

The fakes are named ``pw.x``, ``dos.x``, and ``projwfc.x`` in one
directory, because the task derives each tool's command from the pw.x
command by naming its sibling. Each fake copies the input it is given to
``inputs/`` and replays a real Quantum ESPRESSO 7.5 capture: ``pw.x``
replays the SCF or the NSCF by the input's ``calculation``, ``dos.x``
writes the real table and prints the real standard output, and
``projwfc.x`` writes the real per-state files. The real-QE test at the
bottom runs when ``$SLAB_TEST_PW`` and ``$SLAB_TEST_PSEUDO_DIR`` point at
a pw.x and a pseudopotential directory.
"""

import json
import os
from copy import deepcopy
from pathlib import Path

import pytest
from ase.build import bulk

import foundation.tasks as tasks
from foundation import Workspace
from foundation._ops import read_artifact
from foundation.tasks import density_of_states
from slab import backends
from slab.backends import resolve_pseudopotentials
from slab.errors import QeToolError

DATA = Path(__file__).parent / "data"
SI = {
    "scf": DATA / "qe-si-dos-scf.pwo",
    "nscf": DATA / "qe-si-dos-nscf.pwo",
    "table": DATA / "qe-si-dos.dat",
    "dos": DATA / "qe-si-dos-dos.out",
    "projwfc": DATA / "qe-si-dos-projwfc.out",
    "pdos": DATA / "qe-si-dos-pdos",
    "json": DATA / "qe-si-dos.json",
}
AL = {
    "scf": DATA / "qe-al-dos-scf.pwo",
    "nscf": DATA / "qe-al-dos-nscf.pwo",
    "table": DATA / "qe-al-dos.dat",
    "dos": DATA / "qe-al-dos-dos.out",
    "projwfc": DATA / "qe-al-dos-projwfc.out",
    "pdos": DATA / "qe-al-dos-pdos",
    "json": DATA / "qe-al-dos.json",
}

FENCE = " " + "%" * 40


def _failing(program: str) -> str:
    """A tool that fails the way Quantum ESPRESSO fails: a fenced block, exit 3."""
    return f"""echo "     Program {program} v.7.5 starts on 18Sep2026"
echo "{FENCE}"
echo "     Error in routine davcio (10):"
echo "     error while reading from file"
echo "{FENCE}"
printf '%s\\n     Error in routine davcio (10):\\n' "{FENCE}" > CRASH
exit 3"""


def _fake_bin(tmp_path: Path, capture: dict, *, fail: str | None = None) -> Path:
    """A directory holding fake pw.x, dos.x, and projwfc.x that replay *capture*."""
    root = tmp_path / "bin"
    root.mkdir(exist_ok=True)
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    scripts = {
        "pw.x": f"""
if grep -q "calculation *= *'nscf'" "$2"; then step=nscf; else step=scf; fi
cp "$2" "{inputs}/$step.pwi"
if [ "$step" = nscf ]; then
    {_failing("PWSCF") if fail == "nscf" else f'cat "{capture["nscf"]}"'}
else
    {_failing("PWSCF") if fail == "scf" else f'cat "{capture["scf"]}"'}
fi
""",
        "dos.x": f"""
cp "$2" "{inputs}/dos.in"
{_failing("DOS") if fail == "dos" else ""}
fildos=$(sed -n "s/.*fildos *= *'\\([^']*\\)'.*/\\1/p" "$2")
cp "{capture["table"]}" "$fildos"
cat "{capture["dos"]}"
""",
        "projwfc.x": f"""
cp "$2" "{inputs}/projwfc.in"
{_failing("PROJWFC") if fail == "projwfc" else ""}
filpdos=$(sed -n "s/.*filpdos *= *'\\([^']*\\)'.*/\\1/p" "$2")
for f in {capture["pdos"]}/*; do
    base=$(basename "$f")
    cp "$f" "$filpdos.${{base#*.}}"
done
cat "{capture["projwfc"]}"
""",
    }
    for name, body in scripts.items():
        script = root / name
        script.write_text("#!/bin/sh\n" + body.strip() + "\n")
        script.chmod(0o755)
    return root


def _options(root: Path, tmp_path: Path, symbol: str, upf: str) -> dict:
    return {
        "command": str(root / "pw.x"),
        "pseudo_dir": str(tmp_path),
        "pseudopotentials": {symbol: upf},
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


def _si_options(tmp_path: Path, **kwargs) -> dict:
    root = _fake_bin(tmp_path, SI, **kwargs)
    return _options(root, tmp_path, "Si", "Si.pbesol-n-rrkjus_psl.1.0.0.UPF")


def _si():
    return bulk("Si", "diamond", a=5.43)


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    with Workspace(tmp_path / "ws") as workspace:
        yield workspace


@pytest.fixture()
def scratches(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Every scratch directory the task makes, to check it is gone."""
    made: list[Path] = []
    original = backends.engine_scratch

    def recording(prefix: str) -> Path:
        made.append(original(prefix))
        return made[-1]

    monkeypatch.setattr(backends, "engine_scratch", recording)
    return made


def test_density_of_states_returns_the_verdict_and_keeps_six_artifacts(
    ws: Workspace, tmp_path: Path, scratches: list[Path]
) -> None:
    options = _si_options(tmp_path)
    with ws.start_run(name="si-dos", intent="fake dos pipeline") as run:
        dos, info = density_of_states(
            _si(),
            calculator_options=options,
            dos_kpts=[8, 8, 8],
            delta_e=0.05,
            degauss=0.005,
            projected=True,
            label="si",
        )

    assert info["is_metal"] is False
    assert info["gap"] == 0.5061 and info["gap_kind"] == "indirect"
    assert info["fermi"] == 6.4359
    assert info["dos_at_fermi"] < 1e-4
    assert info["engine_version"] == "7.5"
    assert info["scf_energy"] < 0 and info["energy_unit"] == "eV"
    assert (info["scf_kpts"], info["dos_kpts"]) == ([7, 7, 7], [8, 8, 8])
    assert (info["degauss_ry"], info["delta_e"]) == (0.005, 0.05)
    assert (info["n_bands"], info["n_atoms"]) == (8, 2)
    assert info["projected"] is True
    assert info["projection_groups"] == ["Si-p", "Si-s"]
    assert info["artifacts"] == [
        "si-scf.pwo",
        "si-nscf.pwo",
        "si-dos.dat",
        "si-dos.out",
        "si-projwfc.out",
        "si-dos.json",
    ]

    kept = {a.name: a for a in ws.runs.list_artifacts(run.id)}
    assert set(kept) == set(info["artifacts"])
    assert kept["si-dos.json"].role == "terminal"
    assert all(kept[name].role == "intermediate" for name in info["artifacts"][:-1])
    assert ws.artifacts.get(kept["si-nscf.pwo"].hash).read_text() == SI["nscf"].read_text()
    assert ws.artifacts.get(kept["si-dos.dat"].hash).read_text() == SI["table"].read_text()

    # The result file is the one the real run wrote, number for number.
    written = json.loads(ws.artifacts.get(kept["si-dos.json"].hash).read_text())
    assert written == json.loads(SI["json"].read_text())
    # 450 rows times five curves is past the inline limit, so the arrays
    # stay in the artifact and the return value names it.
    assert "energies" not in dos and dos["dos_in"] == "si-dos.json"
    assert dos["summary"]["gap"] == info["gap"]
    assert len(scratches) == 1 and not scratches[0].exists()


def test_every_step_reads_the_one_save_directory(ws: Workspace, tmp_path: Path) -> None:
    options = _si_options(tmp_path)
    with ws.start_run(name="si-dos", intent="inputs"):
        density_of_states(
            _si(),
            calculator_options=options,
            dos_kpts=[8, 8, 8],
            delta_e=0.05,
            degauss=0.005,
            projected=True,
            label="si",
        )
    scf = (tmp_path / "inputs" / "scf.pwi").read_text()
    nscf = (tmp_path / "inputs" / "nscf.pwi").read_text()
    dos = (tmp_path / "inputs" / "dos.in").read_text()
    projwfc = (tmp_path / "inputs" / "projwfc.in").read_text()

    assert "calculation      = 'scf'" in scf
    assert "K_POINTS automatic\n7 7 7" in scf
    assert "calculation      = 'nscf'" in nscf
    assert "verbosity        = 'high'" in nscf
    assert "nbnd             = 8" in nscf
    assert "tprnfor          = .false." in nscf
    assert "K_POINTS automatic\n8 8 8" in nscf
    for line in ("prefix           = 'pwscf'", "outdir           = './'"):
        assert line in scf and line in nscf
    for written in (dos, projwfc):
        assert "prefix = 'pwscf'" in written
        assert "outdir = './'" in written
        assert "DeltaE = 0.05" in written
        assert "degauss = 0.005" in written
        assert "ngauss = 0" in written
    assert dos.startswith("&DOS") and "fildos = 'si-dos.dat'" in dos
    assert projwfc.startswith("&PROJWFC") and "filpdos = 'si'" in projwfc
    # dos.x and projwfc.x are given one window, so their grids line up.
    window = [line for line in dos.splitlines() if line.strip().startswith(("Emin", "Emax"))]
    assert len(window) == 2
    assert all(line in projwfc.splitlines() for line in window)


def test_the_dense_mesh_defaults_to_twice_the_scf_mesh(ws: Workspace, tmp_path: Path) -> None:
    options = _si_options(tmp_path)
    with ws.start_run(name="si-dos", intent="default mesh"):
        _, info = density_of_states(
            _si(), calculator_options=options, delta_e=0.05, degauss=0.005
        )
    assert info["dos_kpts"] == [14, 14, 14]
    assert "K_POINTS automatic\n14 14 14" in (tmp_path / "inputs" / "nscf.pwi").read_text()
    assert info["projected"] is False and "projection_groups" not in info


def test_a_kspacing_scf_halves_its_spacing_for_the_dense_mesh(
    ws: Workspace, tmp_path: Path
) -> None:
    options = _si_options(tmp_path)
    del options["kpts"]
    options["kspacing"] = 0.2
    with ws.start_run(name="si-dos", intent="kspacing"):
        _, info = density_of_states(
            _si(), calculator_options=options, delta_e=0.05, degauss=0.005
        )
    assert info["dos_kspacing"] == pytest.approx(0.1)
    assert info["dos_kpts"] is None and info["scf_kpts"] == pytest.approx(0.2)


def test_the_caller_names_the_dense_mesh_and_the_broadening(
    ws: Workspace, tmp_path: Path
) -> None:
    options = _si_options(tmp_path)
    with ws.start_run(name="si-dos", intent="overrides"):
        _, info = density_of_states(
            _si(),
            calculator_options=options,
            dos_kspacing=0.02,
            nbands=10,
            delta_e=0.05,
            degauss=0.005,
        )
    nscf = (tmp_path / "inputs" / "nscf.pwi").read_text()
    assert "nbnd             = 10" in nscf
    # ASE turns kspacing into the mesh it writes, so the denser spacing
    # shows up as a denser mesh than the SCF's 7x7x7.
    assert "K_POINTS automatic\n16 16 16" in nscf
    assert info["dos_kspacing"] == 0.02 and info["dos_kpts"] is None


def test_the_broadening_defaults_to_the_scf_smearing(ws: Workspace, tmp_path: Path) -> None:
    options = _si_options(tmp_path)
    with ws.start_run(name="si-dos", intent="default degauss"):
        _, info = density_of_states(
            _si(), calculator_options=options, dos_kpts=[8, 8, 8], delta_e=0.05
        )
    assert info["degauss_ry"] == 0.02
    assert "degauss = 0.02" in (tmp_path / "inputs" / "dos.in").read_text()


def test_fixed_occupations_take_the_named_broadening(ws: Workspace, tmp_path: Path) -> None:
    options = _si_options(tmp_path)
    for key in ("occupations", "smearing", "degauss"):
        options["input_data"]["system"].pop(key)
    options["input_data"]["system"]["occupations"] = "fixed"
    with ws.start_run(name="si-dos", intent="fixed occupations"):
        _, info = density_of_states(
            _si(), calculator_options=options, dos_kpts=[8, 8, 8], delta_e=0.05
        )
    assert info["degauss_ry"] == tasks._DOS_FIXED_DEGAUSS_RY


def test_density_of_states_refuses_what_it_cannot_run(ws: Workspace, tmp_path: Path) -> None:
    options = _si_options(tmp_path)
    with pytest.raises(ValueError, match="needs a Quantum ESPRESSO engine; 'emt'"):
        density_of_states(_si(), engine="emt")
    no_mesh = {k: v for k, v in options.items() if k != "kpts"}
    with pytest.raises(ValueError, match="declares no k-points"):
        density_of_states(_si(), calculator_options=no_mesh)
    relaxing = deepcopy(options)
    relaxing["input_data"]["control"]["calculation"] = "relax"
    with pytest.raises(
        ValueError,
        match="runs its own SCF and then its own NSCF step, but calculator_options "
        "declares calculation='relax'",
    ):
        density_of_states(_si(), calculator_options=relaxing)
    for key, value in (("nspin", 2), ("noncolin", True), ("lspinorb", True)):
        spinning = deepcopy(options)
        spinning["input_data"]["system"][key] = value
        with pytest.raises(
            ValueError, match=f"density_of_states does not support {key}=.* yet, because"
        ):
            density_of_states(_si(), calculator_options=spinning)
    with pytest.raises(ValueError, match="own scratch directory"):
        density_of_states(_si(), calculator_options={**options, "directory": str(tmp_path)})
    with pytest.raises(ValueError, match=r"dos_kpts or dos_kspacing .* not both"):
        density_of_states(
            _si(), calculator_options=options, dos_kpts=[4, 4, 4], dos_kspacing=0.1
        )
    with pytest.raises(ValueError, match="three positive integers"):
        density_of_states(_si(), calculator_options=options, dos_kpts=[4, 4])
    # The occupied count is known after the SCF, and the refusal comes
    # before the NSCF runs.
    with (
        ws.start_run(name="too-few", intent="nbands below the occupied bands"),
        pytest.raises(ValueError, match="nbands=3 holds fewer bands than the 4 occupied"),
    ):
        density_of_states(_si(), calculator_options=options, nbands=3)
    assert not (tmp_path / "inputs" / "nscf.pwi").exists()


@pytest.mark.parametrize("step", ["scf", "nscf", "dos", "projwfc"])
def test_a_failing_step_keeps_its_evidence_under_the_step_name(
    ws: Workspace, tmp_path: Path, scratches: list[Path], step: str
) -> None:
    options = _si_options(tmp_path, fail=step)
    with (
        pytest.raises(Exception) as excinfo,
        ws.start_run(name="si-fail", intent=f"{step} fails") as run,
    ):
        density_of_states(
            _si(),
            calculator_options=options,
            dos_kpts=[8, 8, 8],
            delta_e=0.05,
            degauss=0.005,
            projected=True,
            label="si",
        )
    notes = " ".join(excinfo.value.__notes__)
    assert f"density_of_states failed in its {step} step" in notes
    names = {a.name for a in ws.runs.list_artifacts(run.id)}
    if step in ("scf", "nscf"):
        assert f"si-{step}-failed.pwo" in names
        assert "Error in routine davcio (10)" in notes
    else:
        assert isinstance(excinfo.value, QeToolError)
        assert "Error in routine davcio (10)" in str(excinfo.value)
        assert f"si-{step}-failed.out" in names
    assert ("si-scf.pwo" in names) is (step != "scf")
    assert not scratches[0].exists()


def test_the_callers_options_are_unchanged_and_a_repeat_is_a_cache_hit(
    ws: Workspace, tmp_path: Path
) -> None:
    options = _si_options(tmp_path)
    before = deepcopy(options)
    call = {"dos_kpts": [8, 8, 8], "delta_e": 0.05, "degauss": 0.005, "label": "si"}
    with ws.start_run(name="first", intent="first"):
        first, _ = density_of_states(_si(), calculator_options=options, **call)
    assert options == before
    with ws.start_run(name="again", intent="cache hit") as again:
        second, info = density_of_states(_si(), calculator_options=options, **call)
    assert ws.runs.list_tasks(again.id)[0].cache_hit is True
    assert second == first
    read = read_artifact(ws, run_id=again.id, name=info["artifacts"][-1])
    assert json.loads(read.text)["summary"]["gap"] == 0.5061
    # projected= is a traced argument, so the projected run is its own
    # cache entry and runs projwfc.x.
    with ws.start_run(name="projected", intent="cache split") as third:
        _, projected = density_of_states(
            _si(), calculator_options=options, projected=True, **call
        )
    assert ws.runs.list_tasks(third.id)[0].cache_hit is False
    assert "si-projwfc.out" in projected["artifacts"]


def test_a_short_table_travels_inline(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tasks, "_DOS_INLINE_LIMIT", 10_000)
    options = _si_options(tmp_path)
    with ws.start_run(name="inline", intent="inline limit"):
        dos, _ = density_of_states(
            _si(),
            calculator_options=options,
            dos_kpts=[8, 8, 8],
            delta_e=0.05,
            degauss=0.005,
            projected=True,
            label="si",
        )
    assert len(dos["energies"]) == 450 and "dos_in" not in dos
    assert sorted(dos["projected_dos"]) == ["Si-p", "Si-s"]


def test_a_metal_has_states_at_the_fermi_level(ws: Workspace, tmp_path: Path) -> None:
    root = _fake_bin(tmp_path, AL)
    options = _options(root, tmp_path, "Al", "Al.pbesol-n-kjpaw_psl.1.0.0.UPF")
    with ws.start_run(name="al", intent="metal"):
        _, info = density_of_states(
            bulk("Al", "fcc", a=4.04),
            calculator_options=options,
            dos_kpts=[8, 8, 8],
            delta_e=0.05,
            projected=True,
            label="al",
        )
    assert info["is_metal"] is True
    assert info["gap"] is None and info["gap_kind"] is None
    assert info["dos_at_fermi"] > 0.3
    assert info["projection_groups"] == ["Al-p", "Al-s"]
    assert info["n_bands"] == 6


def test_a_registry_alias_on_the_qe_factory_is_accepted(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fake_bin(tmp_path, SI)
    registry = tmp_path / "engines.json"
    registry.write_text(
        json.dumps(
            {
                "cluster": "test",
                "engines": {
                    "qe-site": {
                        "calculator": "slab.backends.qe_calculator",
                        "options": {
                            "command": str(root / "pw.x"),
                            "pseudo_dir": str(tmp_path),
                        },
                        "version": "7.5",
                    }
                },
            }
        )
    )
    monkeypatch.setenv("SLAB_ENGINES", str(registry))
    located = ("command", "pseudo_dir")
    options = {
        k: v
        for k, v in _options(root, tmp_path, "Si", "Si.pbesol-n-rrkjus_psl.1.0.0.UPF").items()
        if k not in located
    }
    with ws.start_run(name="alias", intent="registry alias"):
        _, info = density_of_states(
            _si(),
            engine="qe-site",
            calculator_options=options,
            dos_kpts=[8, 8, 8],
            delta_e=0.05,
            degauss=0.005,
        )
    assert info["engine"] == "qe-site"
    assert info["gap"] == 0.5061


def test_density_of_states_is_in_the_task_catalog() -> None:
    from foundation._ops import task_catalog

    entry = next(t for t in task_catalog() if t["name"] == "density_of_states")
    assert "density of states" in entry["summary"]


# -- the real thing, when present ------------------------------------------------------


@pytest.mark.skipif(
    not (os.environ.get("SLAB_TEST_PW") and os.environ.get("SLAB_TEST_PSEUDO_DIR")),
    reason="set SLAB_TEST_PW and SLAB_TEST_PSEUDO_DIR to test against a real pw.x",
)
def test_density_of_states_qe_real_integration(ws: Workspace) -> None:
    pw = os.environ["SLAB_TEST_PW"]
    pseudo_dir = os.environ["SLAB_TEST_PSEUDO_DIR"]
    atoms = _si()
    with ws.start_run(name="qe-real-dos", intent="real dos pipeline") as run:
        dos, info = density_of_states(
            atoms,
            dos_kpts=[6, 6, 6],
            delta_e=0.1,
            degauss=0.005,
            projected=True,
            label="si",
            calculator_options={
                "command": pw,
                "pseudo_dir": pseudo_dir,
                "pseudopotentials": resolve_pseudopotentials(atoms, pseudo_dir),
                "kpts": [4, 4, 4],
                "input_data": {
                    "system": {
                        "ecutwfc": 20.0,
                        "occupations": "smearing",
                        "smearing": "cold",
                        "degauss": 0.02,
                    }
                },
            },
        )
    assert info["is_metal"] is False
    assert info["gap"] > 0
    # A coarse mesh and a 0.068 eV Gaussian leave a trace of the valence
    # band inside the gap; the peak of the curve is three orders above it.
    assert info["dos_at_fermi"] < 0.01
    assert info["projection_groups"] == ["Si-p", "Si-s"]
    assert dos["summary"]["gap"] == info["gap"]
    names = {a.name for a in ws.runs.list_artifacts(run.id)}
    assert names == {
        "si-scf.pwo",
        "si-nscf.pwo",
        "si-dos.dat",
        "si-dos.out",
        "si-projwfc.out",
        "si-dos.json",
    }
