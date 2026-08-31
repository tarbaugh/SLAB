"""Tests for the build_structure task (a fake atomsk keeps them fast).

The fake replays a real atomsk capture (see test_slab_atomsk.py); here is
the task contract: structures read back honestly, inputs staged, output
selection, artifact and evidence behavior, and caching on the builder's
identity. A gated test runs the real binary when $SLAB_TEST_ATOMSK names it.
"""

import os
from pathlib import Path

import pytest
from ase.build import bulk

from foundation import Workspace
from foundation.tasks import build_structure
from slab.errors import BuilderError

DATA = Path(__file__).parent / "data"
XSF = DATA / "atomsk-al-fcc-222.xsf"
CREATE_LOG = DATA / "atomsk-create.log"
BAD_CREATE_LOG = DATA / "atomsk-bad-create.log"


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    with Workspace(tmp_path / "ws") as workspace:
        yield workspace


def _script(path: Path, body: str) -> str:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(0o755)
    return str(path)


@pytest.fixture()
def fake_atomsk(tmp_path: Path) -> str:
    return _script(
        tmp_path / "fake-atomsk",
        f"""\
if [ "$1" = "--version" ]; then
  echo "(C) P. Hirel 2010 - Version master-2026-07-24 (Beta)"; exit 0
fi
for a in "$@"; do
  if [ "$a" = "Xx" ]; then cat "{BAD_CREATE_LOG}"; exit 0; fi
done
for a in "$@"; do out="$a"; done
cp "{XSF}" "$out"
cat "{CREATE_LOG}"
""",
    )


ARGS = "--create fcc 4.046 Al -duplicate 2 2 2 al.xsf"


def test_build_returns_the_recorded_supercell(fake_atomsk: str) -> None:
    atoms, info = build_structure(ARGS, command=fake_atomsk)
    assert len(atoms) == 32
    assert atoms.get_chemical_formula() == "Al32"
    assert atoms.cell.lengths() == pytest.approx([8.092, 8.092, 8.092])
    assert info["builder"] == "atomsk"
    assert info["version"] == "master-2026-07-24"
    assert info["args"][0] == "--create"  # the string form was shell-split
    assert info["output"] == "al.xsf"
    assert info["produced"] == ["al.xsf"]
    assert info["n_atoms"] == 32
    assert info["pbc"] == [True, True, True]


def test_build_inside_a_run_keeps_file_and_log(ws: Workspace, fake_atomsk: str) -> None:
    with ws.start_run(name="build") as run:
        build_structure(ARGS, command=fake_atomsk, label="al-222")
    names = sorted(a.name for a in ws.runs.list_artifacts(run.id))
    assert names == ["al-222.log", "al-222.xsf"]


def test_failure_attaches_notes_and_keeps_the_failed_log(
    ws: Workspace, fake_atomsk: str
) -> None:
    with pytest.raises(BuilderError) as excinfo, ws.start_run(name="doomed") as run:
        build_structure("--create fcc 4.046 Xx bad.xsf", command=fake_atomsk)
    assert "non-conform statement" in str(excinfo.value)
    notes = getattr(excinfo.value, "__notes__", [])
    assert any("build-failed.log" in note for note in notes)
    names = [a.name for a in ws.runs.list_artifacts(run.id)]
    assert "build-failed.log" in names


def test_inputs_stage_atoms_and_text(tmp_path: Path) -> None:
    """Both value kinds land in the scratch directory before atomsk runs:
    the fake refuses unless the staged structure and parameter file exist."""
    checker = _script(
        tmp_path / "checking-atomsk",
        f"""\
if [ "$1" = "--version" ]; then echo "Version 1"; exit 0; fi
test -f seed.xsf || {{ echo "X!X ERROR: seed.xsf not staged"; exit 0; }}
test -f poly.txt || {{ echo "X!X ERROR: poly.txt not staged"; exit 0; }}
grep -q "box 40" poly.txt || {{ echo "X!X ERROR: poly.txt content wrong"; exit 0; }}
cp "{XSF}" out.xsf
echo done
""",
    )
    atoms, info = build_structure(
        ["--polycrystal", "seed.xsf", "poly.txt", "out.xsf"],
        inputs={"seed.xsf": bulk("Al", "fcc", a=4.046), "poly.txt": "box 40 40 40\n"},
        command=checker,
    )
    # staged inputs are not "produced" — only atomsk's own output is
    assert info["produced"] == ["out.xsf"]
    assert len(atoms) == 32


def test_staged_names_must_be_bare(fake_atomsk: str) -> None:
    for bad in ("../seed.xsf", "a/b.xsf", ".hidden", "~seed", ""):
        with pytest.raises(BuilderError, match="bare file name"):
            build_structure(ARGS, inputs={bad: bulk("Al", "fcc", a=4.0)}, command=fake_atomsk)


def test_several_produced_files_need_an_explicit_output(tmp_path: Path) -> None:
    twin = _script(
        tmp_path / "twin-atomsk",
        f"""\
if [ "$1" = "--version" ]; then echo "Version 1"; exit 0; fi
cp "{XSF}" al.xsf
cp "{XSF}" al.cfg
echo done
""",
    )
    with pytest.raises(BuilderError, match="pass output="):
        build_structure(["al.xsf", "cfg"], command=twin)
    _atoms, info = build_structure(["al.xsf", "cfg"], command=twin, output="al.xsf")
    assert info["output"] == "al.xsf"
    assert sorted(info["produced"]) == ["al.cfg", "al.xsf"]


def test_naming_an_output_atomsk_never_wrote_is_refused(tmp_path: Path) -> None:
    silent = _script(
        tmp_path / "silent-atomsk",
        'if [ "$1" = "--version" ]; then echo "Version 1"; exit 0; fi\necho done\n',
    )
    with pytest.raises(BuilderError, match="was not produced"):
        build_structure(["al.xsf"], command=silent, output="al.xsf")
    with pytest.raises(BuilderError, match="without writing any file"):
        build_structure(["al.xsf"], command=silent)


def test_traced_build_caches_on_identical_calls(ws: Workspace, fake_atomsk: str) -> None:
    with ws.start_run() as first:
        build_structure(ARGS, command=fake_atomsk)
    with ws.start_run() as second:
        build_structure(ARGS, command=fake_atomsk)
    assert ws.runs.list_tasks(first.id)[0].cache_hit is False
    assert ws.runs.list_tasks(second.id)[0].cache_hit is True


def test_a_different_binary_misses_the_cache(
    ws: Workspace, fake_atomsk: str, tmp_path: Path
) -> None:
    """cache_extra folds the resolved command into the key: the same argument
    list run under a different atomsk is a different computation."""
    other = _script(
        tmp_path / "other-atomsk",
        f"""\
if [ "$1" = "--version" ]; then
  echo "(C) P. Hirel 2010 - Version 0.99 (Beta)"; exit 0
fi
for a in "$@"; do out="$a"; done
cp "{XSF}" "$out"
echo done
""",
    )
    with ws.start_run():
        build_structure(ARGS, command=fake_atomsk)
    with ws.start_run() as second:
        build_structure(ARGS, command=other)
    assert ws.runs.list_tasks(second.id)[0].cache_hit is False


def test_build_feeds_relax_directly(fake_atomsk: str) -> None:
    from foundation.tasks import relax

    atoms, _ = build_structure(ARGS, command=fake_atomsk)
    relaxed, info = relax(atoms, engine="emt", fmax=0.1)
    assert info["converged"] is True
    assert len(relaxed) == 32


# -- the real binary, when present --------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("SLAB_TEST_ATOMSK"),
    reason="set SLAB_TEST_ATOMSK to a real atomsk executable",
)
def test_real_atomsk_builds_the_fixture_supercell(ws: Workspace) -> None:
    command = os.environ["SLAB_TEST_ATOMSK"]
    with ws.start_run(name="real-atomsk") as run:
        atoms, info = build_structure(ARGS, command=command, label="al-222")
    assert len(atoms) == 32
    assert atoms.cell.lengths() == pytest.approx([8.092, 8.092, 8.092])
    assert info["version"] is not None
    names = sorted(a.name for a in ws.runs.list_artifacts(run.id))
    assert names == ["al-222.log", "al-222.xsf"]


def test_lammps_data_output_is_read_under_its_explicit_format(tmp_path: Path) -> None:
    """ASE cannot infer atomsk's .lmp extension; the task names the format.
    The fixture is a real atomsk conversion of the recorded supercell."""
    lmp = DATA / "atomsk-al-fcc-222.lmp"
    writer = _script(
        tmp_path / "lmp-atomsk",
        f"""\
if [ "$1" = "--version" ]; then echo "Version 1"; exit 0; fi
cp "{lmp}" al.lmp
echo done
""",
    )
    atoms, info = build_structure(["al.xsf", "lmp"], command=writer)
    assert info["output"] == "al.lmp"
    assert len(atoms) == 32
    assert atoms.cell.lengths() == pytest.approx([8.092, 8.092, 8.092])


def test_an_unreadable_output_names_the_file_in_a_note(tmp_path: Path) -> None:
    junk = _script(
        tmp_path / "junk-atomsk",
        'if [ "$1" = "--version" ]; then echo "Version 1"; exit 0; fi\n'
        'echo "not a structure" > out.xsf\necho done\n',
    )
    with pytest.raises(Exception) as excinfo:
        build_structure(["out.xsf"], command=junk)
    notes = getattr(excinfo.value, "__notes__", [])
    assert any("cannot read it back" in note for note in notes)
