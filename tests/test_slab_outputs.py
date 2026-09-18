"""Digests of engine outputs, on the real captures under tests/data.

One real session read a 305 KB pw.x output in 400-line windows, took a
block of band eigenvalues in eV for a diverging SCF energy in Ry, and
compacted six times in sixteen minutes arguing with itself. The digest is
the summary a colleague gives first; the bands never appear in it.
"""

from __future__ import annotations

from pathlib import Path

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.io import write

from slab.outputs import digest, extxyz_digest, lammps_log_digest, pwscf_digest

DATA = Path(__file__).parent / "data"


def test_pwscf_digest_reads_the_si_relax_capture() -> None:
    text = (DATA / "qe-si-relax-final.pwo").read_text()
    shown = pwscf_digest("qe-si-relax-final.pwo", text)
    lines = shown.splitlines()
    assert lines[0] == (
        "pw.x output digest: qe-si-relax-final.pwo "
        "(248 lines, PWSCF v.7.4.1, finished: JOB DONE)"
    )
    assert lines[1] == "system: 2 atoms, 8.00 electrons, 4 KS states, volume 270.1072 bohr^3"
    assert "ecutwfc 14.0000 Ry; ecutrho 56.0000 Ry; 8 k-points; mixing beta 0.7000" in lines[2]
    assert "scf: 1 cycle (single point)" in shown
    assert "converged in 4 iterations; trace (Ry): -15.6351, -15.6396, -15.6400" in shown
    assert "final ! -15.64003672 Ry; accuracy < 9.10e-07 Ry" in shown
    assert "forces: max |component| 0.000128 Ry/bohr (0.0033 eV/Å)" in shown
    assert "Total force 0.000182, Total SCF correction 0.000030" in shown
    assert "wall: 0.05s (PWSCF total)" in shown
    assert "warnings: none" in shown  # pw.x's "card &CELL ignored" lines are noise, not warnings
    assert "6.3626" not in shown and "bands" not in shown  # the eigenvalue blocks never appear
    assert len(shown) < 1_000


def test_pwscf_digest_names_a_failed_convergence_and_a_cut_off_run() -> None:
    text = (DATA / "qe-si-relax-final.pwo").read_text()
    failed = text.replace(
        "convergence has been achieved in   4 iterations",
        "convergence NOT achieved after   4 iterations: stopping",
    ).replace("JOB DONE.", "")
    shown = pwscf_digest("bad.pwo", failed)
    assert "NOT finished: no JOB DONE line" in shown.splitlines()[0]
    assert "NOT converged after 4 iterations" in shown
    assert "warnings: convergence NOT achieved after   4 iterations: stopping" in shown
    cut = text[: text.index("End of self-consistent calculation")]
    shown = pwscf_digest("cut.pwo", cut)
    assert "no convergence line (cut off?)" in shown
    assert "forces: none printed" in shown


def test_pwscf_digest_carries_an_error_block() -> None:
    text = (DATA / "qe-si-relax-final.pwo").read_text()
    fence = "%" * 77
    error = (
        f"\n {fence}\n"
        "     Error in routine electrons (1):\n     charge is wrong: smearing is needed\n"
        f" {fence}\n"
    )
    shown = pwscf_digest("err.pwo", text[:2000] + error)
    assert "errors: Error in routine electrons (1): | charge is wrong: smearing is needed" in shown


def test_pwscf_digest_summarises_a_relaxation_by_cycle() -> None:
    text = (DATA / "qe-si-relax-final.pwo").read_text()
    start = text.index("     iteration #  1")
    scf = text[start : text.index("     Writing all to output data dir")]
    three = text.replace(scf, scf + scf.replace("-15.64003672", "-15.64100000") + scf)
    three += "\n     bfgs converged in   3 scf cycles and   2 bfgs steps\n"
    shown = pwscf_digest("relax.pwo", three)
    assert "scf: 3 cycles (a relaxation), 3 converged" in shown
    assert "cycle energies (Ry): -15.640037, -15.641000, -15.640037" in shown
    assert "bfgs converged in 3 scf cycles and 2 bfgs steps" in shown


def test_pwscf_digest_reads_a_bands_run_without_its_rows() -> None:
    text = (DATA / "qe-si-bands-bands.pwo").read_text()
    out = digest("si-bands.pwo", text)
    assert out is not None
    lines = out.splitlines()
    assert lines[0].startswith("pw.x output digest: si-bands.pwo (1295 lines, PWSCF v.7.5, ")
    assert (
        "bands: 60 k-points, 8 bands, eigenvalues -5.8319 to 16.2188 eV; the numbers "
        "are in si-bands.json (band_structure's result file)"
    ) in lines
    assert not any(line.startswith(("scf:", "forces:", "fermi energy:")) for line in lines)
    # No eigenvalue row reaches the reader: the first k-point's energies
    # appear in the file and nowhere in the digest.
    assert "6.1597   6.1597" in text and "6.1597" not in out
    assert len(lines) < 10
    other = digest("espresso.pwo", text)
    assert other is not None and "the numbers are in the run's -bands.json" in other


def test_pwscf_digest_of_the_bands_scf_is_an_ordinary_scf() -> None:
    out = digest("si-scf.pwo", (DATA / "qe-si-bands-scf.pwo").read_text())
    assert out is not None
    assert "scf: 1 cycle (single point)" in out
    assert "fermi energy: 6.1880 eV" in out
    assert "bands:" not in out


def test_lammps_log_digest_reads_the_ase_driven_capture() -> None:
    text = (DATA / "lammps-cu-relax-final.log").read_text()
    shown = lammps_log_digest("lammps-cu-relax-final.log", text)
    lines = shown.splitlines()
    assert lines[0].startswith("LAMMPS log digest: lammps-cu-relax-final.log (77 lines, LAMMPS ?, ")
    assert "1 loop(s) completed" in lines[0]
    assert lines[1] == "setup: units metal; 8 atoms; pair_style eam"
    assert lines[2] == "thermo: text"
    assert lines[3].startswith("thermo table 1 (1 rows): Step Temp Press CPU Pxx")
    assert lines[4].startswith("  first: 0 0 17.45234162594413")
    assert "  loop: 0 steps, 8 atoms, 1 procs, 2.92e-07 s" in shown
    assert "warnings: WARNING: Triclinic box skew is large." in shown
    assert digest("lammps-cu-relax-final.log", text) == shown


def test_lammps_log_digest_shows_both_ends_of_a_long_table() -> None:
    rows = "\n".join(
        f"{step} {300 + step / 10:.1f} {-1.0 - step / 1000:.4f}" for step in range(0, 5001, 100)
    )
    text = (
        "LAMMPS (2 Aug 2023)\nunits metal\nCreated 32 atoms\npair_style eam/alloy\n"
        f"   Step Temp PotEng\n{rows}\nLoop time of 1.5 on 4 procs for 5000 steps with 32 atoms\n"
        "Total wall time: 0:00:02\n"
    )
    shown = lammps_log_digest("run.log", text)
    assert "LAMMPS 2 Aug 2023, finished: Total wall time 0:00:02" in shown
    assert "thermo table 1 (51 rows): Step Temp PotEng" in shown
    assert "  first: 0 300.0 -1.0000" in shown and "  last:  5000 800.0 -6.0000" in shown
    assert "  loop: 5000 steps, 32 atoms, 4 procs, 1.5 s" in shown
    assert "warnings: none" in shown


def test_extxyz_digest_counts_frames_and_labels(tmp_path: Path) -> None:
    frames = []
    for a in (3.5, 3.6, 3.7):
        atoms = bulk("Cu", "fcc", a=a) * (2, 1, 1)
        atoms.calc = EMT()
        atoms.get_forces()
        frames.append(atoms)
    path = tmp_path / "cu.extxyz"
    write(path, frames, format="extxyz")
    text = path.read_text()
    shown = extxyz_digest("cu.extxyz", text)
    assert shown.splitlines()[0].startswith("extended XYZ digest: cu.extxyz (3 frames, ")
    assert "atoms per frame: 2; species: Cu" in shown
    assert "energy: present on 3 frame(s)" in shown
    assert "forces: present; lattice: present" in shown
    # fcc nearest neighbour is a/sqrt(2): 3.5/1.41421 = 2.475 Å, through the periodic images
    assert "spacing: closest pair 2.475 Å (Cu-Cu, frame 0), plausible" in shown
    assert "covalent-radii sum 2.64 Å" in shown
    assert "over all 3 frames" in shown
    assert digest("cu.extxyz", text) == shown


def test_extxyz_digest_flags_an_unphysical_pair(tmp_path: Path) -> None:
    """A training set with two atoms 0.4 Å apart cannot have been labelled
    honestly; the digest names the pair and the frame before anyone fits to it."""
    good = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
    bad = good.copy()
    bad.positions[1] = bad.positions[0] + (0.4, 0.0, 0.0)
    path = tmp_path / "set.extxyz"
    write(path, [good, bad, good], format="extxyz")
    shown = extxyz_digest("set.extxyz", path.read_text())
    assert "spacing: closest pair 0.400 Å (Cu-Cu, frame 1), SUSPECT: under 60% of" in shown
    assert "mean nearest-neighbour distance" in shown


def test_extxyz_digest_survives_a_file_ase_rejects() -> None:
    text = "2\nLattice=\"3 0 0 0 3 0 0 0 3\" Properties=species:S:1:pos:R:3\nCu 0 0 0\nCu 1.5 x 0\n"
    shown = extxyz_digest("odd.extxyz", text)
    assert "spacing: not computed (ASE could not parse the file" in shown


def test_digest_declines_what_it_does_not_recognise() -> None:
    assert digest("PLAN.md", "# Plan\n1. relax\n") is None
    assert digest("numbers.txt", "3\n1 2 3\n") is None


def test_digest_never_reads_a_script_as_a_lammps_log() -> None:
    """A workflow script that builds a LAMMPS input mentions units, styles,
    and thermo lines in its text. The format is decided by the name and the
    header, so the script is shown as text."""
    script = (
        "from ase.calculators.lammpsrun import LAMMPS\n"
        "cmds = ['units metal', 'atom_style atomic', 'pair_style eam/alloy', 'thermo 10']\n"
        "thermo_style = 'custom step pe'\n"
    )
    assert digest("make_input.py", script) is None
    assert digest("parse_pw.py", "MARK = 'Program PWSCF'\n") is None
    # The same text under a log name still opens with a Python line, not a command.
    assert digest("make_input.log", script) is None
    # The ASE-driven capture opens with its echoed commands and still digests.
    text = (DATA / "lammps-cu-relax-final.log").read_text()
    assert digest("lammps-cu-relax-final.log", text) is not None


def test_lammps_ave_time_reads_the_scalar_and_vector_captures() -> None:
    """Real files a LAMMPS 22 Jul 2025 build wrote: a scalar `fix ave/time` of
    temperature and pressure every 100 steps over 1500 steps, and a vector
    one of a 10-bin `compute rdf`."""
    from slab.outputs import lammps_ave_time, lammps_ave_time_digest

    scalar = (DATA / "lammps-ar-ave-time-scalar.dat").read_text()
    parsed = lammps_ave_time(scalar)
    assert parsed["fix"] == "avg" and parsed["mode"] == "scalar"
    assert parsed["columns"] == ["TimeStep", "c_thermo_temp", "c_thermo_press"]
    assert len(parsed["rows"]) == 15
    assert parsed["rows"][0] == [100, 168.698, 4991.83]
    assert parsed["rows"][-1][0] == 1500
    shown = lammps_ave_time_digest("probe-avg.dat", scalar)
    assert shown.splitlines()[0] == (
        "fix ave/time digest: probe-avg.dat (fix avg, scalar mode, 15 rows)"
    )
    assert "first: 100 168.698 4991.83" in shown
    assert shown.splitlines()[-1].startswith("last:  1500 ")
    assert digest("probe-avg.dat", scalar) == shown  # the dispatcher recognises the file

    vector = (DATA / "lammps-ar-ave-time-vector.dat").read_text()
    parsed = lammps_ave_time(vector)
    assert parsed["fix"] == "vec" and parsed["mode"] == "vector"
    assert parsed["columns"][:2] == ["Row", "c_rdf[1]"] and parsed["rows"] == []
    shown = digest("vec.dat", vector)
    assert shown is not None
    assert "(fix vec, vector mode, 2 block(s); the blocks are in the file itself)" in shown
    assert "columns: Row c_rdf[1] c_rdf[2] c_rdf[3]" in shown


# -- YAML thermo (thermo_modify line yaml) -------------------------------------


def test_lammps_thermo_reads_a_two_run_yaml_log() -> None:
    """Both documents of a real two-run log (LAMMPS 22 Jul 2025 - Update 4),
    each with its loop line, in the text parser's shape."""
    from slab.outputs import lammps_thermo, lammps_thermo_format, lammps_yaml_thermo

    text = (DATA / "lammps-cu-two-runs-yaml.log").read_text()
    tables = lammps_thermo(text)
    assert lammps_yaml_thermo(text) == tables
    assert lammps_thermo_format(text) == "yaml"
    assert [t["columns"] for t in tables] == [
        ["Step", "Temp", "PotEng", "KinEng", "TotEng", "Press", "Volume"],
        ["Step", "Temp", "PotEng", "Press", "c_thermo_temp", "v_x"],
    ]
    assert [len(t["rows"]) for t in tables] == [4, 3]
    assert tables[0]["rows"][0][:2] == [0, 300.0]
    assert all(isinstance(row[0], int) for t in tables for row in t["rows"])
    assert all(isinstance(v, float) for t in tables for row in t["rows"] for v in row[1:])
    assert tables[0]["loop"]["steps"] == 60 and tables[1]["loop"]["steps"] == 40
    assert tables[1]["rows"][-1][-1] == 1.5  # v_x
    assert tables[0]["rows"][-1][0] == 60 and tables[1]["rows"][0][0] == 60


def test_lammps_thermo_reads_a_mixed_log_in_order() -> None:
    """A text table, then `thermo_modify line yaml`, then a YAML table: both
    parse, in log order, and the YAML document is never read as text rows."""
    from slab.outputs import lammps_thermo, lammps_thermo_format, lammps_yaml_thermo

    text = (DATA / "lammps-cu-mixed-thermo.log").read_text()
    tables = lammps_thermo(text)
    assert lammps_thermo_format(text) == "mixed"
    assert len(tables) == 2 and len(lammps_yaml_thermo(text)) == 1
    assert tables[0]["columns"] == tables[1]["columns"]
    assert [row[0] for row in tables[0]["rows"]] == [0, 20, 40]
    assert [row[0] for row in tables[1]["rows"]] == [40, 60, 80]
    assert tables[0]["rows"][-1][0] == tables[1]["rows"][0][0] == 40
    assert tables[0]["loop"]["steps"] == tables[1]["loop"]["steps"] == 40


def test_lammps_thermo_reads_a_minimize_yaml_log() -> None:
    from slab.outputs import lammps_thermo

    text = (DATA / "lammps-cu-minimize-yaml.log").read_text()
    (table,) = lammps_thermo(text)
    assert table["columns"] == ["Step", "Temp", "PotEng", "Press"]
    assert len(table["rows"]) == 2 and table["loop"]["steps"] == 1
    assert table["minimize"] is True  # the Minimization stats block marks it


def test_lammps_yaml_thermo_survives_an_unclosed_document_and_odd_values() -> None:
    """LAMMPS died inside a run: the `...` never came, the rows still count.
    Values PyYAML leaves as text (1e+20, inf, -nan) become floats."""
    from slab.outputs import lammps_thermo, lammps_thermo_format

    text = (
        "---\nkeywords: ['Step', 'Temp', 'Press', ]\ndata:\n"
        "  - [0, 300, 1e+20, ]\nfix print says hello\n  - [10, inf, -nan, ]\n"
        "ERROR: Lost atoms: original 4 current 3 (src/thermo.cpp:1)\n"
    )
    (table,) = lammps_thermo(text)
    assert lammps_thermo_format(text) == "yaml"
    assert table["rows"][0] == [0, 300.0, 1e20]
    assert table["rows"][1][1] == float("inf") and table["rows"][1][2] != table["rows"][1][2]
    assert table["loop"] is None
    # The timing breakdown's rule of dashes opens no document.
    assert lammps_thermo("-" * 63 + "\nPair | 0.1 | 0.1 | 0.1 | 0.0 | 90.0\n") == []


def test_lammps_log_digest_of_a_yaml_log_shows_no_raw_yaml() -> None:
    text = (DATA / "lammps-cu-two-runs-yaml.log").read_text()
    shown = lammps_log_digest("two.log", text)
    assert "keywords:" not in shown and "- [" not in shown and "data:" not in shown
    lines = shown.splitlines()
    assert lines[1] == "setup: units metal; 108 atoms; pair_style lj/cut 6.0"
    assert lines[2] == "thermo: yaml"
    assert lines[3] == "thermo table 1 (4 rows): Step Temp PotEng KinEng TotEng Press Volume"
    assert lines[4] == "  first: 0 300 -349.78363 4.1492507 -345.63438 36633.041 1275.5241"
    assert lines[5].startswith("  last:  60 188.00905 ")
    assert lines[6].startswith("  loop: 60 steps, 108 atoms, 1 procs, ")
    assert lines[7] == "thermo table 2 (3 rows): Step Temp PotEng Press c_thermo_temp v_x"
    assert "warnings: WARNING: New thermo_style command" in shown
    mixed = lammps_log_digest("mixed.log", (DATA / "lammps-cu-mixed-thermo.log").read_text())
    assert "thermo: mixed" in mixed and "- [" not in mixed
