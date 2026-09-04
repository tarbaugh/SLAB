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


def test_lammps_log_digest_reads_the_ase_driven_capture() -> None:
    text = (DATA / "lammps-cu-relax-final.log").read_text()
    shown = lammps_log_digest("lammps-cu-relax-final.log", text)
    lines = shown.splitlines()
    assert lines[0].startswith("LAMMPS log digest: lammps-cu-relax-final.log (77 lines, LAMMPS ?, ")
    assert "1 loop(s) completed" in lines[0]
    assert lines[1] == "setup: units metal; 8 atoms; pair_style eam"
    assert lines[2].startswith("thermo table 1 (1 rows): Step Temp Press CPU Pxx")
    assert lines[3].startswith("  first: 0 0 17.45234162594413")
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
