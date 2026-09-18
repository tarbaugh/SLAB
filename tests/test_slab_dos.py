"""The density-of-states readers, on real Quantum ESPRESSO 7.5 output.

The fixtures are the files a real run wrote: an SCF and an NSCF on Si
(an insulator) and on Al (a metal), the table dos.x wrote, the per-state
files projwfc.x wrote, and, for the projected bands, projwfc.x's standard
output and its ``atomic_proj.xml``. Error paths truncate a real capture
in the test body; no synthetic QE output is written here.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from slab.dos import (
    PROJECTION_SUM_LIMIT,
    dos_summary,
    group_projections,
    read_dos,
    read_pdos,
    read_projection_states,
    read_projections,
)

DATA = Path(__file__).parent / "data"
SI_TABLE = DATA / "qe-si-dos.dat"
AL_TABLE = DATA / "qe-al-dos.dat"
SI_PDOS = DATA / "qe-si-dos-pdos"
AL_PDOS = DATA / "qe-al-dos-pdos"
SI_PROJWFC = DATA / "qe-si-pbands-projwfc.out"
SI_PROJ_XML = DATA / "qe-si-pbands-atomic_proj.xml"
SI_DOS_JSON = DATA / "qe-si-dos.json"
AL_DOS_JSON = DATA / "qe-al-dos.json"


def _files(directory: Path) -> dict[str, str]:
    return {path.name: path.read_text() for path in sorted(directory.iterdir())}


def test_read_dos_reads_the_grid_and_the_fermi_level() -> None:
    table = read_dos(SI_TABLE.read_text())
    assert len(table["energies"]) == len(table["dos"]) == 450
    assert table["fermi"] == 6.436
    assert table["delta_e"] == 0.05
    assert table["energy_unit"] == "eV" and table["dos_unit"] == "states/eV/cell"
    assert table["energies"][0] == pytest.approx(-6.033)
    # The integrated dos ends at the electron count of the cell: two Si
    # atoms hold 8 valence electrons, and the grid also covers the 8
    # conduction states the NSCF computed.
    assert table["integrated_dos"][0] == pytest.approx(0.0, abs=1e-6)
    assert table["integrated_dos"][-1] == pytest.approx(16.0, abs=0.01)


def test_silicon_has_a_gap_in_the_dos_and_aluminium_has_none() -> None:
    si = read_dos(SI_TABLE.read_text())
    energies, dos = np.array(si["energies"]), np.array(si["dos"])
    # The band structure fixture puts the gap at about half an eV above
    # the valence band top at 6.16 eV. The curve is flat there.
    inside = (energies > 6.25) & (energies < 6.60)
    assert dos[inside].max() < 0.02
    assert dos.max() > 3.0
    al = read_dos(AL_TABLE.read_text())
    at_fermi = float(np.interp(al["fermi"], al["energies"], al["dos"]))
    assert at_fermi > 0.3


def test_read_dos_refuses_a_spin_polarized_or_a_cut_table() -> None:
    lines = SI_TABLE.read_text().splitlines()
    spinning = "#  E (eV)  dosup(E)  dosdw(E)  Int dos(E) EFermi = 6.436 eV"
    with pytest.raises(ValueError, match="spin-polarized"):
        read_dos("\n".join([spinning, *lines[1:5]]))
    with pytest.raises(ValueError, match="no '#' header line"):
        read_dos("\n".join(lines[1:5]))
    with pytest.raises(ValueError, match="1 row"):
        read_dos("\n".join(lines[:2]))
    cut = "\n".join([*lines[:4], "  -6.4  0.2908E-07"])
    with pytest.raises(ValueError, match="column"):
        read_dos(cut)


def test_read_pdos_sums_the_atoms_and_the_shells_into_groups() -> None:
    read = read_pdos(_files(SI_PDOS))
    assert read["group_names"] == ["Si-p", "Si-s"]
    assert len(read["energies"]) == 451
    grouped = sum(np.array(curve) for curve in read["groups"].values())
    total = np.array(read["total_projected"])
    # The group curves are the projected total, row for row: the groups
    # hold every state of every atom. The files print 3 significant
    # digits, so the sum of five of them carries that much rounding.
    assert np.abs(grouped - total).max() < 0.02
    # The projection misses part of the total, because pseudo-atomic
    # orbitals are not a complete basis.
    assert grouped.max() < np.array(read["total"]).max()
    assert grouped.max() > 0.8 * np.array(read["total"]).max()


def test_read_pdos_reads_a_metal_and_refuses_what_it_cannot_read() -> None:
    assert read_pdos(_files(AL_PDOS))["group_names"] == ["Al-p", "Al-s"]
    one = _files(SI_PDOS)
    name = "si.pdos_atm#1(Si)_wfc#1(s)"
    with pytest.raises(ValueError, match="no per-state file"):
        read_pdos({k: v for k, v in one.items() if k.endswith("pdos_tot")})
    with pytest.raises(ValueError, match=re.escape("not a projwfc.x pdos file name")):
        read_pdos({"si.ldos": one[name]})
    spinning = one[name].replace("ldos(E)", "ldosup(E)  ldosdw(E)", 1)
    with pytest.raises(ValueError, match="is spin-polarized"):
        read_pdos({name: spinning})
    cut = "\n".join(one[name].splitlines()[:6])
    with pytest.raises(ValueError, match="rows, but an earlier file"):
        read_pdos({**one, name: cut})


def test_read_projection_states_names_every_orbital() -> None:
    states = read_projection_states(SI_PROJWFC.read_text())
    assert len(states) == 8
    assert [state["group"] for state in states] == ["Si-s", *["Si-p"] * 3] * 2
    assert states[1] == {
        "state": 2,
        "atom": 1,
        "element": "Si",
        "wfc": 2,
        "l": 1,
        "m": 1,
        "group": "Si-p",
    }
    with pytest.raises(ValueError, match="prints no 'state #' line"):
        read_projection_states("     Program PROJWFC v.7.5\n")
    relativistic = "     state #   1: atom   1 (Si ), wfc  1 (j=0.5 l=0 m_j=-0.5)"
    with pytest.raises(ValueError, match="noncollinear or spin-orbit"):
        read_projection_states(relativistic)


def test_read_projections_reads_every_band_of_every_k_point() -> None:
    read = read_projections(SI_PROJ_XML.read_text())
    assert (read["n_kpoints"], read["n_bands"], read["n_states"]) == (20, 8, 8)
    assert read["n_electrons"] == 8.0
    assert read["fermi"] == pytest.approx(6.3356, abs=1e-3)
    assert len(read["weights"]) == 20
    assert len(read["weights"][0]) == 8 and len(read["weights"][0][0]) == 8
    # The eigenvalues come back in eV and match the bands step's own
    # listing at Γ: one s-like level far below, then three degenerate
    # levels at the valence band top.
    assert read["energies"][0][0] == pytest.approx(-5.8293, abs=1e-3)
    assert read["energies"][0][3] == pytest.approx(6.1642, abs=1e-3)


def test_read_projections_refuses_a_file_it_cannot_trust() -> None:
    text = SI_PROJ_XML.read_text()
    with pytest.raises(ValueError, match="does not parse as XML"):
        read_projections(text[: len(text) // 2])
    with pytest.raises(ValueError, match="root element"):
        read_projections("<OTHER></OTHER>")
    spinning = text.replace('NUMBER_OF_SPIN_COMPONENTS="1"', 'NUMBER_OF_SPIN_COMPONENTS="2"', 1)
    with pytest.raises(ValueError, match="2 spin components"):
        read_projections(spinning)
    short = text.replace('NUMBER_OF_K-POINTS="20"', 'NUMBER_OF_K-POINTS="21"', 1)
    with pytest.raises(ValueError, match="cut short"):
        read_projections(short)


def test_group_projections_gives_the_fat_band_weights() -> None:
    states = read_projection_states(SI_PROJWFC.read_text())
    read = read_projections(SI_PROJ_XML.read_text())
    groups = group_projections(read["weights"], states)
    assert sorted(groups) == ["Si-p", "Si-s"]
    weights = np.array(groups["Si-s"]) + np.array(groups["Si-p"])
    assert weights.shape == (20, 8)
    # The weights of one band sum to at most one, and to nearly one for
    # the valence bands: pseudo-atomic orbitals span them well.
    assert weights.max() <= 1.0
    assert weights[:, :4].min() > 0.9
    # The valence band top at Γ is p, as diamond-structure Si is.
    assert groups["Si-p"][0][3] == pytest.approx(0.961, abs=1e-3)
    assert groups["Si-s"][0][3] < 1e-6
    # The lowest band at Γ is the s combination instead.
    assert groups["Si-s"][0][0] == pytest.approx(0.996, abs=1e-3)


def test_group_projections_refuses_a_mismatched_or_impossible_file() -> None:
    states = read_projection_states(SI_PROJWFC.read_text())
    read = read_projections(SI_PROJ_XML.read_text())
    with pytest.raises(ValueError, match="state list holds 4"):
        group_projections(read["weights"], states[:4])
    doubled = (np.array(read["weights"]) * 4.0).tolist()
    with pytest.raises(ValueError, match=re.escape(f"above the {PROJECTION_SUM_LIMIT}")):
        group_projections(doubled, states)


def test_dos_summary_reads_the_curve_and_the_eigenvalues() -> None:
    table = read_dos(SI_TABLE.read_text())
    written = json.loads(SI_DOS_JSON.read_text())
    summary = dos_summary(
        table["energies"],
        table["dos"],
        written["fermi"],
        eigenvalues=[[6.0, 7.0], [5.0, 7.5]],
        kpoints=[[0, 0, 0], [0.5, 0, 0.5]],
    )
    assert summary["dos_at_fermi"] == pytest.approx(written["summary"]["dos_at_fermi"])
    assert summary["dos_at_fermi"] < 1e-4  # the Fermi level sits in the gap
    assert summary["delta_e"] == 0.05 and summary["n_points"] == 450
    assert summary["energy_range"] == [table["energies"][0], table["energies"][-1]]
    # The verdict reads the eigenvalues, never the broadened curve.
    assert summary["is_metal"] is False and summary["gap"] == 1.0
    # A mesh names no special point, so the band edges carry the k alone.
    assert summary["vbm_at"]["label"] is None and summary["vbm_at"]["distance"] is None
    assert summary["vbm_at"]["k"] == [0.0, 0.0, 0.0]


def test_dos_summary_without_eigenvalues_reports_no_gap() -> None:
    table = read_dos(AL_TABLE.read_text())
    summary = dos_summary(table["energies"], table["dos"], table["fermi"])
    assert summary["is_metal"] is None and summary["gap"] is None
    assert "no eigenvalues" in summary["note"]
    assert summary["dos_at_fermi"] == pytest.approx(
        json.loads(AL_DOS_JSON.read_text())["summary"]["dos_at_fermi"], rel=1e-3
    )
    with pytest.raises(ValueError, match="at least two grid rows"):
        dos_summary([0.0], [1.0], 0.0)
