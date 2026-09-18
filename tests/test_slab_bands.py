"""slab.bands on real pw.x 7.5 output.

The fixtures are a real band structure of primitive Si and of fcc Al: a
protocol SCF (SSSP PBEsol efficiency, balanced) and then
``calculation='bands'`` on ASE's default fcc path with 60 k-points, both
run through :func:`foundation.tasks.band_structure`.
"""

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from slab.bands import band_path, band_summary, path_distances, read_bands, scf_levels

DATA = Path(__file__).parent / "data"
SI_SCF = DATA / "qe-si-bands-scf.pwo"
SI_BANDS = DATA / "qe-si-bands-bands.pwo"
AL_SCF = DATA / "qe-al-bands-scf.pwo"
AL_BANDS = DATA / "qe-al-bands-bands.pwo"


def _si() -> Atoms:
    return bulk("Si", "diamond", a=5.43)


# -- band_path ----------------------------------------------------------------------


def test_band_path_takes_seekpaths_fcc_path_and_labels() -> None:
    found = band_path(_si(), npoints=60)
    assert found["path"] == "GXU,KGLWX"
    assert (found["lattice"], found["spacegroup"], found["spacegroup_number"]) == (
        "cF2",
        "Fd-3m",
        227,
    )
    assert found["seekpath_labels"]["G"] == "GAMMA"
    assert found["cell_changed"] is False and len(found["atoms"]) == 2
    assert sorted(found["special_points"]) == ["G", "K", "L", "U", "W", "X"]
    assert found["special_points"]["G"] == [0.0, 0.0, 0.0]
    assert found["npoints"] == len(found["kpts"]) == 60
    assert found["kpts"][0] == [0.0, 0.0, 0.0]


def test_band_path_follows_a_caller_path() -> None:
    found = band_path(_si(), path="GXL", npoints=11)
    assert found["path"] == "GXL"
    assert found["npoints"] == 11
    assert found["kpts"][-1] == pytest.approx(found["special_points"]["L"])


def test_band_path_refuses_an_unknown_label_and_lists_the_lattice_labels() -> None:
    with pytest.raises(ValueError, match=r"names 'M'.*its labels are G, K, L, U, W, W2, X"):
        band_path(_si(), path="GXM")


def test_band_path_refuses_a_cell_that_is_not_periodic() -> None:
    slab = _si()
    slab.pbc = (True, True, False)
    with pytest.raises(ValueError, match="periodic in all three directions"):
        band_path(slab)


def test_band_path_npoints_beats_density() -> None:
    assert band_path(_si(), npoints=17, density=100.0)["npoints"] == 17
    dense = band_path(_si(), density=40.0)["npoints"]
    sparse = band_path(_si(), density=10.0)["npoints"]
    assert dense > sparse > 17


def test_path_distances_put_the_comma_break_at_zero_length() -> None:
    atoms = _si()
    found = band_path(atoms, npoints=60)
    axis = path_distances(found["kpts"], found["atoms"].cell, found["special_points"])
    assert axis["labels"] == ["G", "X", "U", "K", "G", "L", "W", "X"]
    assert len(axis["x"]) == 60
    assert np.all(np.diff(axis["x"]) >= 0)
    # U,K is a break in the path: both corners sit at the same x.
    assert axis["special_x"][2] == axis["special_x"][3]
    assert axis["special_x"][1] == pytest.approx(2 * np.pi / 5.43, abs=1e-9)  # |GX| = 2pi/a


# -- read_bands and scf_levels on the real files ------------------------------------


def test_read_bands_reads_the_real_si_listing() -> None:
    read = read_bands(SI_BANDS.read_text())
    assert read["n_bands"] == 8
    assert np.shape(read["energies"]) == (60, 8)
    assert read["kpoints"][0] == pytest.approx([0.0, 0.0, 0.0])
    assert read["kpoints"][-1] == pytest.approx([0.5, 0.0, 0.5], abs=1e-6)  # X
    assert read["energies"][0][:4] == [-5.8319, 6.1597, 6.1597, 6.1597]


def test_read_bands_reads_the_real_al_listing() -> None:
    read = read_bands(AL_BANDS.read_text())
    assert np.shape(read["energies"]) == (60, 6)
    assert read["kpoints"][0] == pytest.approx([0.0, 0.0, 0.0])
    assert read["kpoints"][-1] == pytest.approx([0.5, 0.0, 0.5], abs=1e-6)


def test_read_bands_matches_the_path_the_task_wrote() -> None:
    found = band_path(_si(), npoints=60)
    read = read_bands(SI_BANDS.read_text())
    assert np.allclose(read["kpoints"], found["kpts"], atol=1e-6)


def test_scf_levels_reads_the_fermi_level_and_the_counts() -> None:
    si = scf_levels(SI_SCF.read_text())
    assert si == {
        "fermi": 6.188,
        "fermi_source": "fermi energy",
        "n_electrons": 8.0,
        "n_states": 8,
    }
    assert scf_levels(AL_SCF.read_text())["n_electrons"] == 3.0


def test_scf_levels_reads_fixed_occupations() -> None:
    text = (
        "number of electrons = 8.00\n"
        "highest occupied, lowest unoccupied level (ev):     6.2000    6.8000\n"
    )
    levels = scf_levels(text)
    assert levels["fermi"] == pytest.approx(6.5)
    assert levels["fermi_source"].startswith("midgap")
    with pytest.raises(ValueError, match="no Fermi energy"):
        scf_levels("number of electrons = 8.00\n")


def test_a_truncated_eigenvalue_listing_raises_the_verbosity_error() -> None:
    lines = SI_BANDS.read_text().splitlines(keepends=True)
    listing = next(i for i, line in enumerate(lines) if "End of band structure" in line)
    truncated = "".join(lines[: listing + 40])  # ten k-points' blocks, then nothing
    with pytest.raises(ValueError, match="verbosity='high'"):
        read_bands(truncated)


def test_the_warning_pw_prints_without_verbosity_high_raises_the_verbosity_error() -> None:
    lines = SI_BANDS.read_text().splitlines(keepends=True)
    listing = next(i for i, line in enumerate(lines) if "End of band structure" in line)
    end = next(i for i in range(listing, len(lines)) if "Writing all" in lines[i])
    warned = (
        "".join(lines[: listing + 1])
        + "\n     Number of k-points >= 100: set verbosity='high' to print the bands.\n\n"
        + "".join(lines[end:])
    )
    with pytest.raises(ValueError, match="lists no eigenvalues of the 60 k-points"):
        read_bands(warned)


# -- band_summary ----------------------------------------------------------------------


def _summary(scf: Path, bands: Path) -> dict:
    found = band_path(_si(), npoints=60)
    read = read_bands(bands.read_text())
    fermi = scf_levels(scf.read_text())["fermi"]
    return band_summary(read["energies"], fermi, read["kpoints"], found["special_points"])


def test_si_is_an_indirect_insulator_with_the_vbm_at_gamma() -> None:
    summary = _summary(SI_SCF, SI_BANDS)
    assert summary["is_metal"] is False
    assert summary["gap_kind"] == "indirect"
    assert summary["vbm_at"]["label"] == "G"
    assert summary["vbm_at"]["distance"] == pytest.approx(0.0, abs=1e-6)
    assert summary["cbm_at"]["label"] == "X"  # the minimum sits on G-X, near X
    assert summary["n_valence_bands"] == 4
    # A semilocal gap lies well below the measured 1.17 eV, and well above
    # zero; the window is loose on purpose, and the exact value is the
    # fixture's own.
    assert 0.3 < summary["gap"] < 0.8
    assert summary["gap"] == pytest.approx(summary["cbm"] - summary["vbm"], abs=1e-4)
    assert summary["direct_gap"] > summary["gap"]
    assert summary["vbm"] < summary["fermi"] < summary["cbm"]


def test_al_is_a_metal_with_no_gap() -> None:
    summary = _summary(AL_SCF, AL_BANDS)
    assert summary["is_metal"] is True
    for key in ("gap", "direct_gap", "gap_kind", "vbm", "cbm", "vbm_at", "cbm_at"):
        assert summary[key] is None
    assert "cross the Fermi level" in summary["note"]


def test_no_conduction_band_asks_for_more_bands() -> None:
    read = read_bands(SI_BANDS.read_text())
    valence_only = [row[:4] for row in read["energies"]]
    found = band_path(_si(), npoints=60)
    summary = band_summary(valence_only, 6.188, read["kpoints"], found["special_points"])
    assert summary["is_metal"] is False
    assert summary["gap"] is None and summary["cbm"] is None
    assert summary["vbm_at"]["label"] == "G"
    assert "raise nbands" in summary["note"]


def test_a_direct_gap_is_called_direct() -> None:
    k = [[0, 0, 0], [0.5, 0, 0.5]]
    summary = band_summary([[-1.0, 1.0], [-2.0, 2.0]], 0.0, k, {"G": [0, 0, 0]})
    assert summary["gap_kind"] == "direct"
    assert summary["gap"] == summary["direct_gap"] == 2.0


def test_fixed_occupations_count_bands_because_the_level_touches_the_valence_band() -> None:
    # Under fixed occupations pw.x prints the SCF mesh's highest occupied
    # level. On a mesh that holds G it equals the valence band maximum, and
    # on a shifted mesh it lies below it.
    found = band_path(_si(), npoints=60)
    read = read_bands(SI_BANDS.read_text())
    smeared = _summary(SI_SCF, SI_BANDS)
    for level in (smeared["vbm"], smeared["vbm"] - 0.05):
        crossed = band_summary(read["energies"], level, read["kpoints"], found["special_points"])
        assert crossed["is_metal"] is True  # the crossing rule cannot read this level
        counted = band_summary(
            read["energies"], level, read["kpoints"], found["special_points"], n_occupied=4
        )
        assert counted["is_metal"] is False
        assert counted["n_valence_bands"] == 4
        for key in ("gap", "direct_gap", "gap_kind", "vbm", "cbm"):
            assert counted[key] == smeared[key]


def test_counted_bands_that_overlap_are_a_metal() -> None:
    read = read_bands(AL_BANDS.read_text())
    found = band_path(bulk("Al", "fcc", a=4.05), npoints=60)
    counted = band_summary(
        read["energies"], 0.0, read["kpoints"], found["special_points"], n_occupied=2
    )
    assert counted["is_metal"] is True
    assert counted["gap"] is None
    assert "overlap" in counted["note"]


def test_a_conventional_cell_reduces_to_the_standardized_primitive_cell() -> None:
    cubic = bulk("Si", "diamond", a=5.43, cubic=True)
    found = band_path(cubic, npoints=60)
    assert len(cubic) == 8 and len(found["atoms"]) == 2
    assert found["cell_changed"] is True
    assert found["atoms"].get_volume() == pytest.approx(cubic.get_volume() / 4)
    # The path and its k-points are the primitive cell's own.
    primitive = band_path(_si(), npoints=60)
    assert found["path"] == primitive["path"]
    assert np.allclose(found["kpts"], primitive["kpts"])
    positions = found["atoms"].get_scaled_positions()
    assert np.all((positions >= 0) & (positions < 1))


def test_seekpath_labels_with_a_suffix_fit_a_path_string() -> None:
    # Body-centred tetragonal cells carry points such as SIGMA_0 and S_0.
    tetragonal = Atoms("In", cell=[3.25, 3.25, 4.95], pbc=True, scaled_positions=[[0, 0, 0]])
    tetragonal += Atoms("In", positions=[[1.625, 1.625, 2.475]])
    found = band_path(tetragonal, npoints=40)
    assert found["lattice"].startswith("tI")
    suffixed = {k: v for k, v in found["seekpath_labels"].items() if "_" in v}
    assert suffixed, found["seekpath_labels"]
    for label, name in suffixed.items():
        assert "_" not in label and label[0] == name[0]
    again = band_path(tetragonal, path=found["path"], npoints=40)
    assert np.allclose(again["kpts"], found["kpts"])


def test_matching_mesh_keeps_the_density_on_another_cell() -> None:
    from slab.bands import matching_mesh

    cubic = bulk("Si", "diamond", a=5.43, cubic=True)
    assert matching_mesh((6, 6, 6), cubic.cell, _si().cell) == (11, 11, 11)
    assert matching_mesh((8, 8, 8), _si().cell, _si().cell) == (8, 8, 8)
    # A permuted orthorhombic cell takes the permuted mesh.
    one = Atoms("Cu", cell=[3.0, 4.0, 6.0], pbc=True)
    other = Atoms("Cu", cell=[6.0, 3.0, 4.0], pbc=True)
    assert matching_mesh((8, 6, 4), one.cell, other.cell) == (4, 8, 6)
