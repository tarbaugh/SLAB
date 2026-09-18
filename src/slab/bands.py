"""Band-structure helpers: the k-point path, the eigenvalues, and the gap.

Pure functions over ASE, seekpath, and numpy. :func:`band_path` asks
seekpath for the standardized primitive cell and its high-symmetry
path, :func:`read_bands` reads the eigenvalues a
pw.x ``calculation='bands'`` run printed, :func:`band_summary` turns them
into a gap verdict against the SCF Fermi level, and :func:`path_distances`
gives the x axis a band diagram is drawn on. :func:`scf_levels` reads the
Fermi level, the electron count, and the Kohn-Sham state count from the
SCF that came before.

The verdict uses band crossings, not occupations. A band is valence when
every energy on the path lies below the SCF Fermi level and conduction
when every energy lies above it. One band that crosses the level makes
the system a metal. Smearing in the SCF moves the Fermi level inside the
gap of an insulator but never across a band, so the verdict holds with
the smearing a protocol sets for every system.

Under fixed occupations pw.x prints no Fermi energy, only the highest
occupied level of the SCF mesh. That level equals the valence band
maximum or lies below it, so it cannot separate the bands. The verdict
then counts bands: :func:`band_summary` takes ``n_occupied``, and the
lowest ``n_occupied`` bands are the valence bands.
"""

from __future__ import annotations

import io
import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "band_path",
    "band_summary",
    "matching_mesh",
    "path_distances",
    "read_bands",
    "scf_levels",
]

#: The symmetry tolerance seekpath and spglib use, in angstrom. It is
#: seekpath's own default, which also suits a relaxed structure.
SYMPREC = 1e-5

#: The gap below which an indirect gap is reported as direct, in eV. pw.x
#: prints eigenvalues to 4 decimals, so two band edges closer than 1 meV
#: are not told apart by the output itself.
DIRECT_GAP_TOLERANCE_EV = 1e-3

#: pw.x prints eigenvalues to 4 decimals in eV, so the band edges and the
#: gaps are rounded to the same 4, and a difference of two printed values
#: does not carry float noise into a report.
EIGENVALUE_DECIMALS = 4

#: How close two fractional k-points are to count as the same point. ASE
#: writes the path at 14 decimals and pw.x prints crystal coordinates at 7,
#: so 1e-4 is well above rounding and well below any path spacing.
KPOINT_MATCH_TOLERANCE = 1e-4

_PW_FERMI = re.compile(r"the Fermi energy is\s*(-?[\d.]+)\s*ev")
_PW_HOMO_LUMO = re.compile(
    r"highest occupied, lowest unoccupied level \(ev\):\s*(-?[\d.]+)\s+(-?[\d.]+)"
)
_PW_HOMO = re.compile(r"highest occupied level \(ev\):\s*(-?[\d.]+)")
_PW_ELECTRONS = re.compile(r"number of electrons\s*=\s*([\d.]+)")
_PW_STATES = re.compile(r"number of Kohn-Sham states\s*=\s*(\d+)")
_PW_KPOINTS = re.compile(r"number of k points=\s*(\d+)")


def band_path(
    atoms: Any,
    path: str | None = None,
    npoints: int | None = None,
    density: float = 20.0,
    symprec: float = SYMPREC,
) -> dict[str, Any]:
    """The high-symmetry path for *atoms*, as seekpath identifies it.

    seekpath finds the space group of *atoms* with spglib, builds the
    standardized primitive cell of Hinuma et al. (Comput. Mater. Sci. 128,
    140, 2017), and gives the recommended path with its special points in
    that cell's reciprocal basis. The path is only valid for that cell, so
    the calculation must run on the returned ``atoms``, not on the input.
    A conventional cell or a supercell of a perfect crystal reduces to the
    primitive cell.

    Labels are seekpath's, written so a path string can hold them:
    ``GAMMA`` is ``G``, and an underscore is dropped, so ``SIGMA_0`` is
    ``Sigma0`` and ``W_2`` is ``W2``. A caller *path* uses these labels,
    and a comma starts a new segment. *npoints* wins over *density* when
    both are given; *density* is the number of points per inverse angstrom
    along the path.

    Returns a dict with ``path`` (the string), ``lattice`` (seekpath's
    extended Bravais symbol, such as ``cF2``), ``spacegroup`` and
    ``spacegroup_number``, ``special_points`` (label to fractional
    coordinates), ``seekpath_labels`` (label to seekpath's own name),
    ``kpts`` (the fractional k-points), ``npoints``, ``atoms`` (the
    standardized primitive cell), ``cell_changed`` (True when its lattice
    differs from the input's), and ``bandpath`` (the
    :class:`ase.dft.kpoints.BandPath` itself, for the calculator).

    Raises:
        ValueError: the cell is not periodic in all three directions, or
            the path names a label the lattice does not have.

    Examples:
        >>> from ase.build import bulk
        >>> found = band_path(bulk("Si", "diamond", a=5.43), npoints=60)
        >>> found["path"], found["lattice"], found["spacegroup"], found["npoints"]
        ('GXU,KGLWX', 'cF2', 'Fd-3m', 60)
        >>> cubic = band_path(bulk("Si", "diamond", a=5.43, cubic=True), npoints=60)
        >>> len(cubic["atoms"]), cubic["cell_changed"]
        (2, True)
        >>> try:
        ...     band_path(bulk("Si", "diamond", a=5.43), path="GXQ")
        ... except ValueError as e:
        ...     print(e)  # doctest: +NORMALIZE_WHITESPACE
        path 'GXQ' names 'Q', which the cF2 lattice does not have;
        its labels are G, K, L, U, W, W2, X
    """
    import seekpath
    from ase import Atoms
    from ase.dft.kpoints import parse_path_string

    if not bool(np.all(atoms.pbc)):
        raise ValueError(
            "a band structure needs a cell periodic in all three directions; "
            f"this one has pbc={[bool(p) for p in atoms.pbc]}"
        )
    structure = (atoms.cell[:], atoms.get_scaled_positions(), atoms.numbers)
    found = seekpath.get_path(structure, with_time_reversal=True, symprec=symprec)
    primitive = Atoms(
        numbers=found["primitive_types"],
        cell=found["primitive_lattice"],
        scaled_positions=np.asarray(found["primitive_positions"], dtype=float) % 1.0,
        pbc=True,
    )
    names = {_path_label(name): name for name in found["point_coords"]}
    special = {
        _path_label(name): np.asarray(coords, dtype=float)
        for name, coords in found["point_coords"].items()
    }
    lattice = found["bravais_lattice_extended"]
    if path is None:
        path = _path_string(found["path"])
    else:
        segments = parse_path_string(path)  # type: ignore[no-untyped-call]
        unknown = [label for segment in segments for label in segment if label not in special]
        if unknown:
            raise ValueError(
                f"path {path!r} names {unknown[0]!r}, which the {lattice} "
                f"lattice does not have; its labels are {', '.join(sorted(special))}"
            )
    if npoints is not None:
        bandpath = primitive.cell.bandpath(path, npoints=npoints, special_points=special)
    else:
        bandpath = primitive.cell.bandpath(path, density=density, special_points=special)
    used = {label for segment in parse_path_string(bandpath.path) for label in segment}  # type: ignore[no-untyped-call]
    return {
        "path": bandpath.path,
        "lattice": lattice,
        "spacegroup": found["spacegroup_international"],
        "spacegroup_number": int(found["spacegroup_number"]),
        "special_points": {
            label: [float(c) for c in coords]
            for label, coords in special.items()
            if label in used
        },
        "seekpath_labels": {label: names[label] for label in sorted(used)},
        "kpts": [[float(c) for c in k] for k in bandpath.kpts],
        "npoints": len(bandpath.kpts),
        "atoms": primitive,
        "cell_changed": not _same_lattice(atoms.cell[:], primitive.cell[:]),
        "bandpath": bandpath,
    }


def _path_label(name: str) -> str:
    """seekpath's point name as a label an ASE path string can hold."""
    if name == "GAMMA":
        return "G"
    head, _, tail = name.partition("_")
    return (head if len(head) == 1 else head.capitalize()) + tail


def _path_string(segments: Sequence[tuple[str, str]]) -> str:
    """seekpath's list of segments as one path string, a comma at each break."""
    out = ""
    previous = None
    for start, stop in segments:
        if start != previous:
            out += ("," if out else "") + _path_label(start)
        out += _path_label(stop)
        previous = stop
    return out


def _same_lattice(one: np.ndarray, other: np.ndarray) -> bool:
    return bool(np.allclose(one, other, atol=1e-6))


def matching_mesh(
    kpts: Sequence[int], cell: Any, new_cell: Any
) -> tuple[int, int, int]:
    """A Monkhorst-Pack mesh on *new_cell* at least as dense as *kpts* on *cell*.

    The finest spacing of the old mesh along any reciprocal vector sets
    the spacing of the new one. A mesh made for one cell is wrong for
    another setting of the same crystal, where the axes are permuted or
    the cell is smaller.

    Examples:
        >>> from ase.build import bulk
        >>> cubic = bulk("Si", "diamond", a=5.43, cubic=True)
        >>> matching_mesh((6, 6, 6), cubic.cell, bulk("Si", "diamond", a=5.43).cell)
        (11, 11, 11)
    """
    from ase.cell import Cell

    old = np.linalg.norm(Cell.new(cell).reciprocal()[:], axis=1)  # type: ignore[no-untyped-call]
    new = np.linalg.norm(Cell.new(new_cell).reciprocal()[:], axis=1)  # type: ignore[no-untyped-call]
    spacing = float(np.min(old / np.asarray(kpts, dtype=float)))
    mesh = [max(1, int(np.ceil(length / spacing - 1e-6))) for length in new]
    return (mesh[0], mesh[1], mesh[2])


def scf_levels(pwo_text: str) -> dict[str, Any]:
    """The Fermi level, electron count, and state count an SCF output printed.

    ``fermi`` is the Fermi energy under smearing. Under fixed occupations
    pw.x prints the highest occupied level instead, and ``fermi`` is then
    midway between it and the lowest unoccupied level when pw.x printed
    both. ``fermi_source`` says which line gave the number.

    Raises:
        ValueError: the output holds none of those lines, or no electron
            count.

    Examples:
        >>> text = "number of electrons = 8.00\\nnumber of Kohn-Sham states= 8\\n"
        >>> levels = scf_levels(text + "the Fermi energy is  6.1234 ev\\n")
        >>> levels["fermi"], levels["n_electrons"], levels["n_states"]
        (6.1234, 8.0, 8)
    """
    electrons = _PW_ELECTRONS.search(pwo_text)
    states = _PW_STATES.search(pwo_text)
    if electrons is None:
        raise ValueError("the SCF output holds no 'number of electrons' line")
    fermi: float | None = None
    source = None
    if found := _PW_FERMI.findall(pwo_text):
        fermi, source = float(found[-1]), "fermi energy"
    elif found := _PW_HOMO_LUMO.findall(pwo_text):
        homo, lumo = (float(v) for v in found[-1])
        fermi, source = (homo + lumo) / 2, "midgap of highest occupied and lowest unoccupied"
    elif found := _PW_HOMO.findall(pwo_text):
        fermi, source = float(found[-1]), "highest occupied level"
    if fermi is None:
        raise ValueError(
            "the SCF output prints no Fermi energy and no highest occupied "
            "level; did the SCF finish?"
        )
    return {
        "fermi": fermi,
        "fermi_source": source,
        "n_electrons": float(electrons.group(1)),
        "n_states": int(states.group(1)) if states else None,
    }


def read_bands(pwo_text: str) -> dict[str, Any]:
    """The eigenvalues a pw.x bands run printed, per k-point, in eV.

    Wraps ASE's pw.x output parser. Returns ``kpoints`` (fractional, one
    row per k-point), ``energies`` (n_k by n_bands, eV), and ``n_bands``.

    Raises:
        ValueError: the output holds fewer eigenvalue listings than
            k-points, which is what pw.x prints for more than 100 k-points
            without ``verbosity='high'``; or the run is spin-polarized.
    """
    from ase.io.espresso import read_espresso_out

    declared = _PW_KPOINTS.findall(pwo_text)
    asked = int(declared[-1]) if declared else None
    try:
        images = list(read_espresso_out(io.StringIO(pwo_text), index=slice(None)))
    except (AssertionError, IndexError) as e:
        # ASE's parser asserts one listing per k-point, and runs off the end
        # of a file cut inside the listing.
        raise ValueError(_verbosity_message(asked, None)) from e
    kpts = getattr(images[-1].calc, "kpts", None) if images else None
    if not kpts:
        raise ValueError(_verbosity_message(asked, 0))
    if any(k.s != 0 for k in kpts):
        raise ValueError(
            "the bands output is spin-polarized; read_bands reads one spin "
            "channel, and band_structure does not run nspin=2 yet"
        )
    kpoints = np.array(images[-1].calc.get_ibz_k_points(), dtype=float)
    energies = np.array([k.eps_n for k in kpts], dtype=float)
    if asked is not None and len(energies) < asked:
        raise ValueError(_verbosity_message(asked, len(energies)))
    return {
        "kpoints": kpoints.tolist(),
        "energies": energies.tolist(),
        "n_bands": int(energies.shape[1]),
    }


def _verbosity_message(asked: int | None, found: int | None) -> str:
    listed = "no" if not found else f"only {found}"
    of = f" of the {asked} k-points it declares" if asked else ""
    return (
        f"the pw.x output lists {listed} eigenvalues{of}; pw.x prints the "
        f"bands of more than 100 k-points only with verbosity='high', and "
        f"a cut-short run lists fewer"
    )


def band_summary(
    energies: Sequence[Sequence[float]] | np.ndarray,
    fermi: float,
    kpoints: Sequence[Sequence[float]] | np.ndarray,
    labels: Mapping[str, Sequence[float]] | None = None,
    n_occupied: int | None = None,
) -> dict[str, Any]:
    """The gap verdict of a band structure against the SCF Fermi level.

    *energies* is n_k by n_bands in eV, *fermi* is the SCF's Fermi level in
    eV, *kpoints* are the fractional k-points of the rows, and *labels*
    maps each special point to its fractional coordinates. On a mesh
    there are no special points to name, so *labels* may be None or
    empty, and ``vbm_at`` and ``cbm_at`` then carry the k-point alone.

    A band is valence when every energy lies below *fermi* and conduction
    when every energy lies above it. A band that crosses *fermi* makes the
    system a metal, and the gap fields are then None. With no conduction
    band in the listing the gap fields are also None and ``note`` says to
    raise ``nbands``. ``vbm_at`` and ``cbm_at`` give the fractional k of
    the band edge, the nearest special point, and its distance in
    fractional units. ``gap_kind`` is ``"direct"`` when the smallest gap
    at one k-point is within :data:`DIRECT_GAP_TOLERANCE_EV` of the gap.

    Pass *n_occupied* when the SCF ran with fixed occupations. *fermi* is
    then the highest occupied level of the SCF mesh, which touches the
    valence band or cuts it, so the lowest *n_occupied* bands are the
    valence bands and the rest are conduction bands. The system is a
    metal when the two sets overlap in energy.

    Examples:
        >>> k = [[0, 0, 0], [0.5, 0, 0.5]]
        >>> labels = {"G": [0, 0, 0], "X": [0.5, 0, 0.5]}
        >>> s = band_summary([[-1.0, 2.0], [-2.0, 1.0]], 0.0, k, labels)
        >>> s["is_metal"], s["gap"], s["gap_kind"], s["vbm_at"]["label"], s["cbm_at"]["label"]
        (False, 2.0, 'indirect', 'G', 'X')
        >>> band_summary([[-1.0, 0.5], [-2.0, -0.5]], 0.0, k, {"G": [0, 0, 0]})["is_metal"]
        True
        >>> s = band_summary([[-1.0, 2.0], [-2.0, 1.0]], -1.0, k, labels, n_occupied=1)
        >>> s["is_metal"], s["gap"]
        (False, 2.0)
    """
    bands = np.asarray(energies, dtype=float)
    points = np.asarray(kpoints, dtype=float)
    overlap = False
    if n_occupied is None:
        valence = [n for n in range(bands.shape[1]) if bool(np.all(bands[:, n] < fermi))]
        conduction = [n for n in range(bands.shape[1]) if bool(np.all(bands[:, n] > fermi))]
        crossing = bands.shape[1] - len(valence) - len(conduction)
    else:
        valence = list(range(min(n_occupied, bands.shape[1])))
        conduction = list(range(len(valence), bands.shape[1]))
        crossing = 0
        overlap = bool(
            valence and conduction
            and np.max(bands[:, valence[-1]]) >= np.min(bands[:, conduction[0]])
        )
    summary: dict[str, Any] = {
        "is_metal": crossing > 0 or overlap,
        "fermi": float(fermi),
        "vbm": None,
        "cbm": None,
        "gap": None,
        "direct_gap": None,
        "gap_kind": None,
        "vbm_at": None,
        "cbm_at": None,
        "n_valence_bands": len(valence),
    }
    if crossing:
        summary["note"] = f"{crossing} band(s) cross the Fermi level, so the system is a metal"
        return summary
    if overlap:
        summary["note"] = (
            f"band {len(valence)} and band {len(valence) + 1} overlap in energy, "
            f"so the system is a metal"
        )
        return summary
    if not valence:
        summary["note"] = "no band lies wholly below the Fermi level"
        return summary
    top = bands[:, valence[-1]]
    ivbm = int(np.argmax(top))
    summary["vbm"] = float(top[ivbm])
    summary["vbm_at"] = _nearest_label(points[ivbm], labels)
    if not conduction:
        summary["note"] = (
            "the listing holds no conduction band, so there is no gap to "
            "report; raise nbands"
        )
        return summary
    bottom = bands[:, conduction[0]]
    icbm = int(np.argmin(bottom))
    gap = round(float(bottom[icbm] - top[ivbm]), EIGENVALUE_DECIMALS)
    direct = round(float(np.min(bottom - top)), EIGENVALUE_DECIMALS)
    summary.update(
        cbm=float(bottom[icbm]),
        gap=gap,
        direct_gap=direct,
        gap_kind="direct" if direct - gap <= DIRECT_GAP_TOLERANCE_EV else "indirect",
        cbm_at=_nearest_label(points[icbm], labels),
    )
    return summary


def _nearest_label(
    k: np.ndarray, labels: Mapping[str, Sequence[float]] | None
) -> dict[str, Any]:
    """The special point nearest *k*, with the distance in fractional units.

    With no labels the answer is the k-point alone: a mesh run names no
    special point, and a made-up nearest label would read as one.
    """
    best, distance = None, float("inf")
    for label, coords in (labels or {}).items():
        d = float(np.linalg.norm(k - np.asarray(coords, dtype=float)))
        if d < distance:
            best, distance = label, d
    return {
        "label": best,
        "distance": None if best is None else distance,
        "k": [float(c) for c in k],
    }


def path_distances(
    kpoints: Sequence[Sequence[float]] | np.ndarray,
    cell: Any,
    special_points: Mapping[str, Sequence[float]] | None = None,
) -> dict[str, Any]:
    """The x axis of a band diagram along *kpoints*, in inverse angstrom.

    Returns ``x`` (the cumulative distance of every k-point), ``special_x``
    (the x of each corner of the path), and ``labels`` (the special point
    at each corner, ``"?"`` where none matches). A comma break in the path
    adds no distance, as in ASE's own band plots.

    Examples:
        >>> from ase.build import bulk
        >>> atoms = bulk("Si", "diamond", a=5.43)
        >>> found = band_path(atoms, path="GX", npoints=5)
        >>> axis = path_distances(found["kpts"], atoms.cell, found["special_points"])
        >>> axis["labels"], round(axis["special_x"][-1], 4)
        (['G', 'X'], 1.1571)
    """
    from ase.cell import Cell
    from ase.dft.kpoints import labels_from_kpts

    points = np.asarray(kpoints, dtype=float)
    specials = (
        None
        if special_points is None
        else {k: np.asarray(v, dtype=float) for k, v in special_points.items()}
    )
    x, special_x, labels = labels_from_kpts(  # type: ignore[no-untyped-call]
        points,
        Cell.new(cell),  # type: ignore[no-untyped-call]
        eps=KPOINT_MATCH_TOLERANCE,
        special_points=specials,
    )
    return {
        "x": [float(v) for v in x],
        "special_x": [float(v) for v in special_x],
        "labels": list(labels),
    }
