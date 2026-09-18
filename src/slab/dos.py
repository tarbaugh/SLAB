"""Density-of-states helpers: the dos.x table, the projections, and the verdict.

Pure functions over the files Quantum ESPRESSO's post-processing tools
write. :func:`read_dos` reads the table ``dos.x`` writes, :func:`read_pdos`
sums ``projwfc.x``'s per-state files into one curve per element and
angular momentum, :func:`read_projection_states` reads the state list
``projwfc.x`` prints, :func:`read_projections` reads the projections it
writes to ``atomic_proj.xml``, :func:`group_projections` sums them into
the same groups, and :func:`dos_summary` turns a table and the
eigenvalues into a verdict.

A group is named ``"<element>-<l>"``, for example ``"Si-s"`` or
``"Si-p"``. A pseudopotential that carries two wavefunctions of the same
l (a semicore shell and a valence shell) sums both into one group,
because the group is the physical channel and not the pseudopotential's
bookkeeping.

The projections are onto pseudo-atomic orbitals. That basis is not
complete and not orthogonal to the other atoms' orbitals, so the weights
of one state sum to near one and not to one. Report them as weights, not
as charges.

One spin channel only. A spin-polarized, noncollinear, or spin-orbit file
is refused, because each of them writes two channels or spinor states
that these readers do not separate.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "dos_summary",
    "group_projections",
    "read_dos",
    "read_pdos",
    "read_projection_states",
    "read_projections",
]

#: Rydberg to electronvolt. ``projwfc.x`` writes the eigenvalues and the
#: Fermi level of ``atomic_proj.xml`` in Rydberg, while every other number
#: these readers return is in eV, as ``dos.x`` and ``pw.x`` print it.
RY_TO_EV = 13.605693122994

#: How far above one the weights of a state may sum before
#: :func:`group_projections` calls the file inconsistent. The projections
#: onto pseudo-atomic orbitals of neighbouring atoms overlap, so a sum
#: slightly above one is physical; a sum far above one is a misread file.
PROJECTION_SUM_LIMIT = 1.5

#: The angular momentum names, by l. ``projwfc.x`` prints l as a number,
#: and a group name carries the letter a reader expects.
ORBITAL_LETTERS = ("s", "p", "d", "f", "g")

#: The energies of two grid rows are the same grid step apart to this many
#: electronvolts. ``dos.x`` prints the grid to 3 decimals, so a spacing
#: read back from the table carries no more precision than that.
GRID_DECIMALS = 3

_DOS_FERMI = re.compile(r"EFermi\s*=\s*(-?[\d.]+)\s*eV")
_PDOS_NAME = re.compile(r"pdos_atm#(\d+)\(([A-Za-z]+)\s*\)_wfc#(\d+)\(([a-z])\)")
_PROJWFC_STATE = re.compile(
    r"state #\s*(\d+):\s*atom\s+(\d+)\s*\(\s*([A-Za-z]+)\s*\),\s*wfc\s+(\d+)\s*\(([^)]*)\)"
)
_STATE_L = re.compile(r"l=\s*(\d+)")
_STATE_M = re.compile(r"\bm=\s*(-?\d+)")
_STATE_J = re.compile(r"j=\s*([\d.]+)")


def read_dos(text: str) -> dict[str, Any]:
    """The table ``dos.x`` wrote, in eV.

    The header line names the columns and carries the Fermi level. The
    three columns are the energy, the density of states, and the
    integrated density of states.

    Returns ``energies``, ``dos``, ``integrated_dos``, ``fermi`` (None
    when the header carries none), ``energy_unit`` (``"eV"``),
    ``dos_unit`` (``"states/eV/cell"``), and ``delta_e`` (the grid step).

    Raises:
        ValueError: the file is spin-polarized, holds no header, or holds
            fewer than two rows.

    Examples:
        >>> table = read_dos(
        ...     "#  E (eV)   dos(E)     Int dos(E) EFermi =    6.193 eV\\n"
        ...     "  -6.646  0.2908E-07  0.1454E-08\\n"
        ...     "  -6.596  0.1132E-06  0.7116E-08\\n"
        ... )
        >>> table["fermi"], table["dos"][1], table["delta_e"]
        (6.193, 1.132e-07, 0.05)
    """
    header, rows = _table(text, "the dos.x table")
    if "dosup" in header.replace(" ", "") or "dosdw" in header.replace(" ", ""):
        raise ValueError(
            "the dos.x table is spin-polarized (it holds dosup and dosdw "
            "columns); these readers read one channel, and the tasks do "
            "not run nspin=2 yet"
        )
    if rows.shape[1] < 3:
        raise ValueError(
            f"the dos.x table holds {rows.shape[1]} column(s); it needs the "
            f"energy, the dos, and the integrated dos"
        )
    fermi = _DOS_FERMI.search(header)
    return {
        "energies": rows[:, 0].tolist(),
        "dos": rows[:, 1].tolist(),
        "integrated_dos": rows[:, 2].tolist(),
        "fermi": float(fermi.group(1)) if fermi else None,
        "energy_unit": "eV",
        "dos_unit": "states/eV/cell",
        "delta_e": _grid_step(rows[:, 0]),
    }


def read_pdos(files: Mapping[str, str]) -> dict[str, Any]:
    """The per-state files ``projwfc.x`` wrote, summed per element and l.

    *files* maps each file name to its text. A name of the shape
    ``<prefix>.pdos_atm#1(Si)_wfc#2(p)`` names the atom, its element, and
    the wavefunction's angular momentum, and the file's second column is
    that wavefunction's l-summed density of states. Every file of one
    element and one l adds into one group, over the atoms of the element
    and over the pseudopotential's shells of that l. A
    ``<prefix>.pdos_tot`` file gives the total.

    Returns ``energies``, ``groups`` (the group name to its curve),
    ``group_names`` (sorted), ``total`` (the total density of states from
    the ``pdos_tot`` file, None without it), ``total_projected`` (the sum
    of every projection, from the same file), ``energy_unit``, and
    ``dos_unit``.

    Raises:
        ValueError: a file is spin-polarized, a name is not a pdos name,
            the grids differ, or *files* holds no per-state file.

    Examples:
        >>> rows = "  -6.646  0.141E-07  0.141E-07\\n  -6.596  0.548E-07  0.548E-07\\n"
        >>> read = read_pdos({"si.pdos_atm#1(Si)_wfc#1(s)": "# E (eV) ldos(E)\\n" + rows})
        >>> read["group_names"], read["groups"]["Si-s"][0]
        (['Si-s'], 1.41e-08)
    """
    energies: list[float] | None = None
    groups: dict[str, np.ndarray] = {}
    total = None
    total_projected = None
    for name, text in sorted(files.items()):
        header, rows = _table(text, f"the pdos file {name!r}")
        _refuse_spin_pdos(name, header)
        if energies is None:
            energies = rows[:, 0].tolist()
        elif len(energies) != rows.shape[0]:
            raise ValueError(
                f"the pdos file {name!r} holds {rows.shape[0]} rows, but an "
                f"earlier file holds {len(energies)}; the files come from "
                f"one projwfc.x run or from none"
            )
        if name.endswith(".pdos_tot"):
            total = rows[:, 1].tolist()
            total_projected = rows[:, 2].tolist() if rows.shape[1] > 2 else None
            continue
        found = _PDOS_NAME.search(name)
        if found is None:
            raise ValueError(
                f"{name!r} is not a projwfc.x pdos file name; a per-state "
                f"file is named like 'si.pdos_atm#1(Si)_wfc#2(p)' and the "
                f"total like 'si.pdos_tot'"
            )
        group = f"{found.group(2)}-{found.group(4)}"
        # Column 2 is the l-summed ldos of that wavefunction. The columns
        # after it are its m components, and summing them again would
        # count the same states twice.
        groups[group] = groups.get(group, 0.0) + rows[:, 1]
    if energies is None or not groups:
        raise ValueError(
            "read_pdos was given no per-state file; pass every "
            "'<prefix>.pdos_atm#...' file projwfc.x wrote"
        )
    return {
        "energies": energies,
        "groups": {name: groups[name].tolist() for name in sorted(groups)},
        "group_names": sorted(groups),
        "total": total,
        "total_projected": total_projected,
        "energy_unit": "eV",
        "dos_unit": "states/eV/cell",
    }


def read_projection_states(text: str) -> list[dict[str, Any]]:
    """The state list ``projwfc.x`` printed on its standard output.

    ``projwfc.x`` numbers its pseudo-atomic orbitals and prints one line
    per state, ``state #   2: atom   1 (Si ), wfc  2 (l=1 m= 2)``. The
    list says which atom and which angular momentum every column of
    ``atomic_proj.xml`` belongs to, and ``atomic_proj.xml`` itself says
    nothing about them.

    Returns one dict per state, in the file's order, with ``state`` (the
    1-based index), ``atom`` (the 1-based atom index), ``element``,
    ``wfc`` (the pseudopotential's wavefunction index), ``l``, ``m``
    (None when the line prints none), and ``group``.

    Raises:
        ValueError: the output prints no state line, or prints total
            angular momenta, which a noncollinear or spin-orbit run does
            and these readers do not read.

    Examples:
        >>> states = read_projection_states("     state #   2: atom   1 (Si ), wfc  2 (l=1 m= 2)")
        >>> states[0]["group"], states[0]["l"], states[0]["m"]
        ('Si-p', 1, 2)
    """
    states: list[dict[str, Any]] = []
    for found in _PROJWFC_STATE.finditer(text):
        detail = found.group(5)
        if _STATE_J.search(detail):
            raise ValueError(
                "the projwfc.x output lists states by total angular "
                "momentum j, so the run is noncollinear or spin-orbit; "
                "these readers read one scalar channel"
            )
        angular = _STATE_L.search(detail)
        if angular is None:
            raise ValueError(f"the projwfc.x state line {found.group(0)!r} names no l")
        l_value = int(angular.group(1))
        magnetic = _STATE_M.search(detail)
        element = found.group(3)
        states.append(
            {
                "state": int(found.group(1)),
                "atom": int(found.group(2)),
                "element": element,
                "wfc": int(found.group(4)),
                "l": l_value,
                "m": int(magnetic.group(1)) if magnetic else None,
                "group": f"{element}-{_letter(l_value)}",
            }
        )
    if not states:
        raise ValueError(
            "the projwfc.x output prints no 'state #' line; did projwfc.x "
            "finish, and is this its standard output?"
        )
    return states


def read_projections(xml_text: str) -> dict[str, Any]:
    """The projections ``projwfc.x`` wrote to ``atomic_proj.xml``.

    Every k-point carries the eigenvalues and one complex projection per
    band and per pseudo-atomic orbital. The weight of an orbital in a band
    is the square of the modulus of its projection.

    Returns ``n_kpoints``, ``n_bands``, ``n_states``, ``kpoints`` (in
    units of 2π/alat, as projwfc.x writes them), ``weights``
    (``weights[k][band][state]``), ``energies`` (n_k by n_bands, in eV),
    ``fermi`` (in eV), and ``n_electrons``.

    Raises:
        ValueError: the file declares more than one spin component, or it
            is not a projections file, or it is cut short.

    Examples:
        >>> from textwrap import dedent
        >>> xml = dedent('''\\
        ...     <PROJECTIONS>
        ...       <HEADER NUMBER_OF_BANDS="1" NUMBER_OF_K-POINTS="1"
        ...        NUMBER_OF_SPIN_COMPONENTS="1" NUMBER_OF_ATOMIC_WFC="1"
        ...        NUMBER_OF_ELECTRONS="2.0" FERMI_ENERGY="0.5"/>
        ...       <EIGENSTATES>
        ...         <K-POINT Weight="1.0">0.0 0.0 0.0</K-POINT>
        ...         <E>-0.4</E>
        ...         <PROJS><ATOMIC_WFC index="1" spin="1">0.6 0.8</ATOMIC_WFC></PROJS>
        ...       </EIGENSTATES>
        ...     </PROJECTIONS>''')
        >>> read = read_projections(xml)
        >>> round(read["weights"][0][0][0], 6), round(read["fermi"], 4)
        (1.0, 6.8028)
    """
    from xml.etree import ElementTree

    try:
        root = ElementTree.fromstring(xml_text)
    except ElementTree.ParseError as e:
        raise ValueError(
            f"atomic_proj.xml does not parse as XML ({e}); the file is cut "
            f"short, or projwfc.x died before it closed it"
        ) from e
    header = root.find("HEADER")
    if root.tag != "PROJECTIONS" or header is None:
        raise ValueError(
            f"the projections file has root element {root.tag!r} and no "
            f"HEADER; expected projwfc.x's atomic_proj.xml"
        )
    spins = int(header.get("NUMBER_OF_SPIN_COMPONENTS", "1"))
    if spins != 1:
        raise ValueError(
            f"atomic_proj.xml declares {spins} spin components; these "
            f"readers read one channel, and the tasks do not run nspin=2 yet"
        )
    n_bands = int(header.get("NUMBER_OF_BANDS", "0"))
    n_kpoints = int(header.get("NUMBER_OF_K-POINTS", "0"))
    n_states = int(header.get("NUMBER_OF_ATOMIC_WFC", "0"))
    states = root.find("EIGENSTATES")
    if states is None:
        raise ValueError("atomic_proj.xml holds no EIGENSTATES element")
    kpoints: list[list[float]] = []
    energies: list[list[float]] = []
    weights: list[list[list[float]]] = []
    pending: list[list[float]] = []
    for child in states:
        if child.tag == "K-POINT":
            kpoints.append([float(v) for v in _numbers(child.text)])
        elif child.tag == "E":
            energies.append([v * RY_TO_EV for v in _numbers(child.text)])
        elif child.tag == "PROJS":
            pending = []
            for orbital in child.findall("ATOMIC_WFC"):
                parts = _numbers(orbital.text)
                pending.append([parts[i] ** 2 + parts[i + 1] ** 2 for i in range(0, len(parts), 2)])
            # pending is state-major; the return value is band-major, so a
            # reader indexes it the way a band diagram is drawn.
            weights.append([[column[band] for column in pending] for band in range(n_bands)])
    if len(kpoints) != n_kpoints or len(weights) != n_kpoints:
        raise ValueError(
            f"atomic_proj.xml declares {n_kpoints} k-points but holds "
            f"{len(weights)} projection block(s); the file is cut short"
        )
    fermi = header.get("FERMI_ENERGY")
    return {
        "n_kpoints": n_kpoints,
        "n_bands": n_bands,
        "n_states": n_states,
        "kpoints": kpoints,
        "energies": energies,
        "weights": weights,
        "fermi": None if fermi is None else float(fermi) * RY_TO_EV,
        "n_electrons": float(header.get("NUMBER_OF_ELECTRONS", "nan")),
        "energy_unit": "eV",
    }


def group_projections(
    weights: Sequence[Sequence[Sequence[float]]], states: Sequence[Mapping[str, Any]]
) -> dict[str, list[list[float]]]:
    """The per-state weights summed into one curve per element and l.

    *weights* is ``weights[k][band][state]`` from :func:`read_projections`
    and *states* is the list from :func:`read_projection_states`. Returns
    ``{group: weight[k][band]}``, one key per group, sorted by name.

    The weights of one band at one k-point sum to near one and not to
    one, because the pseudo-atomic orbitals are neither complete nor
    orthogonal between atoms.

    Raises:
        ValueError: the state list does not match the width of *weights*,
            or a state's weights sum above
            :data:`PROJECTION_SUM_LIMIT`, which no correct file does.

    Examples:
        >>> states = [{"group": "Si-s"}, {"group": "Si-p"}, {"group": "Si-p"}]
        >>> group_projections([[[0.5, 0.2, 0.2]]], states)
        {'Si-p': [[0.4]], 'Si-s': [[0.5]]}
    """
    block = np.asarray(weights, dtype=float)
    if block.ndim != 3 or block.shape[2] != len(states):
        shape = "x".join(str(n) for n in block.shape)
        raise ValueError(
            f"the projections are {shape} but the state list holds "
            f"{len(states)} state(s); both come from one projwfc.x run"
        )
    worst = float(block.sum(axis=2).max()) if block.size else 0.0
    if worst > PROJECTION_SUM_LIMIT:
        raise ValueError(
            f"a band's projections sum to {worst:.3f}, above the "
            f"{PROJECTION_SUM_LIMIT} a projection onto pseudo-atomic "
            f"orbitals reaches; the file or the state list is misread"
        )
    names = sorted({str(state["group"]) for state in states})
    grouped: dict[str, list[list[float]]] = {}
    for name in names:
        columns = [i for i, state in enumerate(states) if state["group"] == name]
        grouped[name] = block[:, :, columns].sum(axis=2).tolist()
    return grouped


def dos_summary(
    energies: Sequence[float],
    dos: Sequence[float],
    fermi: float,
    eigenvalues: Sequence[Sequence[float]] | None = None,
    kpoints: Sequence[Sequence[float]] | None = None,
    n_occupied: int | None = None,
) -> dict[str, Any]:
    """The verdict of a density of states against its eigenvalues.

    ``dos_at_fermi`` is the table's density of states at *fermi*, linearly
    interpolated between the two grid rows around it. It is a broadened
    number: a Gaussian of width ``degauss`` smears states across the gap
    of a small-gap insulator, so a nonzero value alone does not make a
    metal.

    The verdict itself comes from the eigenvalues, through
    :func:`slab.bands.band_summary`. Pass *eigenvalues* (n_k by n_bands,
    in eV) and the *kpoints* they sit on. Pass *n_occupied* when the run
    used fixed occupations, exactly as :func:`slab.bands.band_summary`
    takes it. Without *eigenvalues* the gap fields stay None and ``note``
    says why.

    Returns ``fermi``, ``dos_at_fermi``, ``dos_unit``, ``energy_range``,
    ``delta_e``, ``n_points``, and the band fields ``is_metal``, ``vbm``,
    ``cbm``, ``gap``, ``direct_gap``, ``gap_kind``, ``vbm_at``,
    ``cbm_at``, and ``n_valence_bands``.

    Examples:
        >>> grid = [-1.0, 0.0, 1.0, 2.0]
        >>> summary = dos_summary(grid, [1.0, 0.0, 0.0, 2.0], 0.5,
        ...     eigenvalues=[[-1.0, 2.0], [-2.0, 1.0]], kpoints=[[0, 0, 0], [0.5, 0, 0.5]])
        >>> summary["dos_at_fermi"], summary["is_metal"], summary["gap"]
        (0.0, False, 2.0)
    """
    from slab.bands import band_summary

    grid = np.asarray(energies, dtype=float)
    curve = np.asarray(dos, dtype=float)
    if grid.size < 2 or grid.shape != curve.shape:
        raise ValueError(
            f"dos_summary needs at least two grid rows and one dos value "
            f"per row; it was given {grid.size} and {curve.size}"
        )
    summary: dict[str, Any] = {
        "fermi": float(fermi),
        "dos_at_fermi": float(np.interp(fermi, grid, curve)),
        "dos_unit": "states/eV/cell",
        "energy_range": [float(grid.min()), float(grid.max())],
        "delta_e": _grid_step(grid),
        "n_points": int(grid.size),
    }
    if eigenvalues is None:
        summary.update(
            is_metal=None,
            vbm=None,
            cbm=None,
            gap=None,
            direct_gap=None,
            gap_kind=None,
            vbm_at=None,
            cbm_at=None,
            n_valence_bands=None,
            note="no eigenvalues were given, so there is no gap verdict",
        )
        return summary
    bands = band_summary(
        eigenvalues,
        fermi,
        kpoints if kpoints is not None else [[0.0, 0.0, 0.0]] * len(eigenvalues),
        None,
        n_occupied=n_occupied,
    )
    bands.pop("fermi", None)
    summary.update(bands)
    return summary


def _letter(l_value: int) -> str:
    """The name of angular momentum *l*, or ``l=<n>`` beyond the named ones."""
    if 0 <= l_value < len(ORBITAL_LETTERS):
        return ORBITAL_LETTERS[l_value]
    return f"l={l_value}"


def _numbers(text: str | None) -> list[float]:
    """Every float in *text*, in order."""
    return [float(token) for token in (text or "").split()]


def _grid_step(grid: Any) -> float | None:
    """The spacing of an even grid, rounded as the table prints it, else None."""
    values = np.asarray(grid, dtype=float)
    if values.size < 2:
        return None
    return round(float(values[1] - values[0]), GRID_DECIMALS)


def _table(text: str, what: str) -> tuple[str, np.ndarray]:
    """``(header line, rows)`` of a QE post-processing table."""
    header = ""
    rows: list[list[float]] = []
    width = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if not header:
                header = stripped
            continue
        try:
            values = [float(token) for token in stripped.split()]
        except ValueError as e:
            raise ValueError(f"{what} holds the unreadable row {stripped!r}") from e
        width = width or len(values)
        if len(values) != width:
            raise ValueError(
                f"{what} holds a row of {len(values)} column(s) after rows "
                f"of {width}; the file is cut short or interleaved"
            )
        rows.append(values)
    if not header:
        raise ValueError(f"{what} holds no '#' header line naming its columns")
    if len(rows) < 2:
        raise ValueError(f"{what} holds {len(rows)} row(s) of numbers; it needs at least two")
    return header, np.asarray(rows, dtype=float)


def _refuse_spin_pdos(name: str, header: str) -> None:
    """Refuse a spin-polarized pdos file, which prints two channels per column."""
    squeezed = header.replace(" ", "")
    if "ldosup" in squeezed or "dosup" in squeezed:
        raise ValueError(
            f"the pdos file {name!r} is spin-polarized (it holds up and "
            f"down columns); these readers read one channel, and the tasks "
            f"do not run nspin=2 yet"
        )
