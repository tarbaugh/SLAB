"""Sanity-check a built structure file: composition, cell, and close contacts.

Reads any structure file ASE understands and reports the facts a builder
can get wrong: atom count, formula, cell lengths and angles, density, and
the minimum interatomic distance under periodic boundary conditions. The
minimum distance is also compared with the shortest bond the closest
pair's covalent radii predict. A minimum far below that bond means
overlapping atoms, the classic failure after a merge or a defect
insertion, and the fix is to rebuild, not to ask an optimizer to untangle
it. By default the check fails when the minimum is below 0.6 of the
expected bond; ``--expect-atoms`` also fails on a wrong atom count.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np


def analyze(path: str, threshold: float) -> dict[str, Any]:
    from ase.data import atomic_numbers, covalent_radii
    from ase.io import read

    atoms = read(path)
    if isinstance(atoms, list):  # multi-frame file: judge the last frame
        atoms = atoms[-1]
    report: dict[str, Any] = {
        "file": path,
        "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
        "pbc": [bool(flag) for flag in atoms.pbc],
        "threshold_A": threshold,
    }
    if atoms.cell.rank == 3:
        lengths = atoms.cell.lengths()
        angles = atoms.cell.angles()
        volume = float(atoms.get_volume())
        report["cell_lengths_A"] = [round(float(x), 6) for x in lengths]
        report["cell_angles_deg"] = [round(float(x), 4) for x in angles]
        report["volume_A3"] = round(volume, 6)
        # amu/A^3 -> g/cm^3
        report["density_g_cm3"] = round(float(atoms.get_masses().sum()) / volume * 1.66054, 6)
    minimum, close, pair = _closest(atoms, threshold)
    report["min_distance_A"] = round(minimum, 6) if minimum is not None else None
    report["close_pairs"] = close
    report["closest_pair"] = list(pair) if pair is not None else None
    if pair is not None and minimum is not None:
        expected = sum(covalent_radii[atomic_numbers[symbol]] for symbol in pair)
        report["shortest_expected_bond_A"] = round(float(expected), 6)
        report["min_distance_fraction"] = round(minimum / float(expected), 6)
    else:
        report["shortest_expected_bond_A"] = None
        report["min_distance_fraction"] = None
    return report


def _closest(
    atoms: Any, threshold: float
) -> tuple[float | None, int, tuple[str, str] | None]:
    """(minimum pair distance, pairs below threshold, closest pair's species).

    Everything is PBC-aware. Small systems get the exact full distance
    matrix. Large ones use a neighbor list with a finite cutoff; a minimum
    reported as None then means "no pair closer than the cutoff", which
    is what the check needs.
    """
    n = len(atoms)
    if n < 2:
        return None, 0, None
    symbols = atoms.get_chemical_symbols()
    if n <= 1500:
        from ase.geometry import get_distances

        _, distances = get_distances(atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
        off_diagonal = ~np.eye(n, dtype=bool)
        masked = np.where(off_diagonal, distances, np.inf)
        i, j = np.unravel_index(int(np.argmin(masked)), masked.shape)
        minimum = float(masked[i, j])
        close = int((distances[np.triu(off_diagonal)] < threshold).sum())
        return minimum, close, (symbols[int(i)], symbols[int(j)])
    from ase.neighborlist import neighbor_list

    cutoff = max(2.0 * threshold, 3.0)
    first, second, distances = neighbor_list("ijd", atoms, cutoff)
    if len(distances) == 0:
        return None, 0, None
    k = int(np.argmin(distances))
    pair = (symbols[int(first[k])], symbols[int(second[k])])
    return float(distances[k]), int((distances < threshold).sum() // 2), pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="structure file (any format ASE reads)")
    parser.add_argument(
        "--min-distance",
        type=float,
        default=0.8,
        help="distance (A) below which a pair counts as a close contact (default 0.8)",
    )
    parser.add_argument(
        "--fail-below",
        type=float,
        default=None,
        help="exit with an error when the minimum distance is below this (A)",
    )
    parser.add_argument(
        "--fail-below-fraction",
        type=float,
        default=0.6,
        help="exit with an error when the minimum distance is below this fraction "
        "of the shortest expected bond, the sum of the closest pair's covalent "
        "radii (default 0.6; 0 disables)",
    )
    parser.add_argument(
        "--expect-atoms",
        type=int,
        default=None,
        help="exit with an error when the atom count differs from this",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    options = parser.parse_args()
    if options.fail_below_fraction < 0:
        parser.error("--fail-below-fraction must be 0 or positive")

    try:
        report = analyze(options.file, options.min_distance)
    except FileNotFoundError:
        sys.exit(f"no such file: {options.file}")
    except Exception as e:
        sys.exit(f"cannot read {options.file}: {e}")

    report["expected_atoms"] = options.expect_atoms
    if options.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"file:         {report['file']}")
        print(f"atoms:        {report['n_atoms']}  ({report['formula']})")
        print(f"pbc:          {report['pbc']}")
        if "cell_lengths_A" in report:
            print(f"cell (A):     {report['cell_lengths_A']}")
            print(f"angles (deg): {report['cell_angles_deg']}")
            print(f"volume (A^3): {report['volume_A3']}")
            print(f"density:      {report['density_g_cm3']} g/cm^3")
        print(f"min distance: {report['min_distance_A']} A")
        if report["shortest_expected_bond_A"] is not None:
            pair = "-".join(report["closest_pair"])
            print(
                f"expected bond: {report['shortest_expected_bond_A']} A ({pair}); "
                f"min distance is {report['min_distance_fraction']} of it"
            )
        print(f"close pairs:  {report['close_pairs']} (below {report['threshold_A']} A)")
        if options.expect_atoms is not None:
            print(f"expected:     {options.expect_atoms} atoms")

    if options.expect_atoms is not None and report["n_atoms"] != options.expect_atoms:
        sys.exit(
            f"{report['n_atoms']} atoms, but {options.expect_atoms} were expected: "
            f"check the duplication factors and the oriented cell's multiplicity"
        )
    minimum = report["min_distance_A"]
    if options.fail_below is not None and minimum is not None and minimum < options.fail_below:
        sys.exit(
            f"minimum interatomic distance {minimum} A is below "
            f"{options.fail_below} A: atoms overlap; rebuild the structure"
        )
    fraction = report["min_distance_fraction"]
    if (
        options.fail_below_fraction > 0
        and fraction is not None
        and fraction < options.fail_below_fraction
    ):
        sys.exit(
            f"minimum interatomic distance {minimum} A is {fraction} of the "
            f"{report['shortest_expected_bond_A']} A bond the closest pair's covalent "
            f"radii predict, below {options.fail_below_fraction}: atoms overlap; "
            f"rebuild the structure"
        )
    sys.exit(0)


if __name__ == "__main__":
    main()
