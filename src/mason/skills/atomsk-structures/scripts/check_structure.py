"""Sanity-check a built structure file: composition, cell, and close contacts.

Reads any structure file ASE understands and reports the facts a builder
can get wrong: atom count, formula, cell lengths and angles, density, and
the minimum interatomic distance under periodic boundary conditions. A
minimum distance far below a bond length means overlapping atoms — the
classic failure after a merge or a defect insertion — and should be fixed
by rebuilding, not by asking an optimizer to untangle it.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np


def analyze(path: str, threshold: float) -> dict[str, Any]:
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
    minimum, close = _closest(atoms, threshold)
    report["min_distance_A"] = round(minimum, 6) if minimum is not None else None
    report["close_pairs"] = close
    return report


def _closest(atoms: Any, threshold: float) -> tuple[float | None, int]:
    """(minimum pair distance, pairs below threshold), both PBC-aware.

    Small systems get the exact full distance matrix. Large ones use a
    neighbor list with a finite cutoff; a minimum reported as None then
    means "no pair closer than the cutoff", which is what the check needs.
    """
    n = len(atoms)
    if n < 2:
        return None, 0
    if n <= 1500:
        from ase.geometry import get_distances

        _, distances = get_distances(atoms.positions, cell=atoms.cell, pbc=atoms.pbc)
        off_diagonal = distances[~np.eye(n, dtype=bool)]
        minimum = float(off_diagonal.min())
        close = int((distances[np.triu(~np.eye(n, dtype=bool))] < threshold).sum())
        return minimum, close
    from ase.neighborlist import neighbor_list

    cutoff = max(2.0 * threshold, 3.0)
    distances = neighbor_list("d", atoms, cutoff)
    if len(distances) == 0:
        return None, 0
    return float(distances.min()), int((distances < threshold).sum() // 2)


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
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    options = parser.parse_args()

    try:
        report = analyze(options.file, options.min_distance)
    except FileNotFoundError:
        sys.exit(f"no such file: {options.file}")
    except Exception as e:
        sys.exit(f"cannot read {options.file}: {e}")

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
        print(f"close pairs:  {report['close_pairs']} (below {report['threshold_A']} A)")

    minimum = report["min_distance_A"]
    if options.fail_below is not None and minimum is not None and minimum < options.fail_below:
        sys.exit(
            f"minimum interatomic distance {minimum} A is below "
            f"{options.fail_below} A: atoms overlap; rebuild the structure"
        )
    sys.exit(0)


if __name__ == "__main__":
    main()
