"""Equation-of-state scan: single-point energies on a ladder of scaled cells.

Runs as-is (EMT copper) as a shakeout. For real work, adapt the three
constants below, then launch it as a traced run so every point has
provenance. The scan writes ``eos.json`` for ``scripts/fit_eos.py``.
"""

import json

from ase.build import bulk

from foundation import check
from foundation.tasks import single_point

STRUCTURE = bulk("Cu", "fcc", a=3.58)
ENGINE = "emt"
SCALES = [0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06]

points = []
for scale in SCALES:
    atoms = STRUCTURE.copy()
    atoms.set_cell(STRUCTURE.cell * scale, scale_atoms=True)
    _, info = single_point(atoms, engine=ENGINE, label=f"eos-{scale:.2f}")
    points.append({"volume": atoms.get_volume(), "energy": info["energy"]})
    print(f"scale {scale:.2f}: V = {atoms.get_volume():.3f} A^3, E = {info['energy']:.6f} eV")

with open("eos.json", "w", encoding="utf-8") as handle:
    json.dump(points, handle, indent=1)
print(f"wrote eos.json with {len(points)} points")


@check
def energies_are_finite() -> bool:
    return all(abs(point["energy"]) < 1e6 for point in points)


@check
def minimum_is_interior() -> bool:
    lowest = min(points, key=lambda point: point["energy"])
    return lowest is not points[0] and lowest is not points[-1]
