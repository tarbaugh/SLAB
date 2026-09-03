"""Equation-of-state scan: single-point energies on a ladder of scaled cells.

Runs as-is (EMT copper) as a shakeout. For real work, adapt the constants
below, then launch it as a traced run so every point has provenance. The
scan writes ``eos.json`` for ``scripts/fit_eos.py``.

The scan holds the k-mesh, cutoff, and smearing fixed across volumes:
override them explicitly in ``calculator_options`` so a spacing-derived
mesh does not change between points.
"""

import json

from ase.build import bulk

from foundation import check
from foundation.tasks import relax, single_point

STRUCTURE = bulk("Cu", "fcc", a=3.58)
ENGINE = "emt"
# Volume fractions of the reference cell. Seven points over +/-6 % in
# volume is the Delta-protocol range; widen only when the minimum sits
# at an edge, and state the range in the report.
VOLUME_SCALES = [0.94, 0.96, 0.98, 1.00, 1.02, 1.04, 1.06]
# True relaxes the internal coordinates at each fixed volume (needed for
# cells with free Wyckoff parameters); the cell itself is never relaxed.
RELAX_INTERNAL = False
RELAX_FMAX = 0.005

points = []
for volume_scale in VOLUME_SCALES:
    atoms = STRUCTURE.copy()
    atoms.set_cell(STRUCTURE.cell * volume_scale ** (1.0 / 3.0), scale_atoms=True)
    label = f"eos-{volume_scale:.2f}"
    if RELAX_INTERNAL:
        atoms, info = relax(atoms, engine=ENGINE, fmax=RELAX_FMAX, label=label)
    else:
        _, info = single_point(atoms, engine=ENGINE, label=label)
    points.append({"volume": atoms.get_volume(), "energy": info["energy"]})
    print(
        f"V/V_ref {volume_scale:.2f}: V = {atoms.get_volume():.3f} A^3, "
        f"E = {info['energy']:.6f} eV"
    )

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
