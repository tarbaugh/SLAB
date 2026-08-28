"""Energy-strain scan: one energy ladder per deformation mode.

Runs as-is (EMT copper, cubic) as a shakeout. For real work, adapt the
constants below, then launch it as a traced run so every point has
provenance. The scan writes ``elastic.json`` for ``scripts/fit_elastic.py``.
"""

import json

import numpy as np
from ase import Atoms
from ase.build import bulk

from foundation import check
from foundation.tasks import relax, single_point

# The shakeout lattice constant is EMT's own equilibrium: the fit script
# checks for a linear term, and a strained reference would trip it.
STRUCTURE = bulk("Cu", "fcc", a=3.5898, cubic=True)
ENGINE = "emt"
SYMMETRY = "cubic"  # "isotropic" | "cubic" | "orthorhombic"
DELTAS = [-0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015]
# Materials with internal degrees of freedom (layered, molecular, ribboned
# crystals) need the ions relaxed at each fixed strained cell; the plain
# single-point ladder overstates their stiffness.
RELAX_INTERNAL = False

PATTERNS = {
    "e1": (1, 0, 0, 0, 0, 0),
    "e2": (0, 1, 0, 0, 0, 0),
    "e3": (0, 0, 1, 0, 0, 0),
    "e4": (0, 0, 0, 1, 0, 0),
    "e5": (0, 0, 0, 0, 1, 0),
    "e6": (0, 0, 0, 0, 0, 1),
    "e12": (1, 1, 0, 0, 0, 0),
    "e13": (1, 0, 1, 0, 0, 0),
    "e23": (0, 1, 1, 0, 0, 0),
}
MODES = {
    "isotropic": ("e1", "e4"),
    "cubic": ("e1", "e4", "e12"),
    "orthorhombic": ("e1", "e2", "e3", "e4", "e5", "e6", "e12", "e13", "e23"),
}[SYMMETRY]


def strained(atoms: Atoms, voigt: tuple[int, ...], delta: float) -> Atoms:
    """A copy of *atoms* under the Voigt strain pattern scaled by *delta*.

    Components 4-6 are engineering shears, matching the fit script."""
    v = [component * delta for component in voigt]
    epsilon = np.array(
        [
            [v[0], v[5] / 2.0, v[4] / 2.0],
            [v[5] / 2.0, v[1], v[3] / 2.0],
            [v[4] / 2.0, v[3] / 2.0, v[2]],
        ]
    )
    deformed: Atoms = atoms.copy()
    deformed.set_cell(deformed.get_cell() @ (np.eye(3) + epsilon), scale_atoms=True)
    return deformed


modes: dict[str, list[dict[str, float]]] = {}
for mode in MODES:
    rows: list[dict[str, float]] = []
    for delta in DELTAS:
        atoms = strained(STRUCTURE, PATTERNS[mode], delta)
        label = f"{mode}{delta:+.3f}"
        if RELAX_INTERNAL:
            _, info = relax(atoms, engine=ENGINE, label=label)
        else:
            _, info = single_point(atoms, engine=ENGINE, label=label)
        rows.append({"delta": delta, "energy": info["energy"]})
        print(f"{mode} delta {delta:+.3f}: E = {info['energy']:.6f} eV")
    modes[mode] = rows

payload = {"symmetry": SYMMETRY, "v0": STRUCTURE.get_volume(), "modes": modes}
with open("elastic.json", "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=1)
print(f"wrote elastic.json with {len(modes)} mode(s) x {len(DELTAS)} points")


@check
def energies_are_finite() -> bool:
    return all(abs(row["energy"]) < 1e6 for rows in modes.values() for row in rows)


@check
def each_mode_minimum_is_interior() -> bool:
    for rows in modes.values():
        ordered = sorted(rows, key=lambda row: row["delta"])
        lowest = min(ordered, key=lambda row: row["energy"])
        if lowest is ordered[0] or lowest is ordered[-1]:
            return False
    return True
