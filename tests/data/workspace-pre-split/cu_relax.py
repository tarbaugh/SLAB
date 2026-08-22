"""Pre-split fixture workflow: relax a rattled 8-atom Cu cell under EMT.

Recorded with the single-package `slab` layout so the three-package split can
prove that a workspace written before the move still opens after it.
"""

from ase.build import bulk

from slab import check, converged
from slab.tasks import relax

atoms = bulk("Cu", "fcc", a=3.58, cubic=True) * (2, 1, 1)  # 8 atoms
atoms.rattle(stdev=0.04, seed=11)

relaxed, info = relax(atoms, engine="emt", fmax=0.05, label="cu")
print("energy (eV):", info["energy"])


@check
def forces_converged():
    return converged(info["fmax"], below=0.05)
