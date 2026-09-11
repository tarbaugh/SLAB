"""Temperature ladder under NPT inside LAMMPS: mean enthalpy, volume, and cell per rung.

Runs as-is as a shakeout: 108 argon atoms under a Lennard-Jones potential
that needs no potential file, a short cold ladder. For real work change
STRUCTURE, the POTENTIAL lines (the lammps-potentials skill gives them,
`pair_style grace` included), the rungs, and the lengths, and keep the
checks. The ladder writes ``ramp.json`` for ``scripts/fit_thermal_ramp.py``:
heat capacity from dH/dT, thermal expansion from dV/dT (per axis when the
cell is not cubic), and latent heat between two phases' ladders.

One ``run_lammps`` call runs the whole ladder as one continuous
trajectory. Each rung is two ``run`` commands under Nose-Hoover NPT
(``fix npt``): an equilibration whose rows are discarded, and an
averaging span whose thermo rows the workflow reads back from the run's
``-thermo.json`` artifact. Every row of ``ramp.json`` carries the atom
count, the pressure, H = E + PV at the set pressure, the measured
temperature, the mean cell lengths, and block standard errors, so the
fit can use them.
"""

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from ase import units
from ase.build import bulk

from foundation import check, current_run
from foundation.tasks import run_lammps

STRUCTURE = bulk("Ar", "fcc", a=5.26, cubic=True) * (3, 3, 3)
# Argon: epsilon 0.0104 eV, sigma 3.40 A, in metal units. The masses ride
# in structure.data, which run_lammps writes from STRUCTURE.
POTENTIAL = """\
pair_style lj/cut 8.5
pair_coeff 1 1 0.0104 3.40
pair_modify shift yes
"""
TIMESTEP_PS = 0.002
# One rung per row of ramp.json, walked in order. Five or more rungs per
# fitted window for a real number; three is the shakeout.
TEMPERATURES = (20.0, 40.0, 60.0)
# Walk back down after the top rung and record the descent too, so the fit
# can test for hysteresis (superheating on the way up, supercooling down).
WALK_DOWN = False
# Shakeout lengths. Equilibrate for at least ten barostat time constants,
# and average long enough that the block error is small against the slope.
EQUILIBRATION_STEPS = 500
AVERAGING_STEPS = 500
SAMPLE_EVERY = 5
BLOCKS = 4
PRESSURE_BAR = 1.0
TDAMP_PS = 0.1  # thermostat time constant, about 100 timesteps
PDAMP_PS = 1.0  # barostat time constant, about 1000 timesteps
# `aniso` lets each cell length breathe on its own (hexagonal, tetragonal,
# orthorhombic phases), so the fit reports an expansion coefficient per axis.
ANISOTROPIC = False
SEED = 20  # seeds the initial velocities, so a rerun reproduces the run
LABEL = "ramp"

ladder = list(TEMPERATURES)
if WALK_DOWN:
    ladder += list(reversed(TEMPERATURES[:-1]))
coupling = "aniso" if ANISOTROPIC else "iso"
rungs = "".join(
    f"""
# rung {i + 1}: {temperature:g} K, {EQUILIBRATION_STEPS} steps discarded, {AVERAGING_STEPS} averaged
fix integrate all npt temp {temperature} {temperature} {TDAMP_PS} &
    {coupling} {PRESSURE_BAR} {PRESSURE_BAR} {PDAMP_PS}
run {EQUILIBRATION_STEPS}
run {AVERAGING_STEPS}
"""
    for i, temperature in enumerate(ladder)
)
SCRIPT = f"""\
units metal
atom_style atomic
boundary p p p
read_data structure.data

{POTENTIAL}
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

timestep {TIMESTEP_PS}
thermo {SAMPLE_EVERY}
thermo_style custom step temp pe ke etotal press vol lx ly lz
thermo_modify flush yes
velocity all create {TEMPERATURES[0]} {SEED} mom yes rot yes dist gaussian
{rungs}
write_data {LABEL}-final.data
"""


def _block_se(values: np.ndarray) -> float:
    blocks = [float(np.mean(b)) for b in np.array_split(values, BLOCKS)]
    return float(np.std(blocks, ddof=1) / np.sqrt(len(blocks)))


def _rung(table: dict[str, Any], temperature: float, direction: str) -> dict[str, Any]:
    """One ramp.json row from the averaging table's rows, its first row dropped."""
    columns = table["columns"]
    block = np.asarray(table["rows"][1:], dtype=float)
    column = {name: block[:, index] for index, name in enumerate(columns)}
    pressure_ev_a3 = PRESSURE_BAR * units.bar
    volumes = column["Volume"]
    enthalpies = column["TotEng"] + pressure_ev_a3 * volumes
    lengths = np.stack([column["Lx"], column["Ly"], column["Lz"]], axis=1)
    return {
        "T": temperature,
        "T_measured": float(np.mean(column["Temp"])),
        "direction": direction,
        "N": len(STRUCTURE),
        "mass_amu": float(STRUCTURE.get_masses().sum()),
        "P_bar": PRESSURE_BAR,
        "H": float(np.mean(enthalpies)),
        "H_se": _block_se(enthalpies),
        "E": float(np.mean(column["TotEng"])),
        "V": float(np.mean(volumes)),
        "V_se": _block_se(volumes),
        "L": [float(x) for x in np.mean(lengths, axis=0)],
    }


result, info = run_lammps(SCRIPT, atoms=STRUCTURE, label=LABEL)
active = current_run()
assert active is not None, "run this template through launch_workflow"
tables = json.loads(
    active.artifacts.get(info["artifacts"][f"{LABEL}-thermo.json"]).read_text(encoding="utf-8")
)
print(
    f"LAMMPS {info['version']}: {result['steps']} steps over {len(ladder)} rung(s), "
    f"{len(tables)} thermo tables"
)
rows: list[dict[str, Any]] = []
for i, temperature in enumerate(ladder):
    direction = "up" if i < len(TEMPERATURES) else "down"
    rows.append(_rung(tables[2 * i + 1], temperature, direction))
    print(
        f"T = {temperature:.0f} K ({direction}): <H> = {rows[-1]['H']:.4f} eV, "
        f"<V> = {rows[-1]['V']:.2f} A^3 (measured {rows[-1]['T_measured']:.0f} K)"
    )

with open("ramp.json", "w", encoding="utf-8") as handle:
    json.dump(rows, handle, indent=1)
print(f"wrote ramp.json with {len(rows)} rung(s)")
active.keep("ramp.json", Path("ramp.json"))


@check
def one_row_per_rung() -> bool:
    return len(rows) == len(ladder) and len(tables) == 2 * len(ladder)


@check
def enthalpy_rises_with_temperature() -> bool:
    upward = [float(row["H"]) for row in rows if row["direction"] == "up"]
    return all(later > earlier for earlier, later in itertools.pairwise(upward))


@check
def volumes_stay_physical() -> bool:
    reference = STRUCTURE.get_volume()
    return all(0.5 * reference < float(row["V"]) < 2.0 * reference for row in rows)
