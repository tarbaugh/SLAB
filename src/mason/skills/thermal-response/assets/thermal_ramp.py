"""Temperature ladder under NPT: mean enthalpy and volume at each rung.

Runs as-is (EMT copper, a short hot ladder) as a shakeout. For real work,
adapt the constants below and launch it as a traced run. The ladder writes
``ramp.json`` for ``scripts/fit_thermal_ramp.py``: heat capacity from
dE/dT, thermal expansion from dV/dT, and latent heat between two phases'
ladders.
"""

import itertools
import json
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.build import bulk
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import Stationary, thermalize_momenta

from foundation import check, current_run
from slab.backends import close_calculator, get_calculator

STRUCTURE = bulk("Cu", "fcc", a=3.58, cubic=True) * (2, 2, 2)
ENGINE = "emt"
TIMESTEP_FS = 2.0
TEMPERATURES = (500.0, 800.0, 1100.0)  # one rung per row of ramp.json, walked in order
EQUILIBRATION_STEPS = 200
AVERAGING_STEPS = 200
SAMPLE_EVERY = 5
PRESSURE_BAR = 1.0
TAUT_FS = 100.0
TAUP_FS = 500.0
COMPRESSIBILITY_PER_BAR = 5e-6
RNG_SEED = 20  # seeds the initial velocities, so a rerun reproduces the run


calculator = get_calculator(ENGINE)
try:
    system: Atoms = STRUCTURE.copy()
    system.calc = calculator
    thermalize_momenta(
        system, temperature_K=TEMPERATURES[0], rng=np.random.default_rng(RNG_SEED)
    )
    Stationary(system)
    dynamics = NPTBerendsen(
        system,
        timestep=TIMESTEP_FS * units.fs,
        temperature_K=TEMPERATURES[0],
        pressure_au=PRESSURE_BAR * units.bar,
        taut=TAUT_FS * units.fs,
        taup=TAUP_FS * units.fs,
        compressibility_au=COMPRESSIBILITY_PER_BAR / units.bar,
    )

    rows: list[dict[str, float]] = []
    for temperature in TEMPERATURES:
        dynamics.set_temperature(temperature_K=temperature)
        dynamics.run(EQUILIBRATION_STEPS)
        energies: list[float] = []
        volumes: list[float] = []
        measured: list[float] = []
        for _ in range(AVERAGING_STEPS // SAMPLE_EVERY):
            dynamics.run(SAMPLE_EVERY)
            energies.append(system.get_potential_energy() + system.get_kinetic_energy())
            volumes.append(system.get_volume())
            measured.append(system.get_temperature())
        rows.append(
            {
                "T": temperature,
                "T_measured": sum(measured) / len(measured),
                "E": sum(energies) / len(energies),
                "V": sum(volumes) / len(volumes),
            }
        )
        print(
            f"T = {temperature:.0f} K: <E> = {rows[-1]['E']:.4f} eV, "
            f"<V> = {rows[-1]['V']:.2f} A^3 (measured {rows[-1]['T_measured']:.0f} K)"
        )
finally:
    close_calculator(calculator)

with open("ramp.json", "w", encoding="utf-8") as handle:
    json.dump(rows, handle, indent=1)
print(f"wrote ramp.json with {len(rows)} rung(s)")

active = current_run()
if active is not None:
    active.keep("ramp.json", Path("ramp.json"))


@check
def one_row_per_rung() -> bool:
    return len(rows) == len(TEMPERATURES)


@check
def energy_rises_with_temperature() -> bool:
    energies = [row["E"] for row in rows]
    return all(later > earlier for earlier, later in itertools.pairwise(energies))


@check
def volumes_stay_physical() -> bool:
    reference = STRUCTURE.get_volume()
    return all(0.5 * reference < row["V"] < 2.0 * reference for row in rows)
