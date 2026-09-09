"""Temperature ladder under NPT: mean enthalpy, volume, and cell at each rung.

Runs as-is (EMT copper, a short hot ladder) as a shakeout. For real work,
adapt the constants below and launch it as a traced run. The ladder writes
``ramp.json`` for ``scripts/fit_thermal_ramp.py``: heat capacity from
dH/dT, thermal expansion from dV/dT (per axis when the cell is not
cubic), and latent heat between two phases' ladders.

Each rung is approached under Berendsen, which reaches a target quickly
but samples no ensemble, and then sampled under ASE's isotropic
Martyna-Tobias-Klein NPT, which does. Every row carries the atom count,
the pressure, H = E + PV, the measured temperature, the mean cell
lengths, and block standard errors, so the fit can use them.
"""

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms, units
from ase.build import bulk
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ase.md.velocitydistribution import Stationary, thermalize_momenta

from foundation import check, current_run
from slab.backends import close_calculator, get_calculator

STRUCTURE = bulk("Cu", "fcc", a=3.58, cubic=True) * (2, 2, 2)
ENGINE = "emt"
TIMESTEP_FS = 2.0
# One rung per row of ramp.json, walked in order. Five or more rungs per
# fitted window for a real number; three is the shakeout.
TEMPERATURES = (500.0, 800.0, 1100.0)
# Walk back down after the top rung and record the descent too, so the fit
# can test for hysteresis (superheating on the way up, supercooling down).
WALK_DOWN = False
# Shakeout lengths. Equilibrate for at least ten barostat time constants,
# and average long enough that the block error is small against the slope.
EQUILIBRATION_STEPS = 200
AVERAGING_STEPS = 200
SAMPLE_EVERY = 5
BLOCKS = 4
PRESSURE_BAR = 1.0
TAUT_FS = 100.0
TAUP_FS = 500.0
# Isothermal compressibility per bar (copper: about 7e-7 solid).
COMPRESSIBILITY_PER_BAR = 7e-7
# True lets each cell axis breathe on its own during the Berendsen approach,
# for hexagonal, tetragonal, or orthorhombic phases; the fit then reports
# an expansion coefficient per axis.
ANISOTROPIC = False
HOLD_TDAMP_FS = 100.0
HOLD_PDAMP_FS = 1000.0
RNG_SEED = 20  # seeds the initial velocities, so a rerun reproduces the run


def _approach(atoms: Atoms, temperature_k: float) -> NPTBerendsen:
    cls = Inhomogeneous_NPTBerendsen if ANISOTROPIC else NPTBerendsen
    return cls(
        atoms,
        timestep=TIMESTEP_FS * units.fs,
        temperature_K=temperature_k,
        pressure_au=PRESSURE_BAR * units.bar,
        taut=TAUT_FS * units.fs,
        taup=TAUP_FS * units.fs,
        compressibility_au=COMPRESSIBILITY_PER_BAR / units.bar,
    )


def _sampler(atoms: Atoms, temperature_k: float) -> IsotropicMTKNPT:
    return IsotropicMTKNPT(
        atoms,
        timestep=TIMESTEP_FS * units.fs,
        temperature_K=temperature_k,
        pressure_au=PRESSURE_BAR * units.bar,
        tdamp=HOLD_TDAMP_FS * units.fs,
        pdamp=HOLD_PDAMP_FS * units.fs,
    )


def _block_se(values: list[float]) -> float:
    blocks = [float(np.mean(b)) for b in np.array_split(np.asarray(values), BLOCKS)]
    return float(np.std(blocks, ddof=1) / np.sqrt(len(blocks)))


def _rung(system: Atoms, temperature: float, direction: str) -> dict[str, Any]:
    _approach(system, temperature).run(EQUILIBRATION_STEPS)
    sampler = _sampler(system, temperature)
    enthalpies: list[float] = []
    volumes: list[float] = []
    measured: list[float] = []
    lengths: list[np.ndarray] = []
    pressure_ev_a3 = PRESSURE_BAR * units.bar
    for _ in range(AVERAGING_STEPS // SAMPLE_EVERY):
        sampler.run(SAMPLE_EVERY)
        volume = system.get_volume()
        energy = system.get_potential_energy() + system.get_kinetic_energy()
        enthalpies.append(energy + pressure_ev_a3 * volume)
        volumes.append(volume)
        measured.append(system.get_temperature())
        lengths.append(np.asarray(system.cell.lengths()))
    return {
        "T": temperature,
        "T_measured": float(np.mean(measured)),
        "direction": direction,
        "N": len(system),
        "mass_amu": float(system.get_masses().sum()),
        "P_bar": PRESSURE_BAR,
        "H": float(np.mean(enthalpies)),
        "H_se": _block_se(enthalpies),
        "E": float(np.mean(enthalpies) - pressure_ev_a3 * np.mean(volumes)),
        "V": float(np.mean(volumes)),
        "V_se": _block_se(volumes),
        "L": [float(x) for x in np.mean(lengths, axis=0)],
    }


calculator = get_calculator(ENGINE)
try:
    system: Atoms = STRUCTURE.copy()
    system.calc = calculator
    thermalize_momenta(
        system, temperature_K=TEMPERATURES[0], rng=np.random.default_rng(RNG_SEED)
    )
    Stationary(system)
    rows: list[dict[str, Any]] = []
    ladder = list(TEMPERATURES)
    if WALK_DOWN:
        ladder += list(reversed(TEMPERATURES[:-1]))
    for i, temperature in enumerate(ladder):
        direction = "up" if i < len(TEMPERATURES) else "down"
        rows.append(_rung(system, temperature, direction))
        print(
            f"T = {temperature:.0f} K ({direction}): <H> = {rows[-1]['H']:.4f} eV, "
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
    return len(rows) == len(ladder)


@check
def enthalpy_rises_with_temperature() -> bool:
    upward = [float(row["H"]) for row in rows if row["direction"] == "up"]
    return all(later > earlier for earlier, later in itertools.pairwise(upward))


@check
def volumes_stay_physical() -> bool:
    reference = STRUCTURE.get_volume()
    return all(0.5 * reference < float(row["V"]) < 2.0 * reference for row in rows)
