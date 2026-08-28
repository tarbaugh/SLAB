"""Melt-quench: NPT melt, then a ladder of quench rates to a glass.

Runs as-is (EMT copper, absurdly fast rates) as a shakeout. For real work,
adapt the constants below and launch it as a traced run. Each rate writes
``quench-<rate>Kps.traj`` for ``scripts/quench_report.py``, and the run
keeps every trajectory plus a ``quench.json`` summary.
"""

import json
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.build import bulk
from ase.io.trajectory import Trajectory
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import Stationary, thermalize_momenta

from foundation import check, current_run
from slab.backends import close_calculator, get_calculator

STRUCTURE = bulk("Cu", "fcc", a=3.58, cubic=True) * (2, 2, 2)
ENGINE = "emt"
TIMESTEP_FS = 2.0
T_MELT = 1800.0
MELT_STEPS = 400
T_FINAL = 300.0
QUENCH_RATES_K_PER_PS = (2500.0, 1250.0)  # shakeout values; real glasses want 10-1000 K/ps
SEGMENT_STEPS = 10  # temperature target updates (and frames) every this many steps
PRESSURE_BAR = 1.0
TAUT_FS = 100.0
TAUP_FS = 500.0
COMPRESSIBILITY_PER_BAR = 5e-6  # a metal-like guess; the barostat only needs the scale
RNG_SEED = 20  # seeds the initial velocities, so a rerun reproduces the run


def _dynamics(atoms: Atoms, temperature_k: float) -> NPTBerendsen:
    return NPTBerendsen(
        atoms,
        timestep=TIMESTEP_FS * units.fs,
        temperature_K=temperature_k,
        pressure_au=PRESSURE_BAR * units.bar,
        taut=TAUT_FS * units.fs,
        taup=TAUP_FS * units.fs,
        compressibility_au=COMPRESSIBILITY_PER_BAR / units.bar,
    )


calculator = get_calculator(ENGINE)
try:
    melt = STRUCTURE.copy()
    melt.calc = calculator
    thermalize_momenta(melt, temperature_K=T_MELT, rng=np.random.default_rng(RNG_SEED))
    Stationary(melt)
    _dynamics(melt, T_MELT).run(MELT_STEPS)
    melt_volume = melt.get_volume()
    print(f"melted {len(melt)} atoms at {T_MELT:.0f} K; V = {melt_volume:.2f} A^3")

    summaries: dict[str, list[dict[str, float]]] = {}
    trajectories: list[Path] = []
    for rate in QUENCH_RATES_K_PER_PS:
        system = melt.copy()  # positions, cell, and momenta of the common melt
        system.calc = calculator
        dynamics = _dynamics(system, T_MELT)
        name = f"quench-{rate:g}Kps.traj"
        writer = Trajectory(name, "w", system)
        trajectories.append(Path(name))
        rows: list[dict[str, float]] = []
        kelvin_per_step = rate * TIMESTEP_FS / 1000.0
        target = T_MELT
        step = 0
        while target > T_FINAL:
            target = max(T_FINAL, target - kelvin_per_step * SEGMENT_STEPS)
            dynamics.set_temperature(temperature_K=target)
            dynamics.run(SEGMENT_STEPS)
            step += SEGMENT_STEPS
            writer.write(system)
            rows.append(
                {
                    "t_ps": step * TIMESTEP_FS / 1000.0,
                    "T_target": target,
                    "T": system.get_temperature(),
                    "V": system.get_volume(),
                    "E": system.get_potential_energy() + system.get_kinetic_energy(),
                }
            )
        writer.close()
        summaries[f"{rate:g}"] = rows
        print(
            f"rate {rate:g} K/ps: {step} steps to {T_FINAL:.0f} K, "
            f"final V = {rows[-1]['V']:.2f} A^3 ({name})"
        )
finally:
    close_calculator(calculator)

with open("quench.json", "w", encoding="utf-8") as handle:
    summary = {"engine": ENGINE, "melt_volume_A3": melt_volume, "rates": summaries}
    json.dump(summary, handle, indent=1)
print(f"wrote quench.json with {len(summaries)} rate(s)")

active = current_run()
if active is not None:
    active.keep("quench.json", Path("quench.json"))
    for path in trajectories:
        active.keep(path.name, path, role="intermediate")


@check
def every_quench_reached_the_final_temperature() -> bool:
    return all(
        rows[-1]["T_target"] == T_FINAL and rows[-1]["T"] < T_MELT / 2.0
        for rows in summaries.values()
    )


@check
def volumes_stay_physical() -> bool:
    return all(
        0.0 < row["V"] < 3.0 * melt_volume for rows in summaries.values() for row in rows
    )
