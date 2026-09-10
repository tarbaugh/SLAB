"""Melt-quench: NPT melt, a ladder of quench rates, and an isothermal hold.

Runs as-is (EMT copper, absurdly fast rates, one replica) as a shakeout.
For real work, adapt the constants below and launch it as a traced run.
Each rate and replica writes ``quench-<rate>Kps-r<k>.traj`` for
``scripts/quench_report.py``; the run keeps every trajectory plus a
``quench.json`` summary that records the hold length.

Integrators: the melt and the ramp use the Berendsen thermostat and
barostat, which reach a target quickly but sample no ensemble. The hold
at the final temperature, where the density is measured, uses ASE's
isotropic Martyna-Tobias-Klein NPT, which samples the isothermal-isobaric
ensemble.
"""

import json
from pathlib import Path

import numpy as np
from ase import Atoms, units
from ase.build import bulk
from ase.io.trajectory import Trajectory
from ase.md.nose_hoover_chain import IsotropicMTKNPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import Stationary, thermalize_momenta

from foundation import check, current_run
from slab.backends import close_calculator, get_calculator

# The shakeout cell is 32 atoms. A reportable density wants 500 to 1000
# atoms or more, so a glass has room for medium-range order.
STRUCTURE = bulk("Cu", "fcc", a=3.58, cubic=True) * (2, 2, 2)
ENGINE = "emt"
TIMESTEP_FS = 2.0
T_MELT = 1800.0
MELT_STEPS = 400
T_FINAL = 300.0
# Shakeout rates. Real glasses are made at 0.01 to 10 K/ps (1e10 to 1e13
# K/s); 1800 -> 300 K at 0.1 K/ps is 15 ns, routine with an EAM or MLIP.
QUENCH_RATES_K_PER_PS = (2500.0, 1250.0)
# Independent melts per rate (different seeds). Two or more for a spread.
REPLICAS = 1
SEGMENT_STEPS = 10  # temperature target updates (and frames) every this many steps
HOLD_STEPS = 100  # isothermal steps at T_FINAL; the density is averaged over these
PRESSURE_BAR = 1.0
TAUT_FS = 100.0
TAUP_FS = 500.0
# Isothermal compressibility of the material, per bar: copper is about
# 7e-7 solid and 1.5e-6 liquid. It sets the Berendsen barostat's effective
# time constant together with TAUP_FS.
COMPRESSIBILITY_PER_BAR = 1.5e-6
HOLD_TDAMP_FS = 100.0
HOLD_PDAMP_FS = 1000.0
RNG_SEED = 20  # seeds the initial velocities, so a rerun reproduces the run


def _berendsen(atoms: Atoms, temperature_k: float) -> NPTBerendsen:
    return NPTBerendsen(
        atoms,
        timestep=TIMESTEP_FS * units.fs,
        temperature_K=temperature_k,
        pressure_au=PRESSURE_BAR * units.bar,
        taut=TAUT_FS * units.fs,
        taup=TAUP_FS * units.fs,
        compressibility_au=COMPRESSIBILITY_PER_BAR / units.bar,
    )


def _hold(atoms: Atoms, temperature_k: float) -> IsotropicMTKNPT:
    return IsotropicMTKNPT(
        atoms,
        timestep=TIMESTEP_FS * units.fs,
        temperature_K=temperature_k,
        pressure_au=PRESSURE_BAR * units.bar,
        tdamp=HOLD_TDAMP_FS * units.fs,
        pdamp=HOLD_PDAMP_FS * units.fs,
    )


def _row(system: Atoms, step: int, target: float, phase: str) -> dict[str, float | str]:
    return {
        "t_ps": step * TIMESTEP_FS / 1000.0,
        "T_target": target,
        "T": system.get_temperature(),
        "V": system.get_volume(),
        "E": system.get_potential_energy() + system.get_kinetic_energy(),
        "phase": phase,
    }


calculator = get_calculator(ENGINE)
hold_frames = HOLD_STEPS // SEGMENT_STEPS
try:
    summaries: dict[str, list[dict[str, float | str]]] = {}
    melt_volumes: list[float] = []
    trajectories: list[Path] = []
    for replica in range(1, REPLICAS + 1):
        melt = STRUCTURE.copy()
        melt.calc = calculator
        rng = np.random.default_rng(RNG_SEED + replica)
        thermalize_momenta(melt, temperature_K=T_MELT, rng=rng)
        Stationary(melt)
        _berendsen(melt, T_MELT).run(MELT_STEPS)
        melt_volumes.append(melt.get_volume())
        print(
            f"replica {replica}: melted {len(melt)} atoms at {T_MELT:.0f} K; "
            f"V = {melt_volumes[-1]:.2f} A^3"
        )
        for rate in QUENCH_RATES_K_PER_PS:
            system = melt.copy()  # positions, cell, and momenta of this replica's melt
            system.calc = calculator
            dynamics = _berendsen(system, T_MELT)
            name = f"quench-{rate:g}Kps-r{replica}.traj"
            writer = Trajectory(name, "w", system)
            trajectories.append(Path(name))
            rows: list[dict[str, float | str]] = []
            kelvin_per_step = rate * TIMESTEP_FS / 1000.0
            target = T_MELT
            step = 0
            while target > T_FINAL:
                target = max(T_FINAL, target - kelvin_per_step * SEGMENT_STEPS)
                dynamics.set_temperature(temperature_K=target)
                dynamics.run(SEGMENT_STEPS)
                step += SEGMENT_STEPS
                writer.write(system)
                rows.append(_row(system, step, target, "ramp"))
            hold = _hold(system, T_FINAL)
            for _ in range(hold_frames):
                hold.run(SEGMENT_STEPS)
                step += SEGMENT_STEPS
                writer.write(system)
                rows.append(_row(system, step, T_FINAL, "hold"))
            writer.close()
            summaries[f"{rate:g}-r{replica}"] = rows
            print(
                f"rate {rate:g} K/ps, replica {replica}: {step} steps to {T_FINAL:.0f} K "
                f"and {HOLD_STEPS} held, final V = {rows[-1]['V']:.2f} A^3 ({name})"
            )
finally:
    close_calculator(calculator)

with open("quench.json", "w", encoding="utf-8") as handle:
    summary = {
        "engine": ENGINE,
        "n_atoms": len(STRUCTURE),
        "pressure_bar": PRESSURE_BAR,
        "melt_volumes_A3": melt_volumes,
        "hold_steps": HOLD_STEPS,
        "hold_frames": hold_frames,
        "rates": summaries,
    }
    json.dump(summary, handle, indent=1)
print(f"wrote quench.json with {len(summaries)} trajectory summaries; hold_frames = {hold_frames}")

active = current_run()
if active is not None:
    active.keep("quench.json", Path("quench.json"))
    for path in trajectories:
        active.keep(path.name, path, role="intermediate")


@check
def every_quench_reached_the_final_temperature() -> bool:
    return all(
        rows[-1]["T_target"] == T_FINAL and float(rows[-1]["T"]) < T_MELT / 2.0
        for rows in summaries.values()
    )


@check
def volumes_stay_physical() -> bool:
    largest = max(melt_volumes)
    return all(
        0.0 < float(row["V"]) < 3.0 * largest for rows in summaries.values() for row in rows
    )


@check
def every_hold_was_recorded() -> bool:
    return all(
        sum(1 for row in rows if row["phase"] == "hold") == hold_frames
        for rows in summaries.values()
    )
