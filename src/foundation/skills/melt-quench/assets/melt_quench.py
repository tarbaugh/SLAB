"""Melt-quench inside LAMMPS: NPT melt, a ladder of quench rates, a hold.

Runs as-is as a shakeout: 108 argon atoms under a Lennard-Jones potential
that needs no potential file, held at 1 kbar so the fluid stays dense
above its boiling point, with absurdly fast rates and one replica. For
real work change STRUCTURE, the POTENTIAL lines (the lammps-potentials
skill gives them, `pair_style grace` included), the temperatures, the
rates, and the lengths, and keep the checks.

One ``run_lammps`` call per replica. The script melts the cell under
``fix npt``, writes a restart, and then for each rate reads the melt
back, ramps the thermostat target linearly to the final temperature
(``fix npt temp T_MELT T_FINAL``), and holds at the final temperature
under the same Nose-Hoover NPT while a dump records the hold. The
workflow reads each hold dump back from the run's artifacts and writes it
as ``quench-<rate>Kps-r<k>.traj`` with the masses, the form
``scripts/quench_report.py`` reads, plus a ``quench.json`` summary that
records the hold length in frames.
"""

import json
from pathlib import Path
from typing import Any

from ase.build import bulk
from ase.io import read, write

from foundation import check, current_run
from foundation.tasks import run_lammps

# The shakeout cell is 108 atoms. A reportable density wants 500 to 1000
# atoms or more, so a glass has room for medium-range order.
STRUCTURE = bulk("Ar", "fcc", a=5.26, cubic=True) * (3, 3, 3)
# Argon: epsilon 0.0104 eV, sigma 3.40 A, in metal units. The masses ride
# in structure.data, which run_lammps writes from STRUCTURE.
POTENTIAL = """\
pair_style lj/cut 8.5
pair_coeff 1 1 0.0104 3.40
pair_modify shift yes
"""
TIMESTEP_PS = 0.002
T_MELT = 200.0
MELT_STEPS = 400
T_FINAL = 20.0
# Shakeout rates. Real glasses are made at 0.01 to 10 K/ps (1e10 to 1e13
# K/s); 1800 -> 300 K at 0.1 K/ps is 15 ns, routine with an EAM or MLIP.
QUENCH_RATES_K_PER_PS = (2500.0, 1250.0)
# Independent melts per rate (different seeds). Two or more for a spread.
REPLICAS = 1
HOLD_STEPS = 100  # isothermal steps at T_FINAL; the density is averaged over these
DUMP_EVERY = 10  # hold frames every this many steps
THERMO_EVERY = 10
PRESSURE_BAR = 1000.0
TDAMP_PS = 0.1  # thermostat time constant, about 100 timesteps
PDAMP_PS = 1.0  # barostat time constant, about 1000 timesteps
SEED = 20  # seeds the initial velocities; replica k adds k, so a rerun reproduces the run
LABEL = "quench"


def _npt(t_start: float, t_stop: float) -> str:
    """Nose-Hoover NPT, isotropic, ramping the target from *t_start* to *t_stop*."""
    return (
        f"fix integrate all npt temp {t_start} {t_stop} {TDAMP_PS} "
        f"iso {PRESSURE_BAR} {PRESSURE_BAR} {PDAMP_PS}"
    )


def _script(replica: int) -> str:
    """The whole input for one replica: melt, restart, then each rate."""
    head = f"""\
units metal
atom_style atomic
boundary p p p
read_data structure.data

{POTENTIAL}
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

timestep {TIMESTEP_PS}
thermo {THERMO_EVERY}
thermo_style custom step temp pe ke etotal press vol
thermo_modify flush yes

velocity all create {T_MELT} {SEED + replica} mom yes rot yes dist gaussian
{_npt(T_MELT, T_MELT)}
run {MELT_STEPS}
write_restart melt.restart
"""
    blocks = []
    for rate in QUENCH_RATES_K_PER_PS:
        ramp_steps = round((T_MELT - T_FINAL) / (rate * TIMESTEP_PS))
        name = f"quench-{rate:g}Kps-r{replica}"
        blocks.append(f"""\

# rate {rate:g} K/ps: {ramp_steps} steps from {T_MELT:g} K to {T_FINAL:g} K, then the hold
clear
read_restart melt.restart
{POTENTIAL}
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes
timestep {TIMESTEP_PS}
thermo {THERMO_EVERY}
thermo_style custom step temp pe ke etotal press vol
thermo_modify flush yes
{_npt(T_MELT, T_FINAL)}
run {ramp_steps}
unfix integrate
reset_timestep 1
{_npt(T_FINAL, T_FINAL)}
dump hold all custom {DUMP_EVERY} {name}.dump id type x y z
dump_modify hold sort id
run {HOLD_STEPS}
undump hold
""")
    return head + "".join(blocks)


def _kept_path(info: dict[str, Any], suffix: str) -> Path:
    """The stored file of the artifact whose kept name ends with *suffix*."""
    active = current_run()
    assert active is not None, "run this template through launch_workflow"
    matches = [name for name in info["artifacts"] if name.endswith(suffix)]
    assert len(matches) == 1, f"{suffix}: kept as {matches}"
    return active.artifacts.get(info["artifacts"][matches[0]])


hold_frames = HOLD_STEPS // DUMP_EVERY
summaries: dict[str, dict[str, Any]] = {}
melt_volumes: list[float] = []
trajectories: list[Path] = []
for replica in range(1, REPLICAS + 1):
    result, info = run_lammps(_script(replica), atoms=STRUCTURE, label=f"{LABEL}-r{replica}")
    specorder = [info["types"][key] for key in sorted(info["types"], key=int)]
    tables = result["tables"]
    melt = tables[0]
    melt_volumes.append(float(melt["last"]["Volume"]))
    print(
        f"replica {replica}: melted {melt['loop']['atoms']} atoms at {T_MELT:g} K "
        f"and {PRESSURE_BAR:g} bar; V = {melt_volumes[-1]:.2f} A^3 "
        f"(LAMMPS {info['version']}, {result['steps']} steps in all)"
    )
    for index, rate in enumerate(QUENCH_RATES_K_PER_PS):
        ramp, hold = tables[1 + 2 * index], tables[2 + 2 * index]
        name = f"quench-{rate:g}Kps-r{replica}"
        frames = read(
            _kept_path(info, f"{name}.dump"),
            index=":",
            format="lammps-dump-text",
            specorder=specorder,
        )
        traj = Path(f"{name}.traj")
        write(traj, frames)
        trajectories.append(traj)
        summaries[f"{rate:g}-r{replica}"] = {
            "ramp_steps": ramp["loop"]["steps"],
            "T_end_of_ramp": float(ramp["last"]["Temp"]),
            "hold_T_mean": float(hold["tail"]["mean"]["Temp"]),
            "hold_V_mean": float(hold["tail"]["mean"]["Volume"]),
            "V_max": max(float(table["last"]["Volume"]) for table in (ramp, hold)),
            "frames": len(frames),
        }
        print(
            f"rate {rate:g} K/ps, replica {replica}: {ramp['loop']['steps']} steps to "
            f"{T_FINAL:g} K and {HOLD_STEPS} held, hold <V> = "
            f"{summaries[f'{rate:g}-r{replica}']['hold_V_mean']:.2f} A^3 ({traj})"
        )

with open("quench.json", "w", encoding="utf-8") as handle:
    summary = {
        "engine": "lammps",
        "potential": POTENTIAL,
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
    return all(row["hold_T_mean"] < T_MELT / 2.0 for row in summaries.values())


@check
def volumes_stay_physical() -> bool:
    largest = max(melt_volumes)
    return all(0.0 < row["V_max"] < 3.0 * largest for row in summaries.values())


@check
def every_hold_was_recorded() -> bool:
    return all(row["frames"] == hold_frames for row in summaries.values())
