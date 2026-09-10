"""NVT molecular dynamics inside LAMMPS, run whole through ``run_lammps``.

Runs as-is as a shakeout: 108 argon atoms under a Lennard-Jones potential
that needs no potential file, 2000 steps at 300 K. For real work change
STRUCTURE, the POTENTIAL lines (the lammps-potentials skill gives them),
the temperature, and the run length, and keep the checks: a run is
verified only when its dynamics held.
"""

from ase.build import bulk

from foundation import check
from foundation.tasks import run_lammps

STRUCTURE = bulk("Ar", "fcc", a=5.26, cubic=True) * (3, 3, 3)
TEMPERATURE_K = 300.0
TIMESTEP_PS = 0.002  # 2 fs; halve it for a lighter element or a stiffer potential
STEPS = 2000
THERMO_EVERY = 100
DUMP_EVERY = 500
SEED = 4928459  # seeds the initial velocities, so a rerun reproduces the run
LABEL = "ar-nvt"

# Argon: epsilon 0.0104 eV, sigma 3.40 Å, in metal units. The structure's
# masses ride in structure.data, which run_lammps writes from STRUCTURE.
POTENTIAL = """\
pair_style lj/cut 8.5
pair_coeff 1 1 0.0104 3.40
pair_modify shift yes
"""

SCRIPT = f"""\
units metal
atom_style atomic
boundary p p p
read_data structure.data

{POTENTIAL}
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

velocity all create {TEMPERATURE_K} {SEED} mom yes rot yes dist gaussian
timestep {TIMESTEP_PS}
fix integrate all nvt temp {TEMPERATURE_K} {TEMPERATURE_K} {100 * TIMESTEP_PS}

thermo {THERMO_EVERY}
thermo_style custom step temp pe ke etotal press vol
thermo_modify flush yes
dump traj all custom {DUMP_EVERY} {LABEL}.dump id type x y z vx vy vz
dump_modify traj sort id

run {STEPS}
write_data {LABEL}-final.data
"""

result, info = run_lammps(SCRIPT, atoms=STRUCTURE, label=LABEL)
table = result["tables"][-1]
final = result["thermo"]
tail = table["tail"]
print(
    f"LAMMPS {info['version']}: {result['steps']} steps, "
    f"{table['loop']['atoms']} atoms, {table['loop']['seconds']:.2f} s"
)
print(
    f"tail of {tail['rows']} rows: T={tail['mean']['Temp']:.1f} K "
    f"(std {tail['std']['Temp']:.1f}), PotEng={tail['mean']['PotEng']:.3f} eV"
)
print(f"final row: step {final['Step']}, T={final['Temp']:.1f} K, Press={final['Press']:.0f} bar")
print(f"artifacts: {sorted(info['artifacts'])}")


@check
def the_run_finished_every_step() -> None:
    assert result["wall_time"] is not None, "no Total wall time line: LAMMPS did not finish"
    assert result["steps"] == STEPS, f"{result['steps']} steps ran, not {STEPS}"
    assert table["loop"]["atoms"] == len(STRUCTURE), "atoms were lost"


@check
def the_thermostat_held_over_the_tail() -> None:
    mean = tail["mean"]["Temp"]
    assert abs(mean - TEMPERATURE_K) < 0.1 * TEMPERATURE_K, (
        f"tail mean {mean:.1f} K is not within 10 % of {TEMPERATURE_K} K"
    )
