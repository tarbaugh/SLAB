---
name: thermal-response
description: Heat capacity, thermal expansion, and latent heat from NPT
  temperature ladders, per atom and per axis, with slope errors and a
  hysteresis check. Use when asked for c_p, a CTE, an enthalpy-vs-
  temperature curve, or the heat of fusion or crystallization between two
  phases.
license: MIT
metadata:
  mason-agents: "md-expert analysis-expert"
---
# Thermal response

The procedure has two parts. A workflow walks one phase up a temperature
ladder under NPT and records the mean enthalpy, volume, and cell at each
rung with block errors. A bundled script fits the slopes, and compares
two phases' ladders for a latent heat.

## 1. Run a ladder per phase

Copy `assets/thermal_ramp.py` into the project directory and adapt the
constants at the top:

- `STRUCTURE` and `ENGINE`: one phase per ladder. For a latent heat you
  run the template twice, once from the crystal and once from the
  amorphous or liquid cell, with the same atom count and everything else
  identical. The amorphous branch depends on its quench rate and age
  (melt-quench skill); state both.
- `TEMPERATURES`: the rungs, walked in order as one continuous
  trajectory. Use five or more per window you will fit, spaced closely
  enough that H(T) is linear between neighbours.
- `WALK_DOWN`: set `True` to record the descent as well. The fit compares
  the two branches at shared temperatures, and a gap is hysteresis
  (superheating on the way up, supercooling down), which means the
  slopes across that range are not equilibrium values.
- `EQUILIBRATION_STEPS` and `AVERAGING_STEPS`: the template's values are
  shakeout lengths. Equilibrate for at least ten barostat time constants
  per rung, and average until the block error on H is small against the
  slope times the rung spacing.
- `ANISOTROPIC`: set `True` for hexagonal, tetragonal, or orthorhombic
  phases so each axis breathes on its own; the fit then reports an
  expansion coefficient per axis, which is the number to quote.
- Each rung is approached under Berendsen and sampled under ASE's
  isotropic Martyna–Tobias–Klein NPT, because Berendsen suppresses the
  volume fluctuations and samples no ensemble.

Run it with `launch_workflow` and give an intent. The ladder writes
`ramp.json`, the run keeps it, and the checks gate the run: one row per
rung, enthalpy rising with temperature on the way up, volumes physical.
Every row carries the atom count, the pressure, H = E + PV, the measured
temperature, the mean cell lengths, and block errors.

## 2. Fit

    python <skill root>/scripts/fit_thermal_ramp.py ramp.json --window 400 700
    python <skill root>/scripts/fit_thermal_ramp.py crystal.json --other liquid.json --at 823

The first form fits the measured temperatures and prints c_p = dH/dT per
atom in k_B, in J/(mol K) and J/(kg K), and per volume in J/(m^3 K), with
the slope's standard error from four or more rungs; the linear expansion
coefficient (dV/dT)/(3V) with its error, and per axis when the cell
lengths were recorded; and the hysteresis at shared temperatures when a
downward branch exists. The second prints the enthalpy difference
between the two ladders at one temperature, per atom (eV and kJ/mol),
per cell, and per volume, and refuses ladders with different atom
counts. `--json` gives the numbers machine-readable.

## 3. Verify and report

- Heed the linearity warnings. A kink inside the window is a phase
  change; fit each side separately instead of averaging across it.
- Classical MD gives about 3 k_B per atom for a solid, plus a modest
  excess from anharmonicity and c_p − c_v; 3.3 k_B is not wrong. Below
  roughly the Debye temperature the classical value overshoots the
  quantum one, which falls toward zero. Say which regime the number is
  from.
- A latent heat needs both ladders to bracket the comparison
  temperature; the script refuses to extrapolate. Sample the melting
  temperature from both sides.
- Report slopes with their errors, the window, the pressure, the cell
  size, the potential, the integrator, and the run ids of both ladders.

## When not to use this

- Harmonic (phonon-level) heat capacities at low temperature are a
  lattice-dynamics calculation, not an MD average; this skill's c_p is
  classical.
- A hysteretic first-order transition inside the ladder shifts apparent
  transition temperatures; get T_m from two-phase coexistence, not from
  where a one-phase ladder jumps.
