---
name: thermal-response
description: Heat capacity, thermal expansion, and latent heat from NPT
  temperature ladders. Use when asked for c_p, a CTE, an enthalpy-vs-
  temperature curve, or the heat of fusion or crystallization between two
  phases.
license: MIT
metadata:
  mason-agents: "md-expert analysis-expert"
---
# Thermal response

The procedure has two parts. A workflow walks one phase up a temperature
ladder under NPT and records the mean total energy and volume at each
rung. A bundled script fits the slopes, and compares two phases' ladders
for a latent heat.

## 1. Run a ladder per phase

Copy `assets/thermal_ramp.py` into the project directory and adapt the
constants at the top:

- `STRUCTURE` and `ENGINE`: one phase per ladder. For a latent heat you
  run the template twice, once from the crystal and once from the
  amorphous or liquid cell, with everything else identical.
- `TEMPERATURES`: the rungs, walked in order as one continuous
  trajectory. Space them closely enough that E(T) is linear between
  neighbors.
- `EQUILIBRATION_STEPS` and `AVERAGING_STEPS`: the template's values are
  shakeout lengths. Real averages want equilibration long past the
  thermostat and barostat time constants, and averaging windows that make
  the run-to-run scatter small against the slope you are fitting.

Run it with `launch_workflow` and give an intent. The ladder writes
`ramp.json`, the run keeps it, and the checks gate the run: one row per
rung, energy rising with temperature, volumes physical.

## 2. Fit

    python <skill root>/scripts/fit_thermal_ramp.py ramp.json --window 400 700
    python <skill root>/scripts/fit_thermal_ramp.py crystal.json --other liquid.json --at 823

The first form prints the volumetric heat capacity (dE/dT)/V in
J/(m^3 K) and the linear expansion coefficient (dV/dT)/(3V) in 1/K over
the window. The second prints the enthalpy difference between the two
ladders at one temperature, in eV per cell and J/m^3. `--json` gives the
numbers machine-readable.

## 3. Verify and report

- Heed the linearity warnings. A kink inside the window is a phase
  change; fit each side separately instead of averaging across it.
- Classical MD misses quantum freezing of phonons: below roughly the
  Debye temperature the classical c_p overshoots the true value toward
  3 kB per atom. Say which regime the number is from.
- A latent heat needs both ladders to bracket the comparison temperature;
  the script refuses to extrapolate. Sample the melting temperature from
  both sides.
- Report slopes with the window, the pressure, the cell size, the
  potential, and the run ids of both ladders.

## When not to use this

- Harmonic (phonon-level) heat capacities at low temperature are a
  lattice-dynamics calculation, not an MD average; this skill's c_p is
  classical.
- A hysteretic first-order transition inside the ladder (superheating,
  supercooling) shifts apparent transition temperatures; get T_m from
  two-phase coexistence, not from where a one-phase ladder jumps.
