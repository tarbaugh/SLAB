---
name: melt-quench
description: Make an amorphous structure by NPT melt-and-quench with an
  isothermal hold, and measure its density against the crystal with a
  block error, replicas, and the density-versus-log-rate law. Use when
  asked for a glass, an amorphous cell, an amorphous density, a
  quench-rate study, or the densification on crystallization.
license: MIT
metadata:
  mason-agents: "md-expert"
---
# Melt-quench

The procedure has two parts. A workflow melts the cell under NPT,
quenches copies of the melt at a ladder of rates, and holds each at the
final temperature. A bundled script turns the recorded holds into
densities with errors and the rate law.

## 1. Run the quench ladder

Copy `assets/melt_quench.py` into the project directory and adapt the
constants at the top:

- `STRUCTURE` and `ENGINE`: the cell and the potential. The template runs
  as-is under `emt` with 32 atoms as a shakeout; a reportable density
  wants 500 to 1000 atoms or more, and an interatomic potential you
  trust for the liquid, whose provenance goes in the report.
- `T_MELT` and `MELT_STEPS`: hold well above melting until the crystal is
  gone. Confirm the melt with the radial-distribution skill (no sharp
  second-shell peaks) and the msd-diffusion skill (D of order
  1e-5 cm^2/s) before trusting any quench from it.
- `QUENCH_RATES_K_PER_PS`: the ladder. The template's values are shakeout
  speeds. Real glasses are made at 0.01 to 10 K/ps (1e10 to 1e13 K/s);
  1800 to 300 K at 0.1 K/ps is 15 ns, routine with an EAM or MLIP. The
  rate always accompanies the number, because glass properties depend on
  it.
- `HOLD_STEPS`: the isothermal hold at `T_FINAL`. The density is
  measured over the hold, never over the end of the ramp, which spans a
  range of temperatures. Hold long past the barostat time constant, and
  longer when the report shows a drift.
- `REPLICAS`: independent melts per rate from different seeds. Two or
  more give the spread that tells a rate-to-rate difference of a percent
  from noise.
- `COMPRESSIBILITY_PER_BAR`: the material's isothermal compressibility;
  it sets the Berendsen barostat's effective time constant.
- The melt and the ramp run under Berendsen, which reaches a target
  quickly and samples no ensemble; the hold runs under ASE's isotropic
  Martyna–Tobias–Klein NPT, which does. The barostat is isotropic. That
  is right for melts and glasses; do not use this template to cool a
  crystal through an anisotropic transition.

Run it with `launch_workflow` and give an intent. Each rate and replica
writes `quench-<rate>Kps-r<k>.traj`, the run keeps the trajectories and
a `quench.json` summary with `hold_frames`, and the checks gate the run:
every quench reached the final temperature, no volume exploded, every
hold was recorded.

## 2. Measure

    python <skill root>/scripts/quench_report.py quench-*.traj --hold-frames 10 --rho-c 4.63

It prints one hold-averaged density per trajectory in g/cm^3 with a block
standard error and the drift across the hold (a drift above three
standard errors is a warning: hold longer), groups the files by rate with
the replica spread, fits rho = rho_0 + a log10(rate) across the rates,
and with `--rho-c` (the crystal density from an equation-of-state or NPT
run at the same temperature and pressure) adds delta_v = 1 - rho_a/rho_c
per rate. `--json` gives the numbers machine-readable. Pass
`--hold-frames` from `quench.json`; without it the script averages a
tail fraction and says the number is not a hold.

## 3. Verify and report

- Confirm the glass is a glass: the mean-squared displacement at the
  final temperature must plateau (msd-diffusion skill), and the radial
  distribution must show liquid-like broad shells, not crystal peaks.
- The sign of the rate dependence is the system's, not a rule. Metals
  and Lennard-Jones glasses pack denser at slower rates; silica loosens.
  A non-monotonic table with replica spreads that overlap is noise, not
  a finding.
- Extrapolate to an experimental rate only with the log-rate law and the
  fitted range stated; never call the result an experimental glass.
- Report rho_a per rate with the spread, the hold length, the potential,
  the cell size, the pressure, and the run id, and quote delta_v against
  the stated crystal density at the same temperature.

## When not to use this

- Quench rates below about 0.01 K/ps are out of MD's reach at useful
  cell sizes; extrapolate the rate dependence and say so.
- Systems that crystallize during the quench (fast crystallizers at slow
  rates) leave you with a poly-crystal, not a glass; check the radial
  distribution before calling the product amorphous.
- A glass ages. A density measured right after the quench differs from
  one measured after an anneal; state which.
