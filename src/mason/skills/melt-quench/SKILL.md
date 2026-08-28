---
name: melt-quench
description: Make an amorphous structure by NPT melt-and-quench and measure
  its density against the crystal. Use when asked for a glass, an amorphous
  cell, an amorphous density, a quench-rate study, or the densification on
  crystallization.
license: MIT
metadata:
  mason-agents: "md-expert"
---
# Melt-quench

The procedure has two parts. A workflow melts the cell under NPT and
quenches copies of the one melt at a ladder of rates. A bundled script
turns the recorded trajectories into tail-averaged densities.

## 1. Run the quench ladder

Copy `assets/melt_quench.py` into the project directory and adapt the
constants at the top:

- `STRUCTURE` and `ENGINE`: the cell and the potential. The template runs
  as-is under `emt` as a shakeout; real glasses need an interatomic
  potential you trust for the liquid, and its provenance goes in the
  report.
- `T_MELT` and `MELT_STEPS`: hold well above melting until the crystal is
  gone. Confirm the melt with the radial-distribution skill (no sharp
  second-shell peaks) before trusting any quench from it.
- `QUENCH_RATES_K_PER_PS`: the ladder. The template's values are shakeout
  speeds; real amorphous densities want the slowest rates you can afford,
  and the rate always accompanies the number, because glass properties
  depend on it.
- The Berendsen barostat is isotropic. That is right for melts and
  glasses; do not use this template to cool a crystal through an
  anisotropic transition.

Run it with `launch_workflow` and give an intent. Each rate writes
`quench-<rate>Kps.traj`, the run keeps the trajectories and a
`quench.json` summary, and the checks gate the run: every quench reached
the final temperature, and no volume exploded.

## 2. Measure

    python <skill root>/scripts/quench_report.py quench-*.traj --rho-c 4.63

It prints one tail-averaged density per trajectory in g/cm^3, with the
spread over the tail; `--rho-c` (the crystal density from an
equation-of-state run) adds delta_v = 1 - rho_a/rho_c per rate, and
`--json` gives the numbers machine-readable.

## 3. Verify and report

- Confirm the glass is a glass: the mean-squared displacement at the
  final temperature must plateau (msd-diffusion skill), and the radial
  distribution must show liquid-like broad shells, not crystal peaks.
- Density must trend with rate; a slower quench packs a denser glass. A
  non-monotonic table means the runs are too short.
- Report rho_a per rate with the potential, the cell size, the pressure,
  and the run id, and quote delta_v against the stated crystal density.

## When not to use this

- Quench rates below ~10 K/ps are usually out of MD's reach at useful
  cell sizes; extrapolate the rate dependence and say so, instead of
  claiming an experimental-rate glass.
- Systems that crystallize during the quench (fast crystallizers at slow
  rates) leave you with a poly-crystal, not a glass; check the radial
  distribution before calling the product amorphous.
