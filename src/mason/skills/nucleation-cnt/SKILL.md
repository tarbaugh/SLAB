---
name: nucleation-cnt
description: Classical nucleation theory arithmetic - interfacial energy
  from seeded-crystallite brackets, and critical radii and barriers from
  it. Use when asked for gamma from grow/shrink seeds, a critical nucleus
  size, a nucleation barrier, or to assemble CNT inputs.
license: MIT
metadata:
  mason-agents: "md-expert analysis-expert"
---
# Nucleation and CNT

Classical nucleation theory links four numbers: the melting temperature
T_m, the volumetric latent heat dH_f, the crystal-liquid interfacial free
energy gamma, and the driving force dG_v = dH_f (T_m - T)/T_m. The
bundled script runs the algebra in both directions; the simulations that
feed it are separate runs.

## 1. Measure gamma by seeded brackets

- Embed a spherical crystallite of radius r in the equilibrated melt,
  anneal at an undercooled temperature, and watch it grow or shrink
  (potential energy drifting down means growth; the radial-distribution
  and msd-diffusion skills tell melt from crystal).
- Bracket: at each temperature, find a radius that grows and a radius
  that shrinks; the critical radius r*(T) lies between. Record the
  bracket, not just the midpoint.
- Repeat at two or three undercoolings. Then:

      python <skill root>/scripts/cnt.py gamma rstar.json --tm 823 --dhf 2.4e8

  with `rstar.json` a JSON list of `{"T": kelvin, "r_star_nm": ...}`
  rows. Each row yields gamma = r* dH_f (T_m - T)/(2 T_m); the script
  prints the per-point values, the mean, and the spread, and warns when
  gamma trends with temperature.

## 2. Run the algebra forward

    python <skill root>/scripts/cnt.py barrier --gamma 0.062 --tm 823 \
        --dhf 2.4e8 --temps 500 600 700 --f-het 0.32 --omega 30.5

At each temperature it prints r*, the barrier dG* = 16 pi gamma^3 /
(3 dG_v^2) in eV and in units of kT, and (with `--omega`, the atomic
volume in A^3) the atom count of the critical nucleus. `--f-het` (from
the interface-adhesion skill) scales the barrier for heterogeneous
nucleation. `--json` on either mode gives machine-readable output.

## 3. Verify and report

- T_m and dH_f must come from the same potential as the seeding runs
  (two-phase-melting and thermal-response skills); mixing sources breaks
  the internal consistency CNT depends on.
- A critical nucleus of tens of atoms or fewer is outside CNT's sharp-
  interface assumption; report the size next to gamma and say when the
  theory is strained.
- Barriers of a few kT mean nucleation is effectively barrierless at
  that temperature; barriers of hundreds of kT mean it never happens in
  simulation. Sanity-check the regime against what the runs showed.
- Report gamma with the bracket widths, the temperatures, the potential,
  and the run ids of the seeding runs.

## When not to use this

- Two-step or diffuse-interface nucleation (dense-liquid precursors,
  strongly faceted nuclei) breaks the spherical-cap picture; the
  arithmetic still runs, and its answer is the wrong model.
- gamma from this route is the *kinetic* crystal-melt gamma at
  undercooling; do not substitute a T = 0 surface energy for it.
