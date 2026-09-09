---
name: nucleation-cnt
description: Classical nucleation theory arithmetic - interfacial energy
  from seeded-crystallite brackets as gamma(T), critical radii, barriers,
  and rates from it. Use when asked for gamma from grow/shrink seeds, a
  critical nucleus size, a nucleation barrier or rate, or to assemble CNT
  inputs.
license: MIT
metadata:
  mason-agents: "md-expert analysis-expert"
---
# Nucleation and CNT

Classical nucleation theory links four numbers: the melting temperature
T_m, the driving force dG_v (from the latent heat dH_f near T_m), the
crystal-liquid interfacial free energy gamma, and the critical nucleus.
The bundled script runs the algebra in both directions; the simulations
that feed it are separate runs.

## 1. Measure gamma by seeding

- Embed a spherical crystallite in the equilibrated melt: freeze the
  seed, relax the melt around it, then release everything under NPT at
  the run pressure and an undercooled temperature.
- Count the crystalline cluster with a stated order parameter (a
  bond-order threshold with its cutoff and mislabeling correction, or
  polyhedral template matching), not the potential energy. The
  critical size depends on that criterion, and so does gamma; record
  the criterion with the brackets under `"order_parameter"`.
- Bracket: at each temperature, find a cluster size that grows and one
  that shrinks in most of ten or more trajectories from different
  velocity seeds; the critical size N*(T) lies between, at 50% growth
  probability. Record the bracket (`n_low`, `n_high`), not just the
  midpoint.
- Repeat at three or more undercoolings within about 20% of T_m, where
  dG_v = dH_f (T_m - T)/T_m holds. Then:

      python <skill root>/scripts/cnt.py gamma seeds.json --tm 823 --dhf 2.4e8 --omega 30.5

  with `seeds.json` a JSON list of `{"T": kelvin, "n_star": atoms,
  "n_low": ..., "n_high": ...}` rows (`r_star_nm` rows work too). Each
  row yields gamma = (3 N* rho_s^2 dmu^3 / 32 pi)^(1/3); the script
  fits gamma(T) as a line, prints gamma at T_m with its slope and error,
  and warns when gamma falls toward T_m, when a nucleus is under fifty
  atoms, or when only one undercooling was run.

gamma from seeding rises toward T_m. That trend is the physics, not an
artifact; carry gamma(T) forward, never one mean.

## 2. Run the algebra forward

    python <skill root>/scripts/cnt.py barrier --gamma 0.062 --gamma-slope 1.5e-4 \
        --tm 823 --dhf 2.4e8 --temps 700 750 --theta 60 --omega 30.5
    python <skill root>/scripts/cnt.py rate --gamma 0.062 --tm 823 --dhf 2.4e8 \
        --temps 700 --omega 30.5 --attachment 2e11

At each temperature `barrier` evaluates gamma(T) from the fit, then
r*, the barrier dG* = 16 pi gamma^3 / (3 dG_v^2) in eV and in units of
kT, and (with `--omega`) the atom count of the critical nucleus.
`--theta` (the contact angle from the interface-adhesion skill) or
`--f-het` scales the barrier and the atom count for heterogeneous
nucleation, f = (2 + cos)(1 - cos)^2/4. `rate` adds the Zeldovich factor
Z and, with the attachment frequency f+ (from the cluster-size
fluctuations of near-critical seeds), the rate J = rho_l Z f+ exp(-dG*/kT).
`--dmu-table` replaces the linear driving force with dG_v(T) from a
Gibbs–Helmholtz integration of the two phases' enthalpies
(thermal-response skill); the script warns past 20% undercooling
without it. `--json` on any mode gives machine-readable output.

## 3. Verify and report

- T_m, dH_f, and the seeding runs must come from the same potential
  (two-phase-melting and thermal-response skills); mixing sources breaks
  the internal consistency CNT depends on.
- A critical nucleus of tens of atoms or fewer is outside CNT's sharp-
  interface assumption; report the size next to gamma and say when the
  theory is strained. The barrier goes as gamma^3, so a 15% error in
  gamma is a 50% error in the barrier; seeding rates carry three to five
  orders of magnitude of uncertainty even when done well.
- Barriers of a few kT mean nucleation is effectively barrierless at
  that temperature; barriers of hundreds of kT mean it never happens in
  simulation. Sanity-check the regime against what the runs showed.
- Report gamma(T) with the brackets, the order parameter, the
  temperatures, the number of trajectories per size, the potential, and
  the run ids of the seeding runs.

## When not to use this

- Two-step or diffuse-interface nucleation (dense-liquid precursors,
  strongly faceted nuclei) breaks the spherical-cap picture; the
  arithmetic still runs, and its answer is the wrong model.
- gamma from this route is the *kinetic* crystal-melt gamma at
  undercooling; do not substitute a T = 0 surface energy for it.
