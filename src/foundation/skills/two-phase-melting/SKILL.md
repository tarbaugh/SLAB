---
name: two-phase-melting
description: Melting temperature and crystal-growth velocities from
  solid-liquid coexistence MD under the N P_z A T ensemble, with the
  interface tracked by a bond-order parameter. Use when asked for a
  melting point from simulation, an interface or growth velocity, growth
  anisotropy, or a v(T) table for kinetics.
license: MIT
metadata:
  mason-agents: "md-expert"
---
# Two-phase melting and growth

A coexistence cell puts crystal and melt in direct contact and lets the
interface vote: below T_m the crystal grows, above it the crystal
shrinks, and the velocity crosses zero at T_m. The bundled script turns
a coexistence trajectory into a crystalline fraction and an interface
velocity; the crossing fit lives in the kinetic-fits skill.

## 1. Build the coexistence cell

- Make the cell long along the growth direction (L x L x 5L, with ten
  or more unit cells across and ten thousand atoms or more for a
  reportable T_m) with the crystal slab spanning the cross-section.
- Melt one half by holding only those atoms at 1.3 T_m or above while
  the other half stays fixed, then relax the pair near the expected T_m
  so the two interfaces are clean.
- Set the lateral cell to the crystal's own a(T) from a crystal-only NPT
  run at the same temperature. The lateral dimensions stay fixed from
  then on.
- The periodic cell holds *two* interfaces; both move. The script
  divides by that count.
- For an anisotropic crystal, build one cell per growth direction; the
  anisotropy is the ratio of the fitted velocities, direction by
  direction. Expect (100) faster than (110) faster than (111) in fcc.

## 2. Run the temperature ladder

- Hold each copy of the cell at one temperature under N P_z A T: the
  barostat acts only along the interface normal, the lateral cell is
  fixed at a(T). An isotropic barostat strains the crystal as the liquid
  fraction changes and shifts T_m.
- Thermostat in layers. The moving interface releases latent heat, and
  a single global thermostat lets the interface run hotter than the set
  point, which changes the kinetic coefficient by up to a factor of two.
  Thermostat slabs of 3 to 4 nm independently, and report the measured
  temperature of the interface region next to the set point.
- Span both sides of the expected T_m, closely spaced near it, and run
  two or more replicas per temperature from different seeds.
- Stop before the two interfaces meet; the script warns when a phase
  was consumed.

Two other routes give T_m directly. Under NPH the released latent heat
drives the cell to T_m in one run. Interface pinning (a harmonic bias on
the crystalline order parameter) gives the chemical-potential difference
at each temperature and is the precise modern method.

## 3. Track the interface

    python <skill root>/scripts/interface_velocity.py coex-1300K-r1.traj coex-1300K-r2.traj --dt-fs 100

The script classifies each atom by the averaged bond-order parameter
q6-bar (about 0.5 in fcc, 0.15 in the liquid; `--threshold` 0.33) with
neighbours within `--cutoff` (default 1.2 times the shortest distance),
multiplies the crystalline fraction by the cell length along `--axis`,
fits the slope over the steady window, divides by `--interfaces`, and
prints the velocity in A/ps and m/s, growth positive, with the fit's
standard error or the spread over replicas. `--json` gives the table of
crystalline fraction against time as well. Record the threshold and the
cutoff with the velocity, because the fraction depends on them.

## 4. Fit and report

- Collect `{"T": kelvin, "value": velocity, "err": ...}` rows and use
  the kinetic-fits skill: `--mode crossing` fits v = k (T_m − T) over a
  window around the zero and gives T_m with an error and the kinetic
  coefficient k. Near T_m the law is linear; over wider undercooling
  the Wilson–Frenkel form v = v0 [1 − exp(−dG/kT)] applies, and VFT is
  a glass-former phenomenology, not a growth law.
- T_m from coexistence carries a finite-size shift; repeat at a larger
  cross-section once and state the change next to the number.
- Report T_m with the potential, the cell geometry, the direction, the
  ladder, the thermostat layout, and the run ids; report v(T) as the
  table plus the fitted law, never the law alone, with the order
  parameter that defined the crystalline fraction.

## When not to use this

- One-phase heating until the crystal collapses measures superheating,
  not T_m; the homogeneous limit sits at 1.2 to 1.3 T_m. Use
  coexistence.
- Below about 0.7 T_m the melt nucleates on its own and the interface
  velocity is no longer the only thing moving.
- Compounds that melt incongruently, or off-stoichiometric cells,
  couple melting to composition; the plain coexistence T_m applies to
  congruent melting at the cell's own stoichiometry, and the report
  must say so.
