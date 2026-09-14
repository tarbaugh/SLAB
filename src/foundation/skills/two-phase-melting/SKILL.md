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
- Take a(T) at each rung's own temperature, from a crystal-only NPT run
  there. A lateral cell set at another temperature strains the crystal
  and shifts the velocity at that rung.
- Write the barostat as `fix baro mobile nph z P P Pdamp dilate mobile`.
  `fix nph` integrates its group, so the layer thermostats must not
  integrate again. Use `fix langevin` or `fix temp/csvr` per layer,
  never `fix nvt`. During the melt `mobile` is the liquid half, so the
  frozen crystal is neither integrated nor remapped. Add `fix_modify
  baro temp tmove`, with `tmove` from the lammps-scripting skill,
  section 5, so the kinetic part of the pressure counts the moving
  atoms only. Give `Pdamp` in the time unit of the unit system. Under
  `units metal` 1 ps is `1.0`, not `1000`. A `Pdamp` of 1000 there is
  1000 ps, longer than the run, so a starting pressure of twenty
  kilobar never relaxes.
- Keep the barostat on during the melt and during the equilibration.
  The melt expands, so a cell held at the crystal's volume compresses
  the liquid half to tens of kilobar.
- Give each stage (melt, equilibration, production) its own `fix halt`,
  and `unfix` it before the next stage starts. A guard written for one
  stage fires in another, because the melt and the equilibration pass
  through states the production never sees. Set each threshold relative
  to the calibrated baselines of section 3, never as an absolute
  fraction.
- Give each stage its own `fix ave/time` file name. A second fix that
  writes the same name truncates the first file, and the earlier
  stage's series is lost.
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

Two routes give the crystalline fraction against time. Inside LAMMPS,
`compute cna all cna/atom <cutoff>` classifies each atom (1 is fcc, 2
is hcp, 5 is unknown; the cutoff sits between the first and second
neighbour shells), `variable isfcc atom "c_cna == 1"` marks the
crystalline ones, `compute nfcc all reduce sum v_isfcc` counts them,
and `fix frac all ave/time 10 10 100 c_nfcc file fraction.dat` writes
the series. The file comes back parsed as
`result["averages"]["fraction.dat"]`, and `series(result,
"fraction.dat")` from `foundation.tasks` gives its rows for the slope
fit, keyed by column. Never parse the `.dat` file yourself.

Set the CNA cutoff for the run temperature, not for 0 K. The cutoff is
the midpoint of a(T)/sqrt(2) and a(T), with a(T) from the crystal-only
NPT run at that temperature, and you restate it at every rung of the
ladder. Even at the right cutoff, a hot perfect crystal under
instantaneous CNA reads well below 1, because thermal displacements
break the neighbour signature. A cutoff left at the 0 K lattice
constant makes this worse, and one campaign read a perfect hot crystal
as 0.08 to 0.2 fcc. Two cures exist:

- Classify time-averaged positions. Average the unwrapped positions
  over 10 to 20 steps (`compute u all property/atom xu yu zu` and `fix
  pos all ave/atom 1 20 20 c_u[1] c_u[2] c_u[3]`), dump them with
  `dump avg all custom 20 avg.dump id type f_pos[1] f_pos[2] f_pos[3]`,
  and run `compute cna/atom` on the dump with `rerun ... dump x y z box yes
  label x f_pos[1] label y f_pos[2] label z f_pos[3] wrapped no`.
  Averaging the per-atom CNA label instead does not remove the deficit.
- Use `compute ptm/atom`, which tolerates thermal noise. It needs the
  PTM package, so check the build's package list first.

Calibrate before you set any gate. Run the crystal-only cell and
the liquid-only cell at the rung's temperature through the same
classification, and record the two fractions as the crystal baseline
and the liquid baseline. The calibration runs are short, and they come
first.

Report the whole-cell fraction against the calibrated baselines. The
crystal length along the axis is the cell length times (f − f_liquid) /
(f_crystal − f_liquid), where f is the measured fraction. The velocity
is the slope of that length, so it follows the phase length and not the
raw count.

The trajectory route is the alternative when the classification needs
the bond-order parameter:

    python <skill root>/scripts/interface_velocity.py coex-1300K-r1.traj coex-1300K-r2.traj --dt-fs 100

The script classifies each atom by the averaged bond-order parameter
q6-bar (about 0.5 in fcc, 0.15 in the liquid; `--threshold` 0.33) with
neighbours within `--cutoff` (default 1.2 times the shortest distance),
multiplies the crystalline fraction by the cell length along `--axis`,
fits the slope over the steady window, divides by `--interfaces`, and
prints the velocity in A/ps and m/s, growth positive, with the fit's
standard error or the spread over replicas. `--json` gives the table of
crystalline fraction against time as well. Record the threshold and the
cutoff with the velocity, because the fraction depends on them. The
script uses the raw fraction, so its velocity is the calibrated one
divided by (f_crystal − f_liquid) measured under the same threshold and
cutoff. State which of the two you report.

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
