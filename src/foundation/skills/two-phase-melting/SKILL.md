---
name: two-phase-melting
description: Melting temperature and crystal-growth velocities from
  solid-liquid coexistence MD, either as one NPH run whose temperature
  plateaus at T_m or as an N P_z A T interface-velocity ladder, with the
  interface tracked by an order parameter that survives a hot crystal.
  Use when asked for a melting point from simulation, an interface or
  growth velocity, growth anisotropy, or a v(T) table for kinetics.
license: MIT
metadata:
  mason-agents: "md-expert"
---
# Two-phase melting and growth

A coexistence cell puts crystal and melt in direct contact and lets the
interface vote. Below T_m the crystal grows, above it the crystal
shrinks. Two routes read that vote. Under NPH the latent heat drives the
cell to T_m and the temperature plateaus there, so one run gives T_m.
Under N P_z A T at a set temperature the interface moves at a velocity
that crosses zero at T_m, so a ladder of temperatures gives T_m and the
growth kinetics. The bundled scripts cover both.

## 1. Choose the route

| Route | Cell | Runs | Steps per run | Gives |
|---|---|---|---|---|
| NPH plateau | under ten thousand atoms | 3, three starting enthalpies with one run each | 100 000 (200 ps) after 30 ps of preparation | T_m alone |
| N P_z A T velocity ladder | ten thousand atoms or more | 12, two seeds at each of 6 rungs | 100 000 (200 ps) per rung | T_m and v(T) |

Run the NPH plateau route first for a cell under ten thousand atoms. It
is three starting enthalpies, one run each, and the temperature each
run settles on is T_m. The velocity ladder costs four times as many
runs of the same length and buys the kinetic coefficient with it, so it
is the route for a large cell and for a v(T) table.

Size the wave from the rate the smoke test itself reports. The log's
`Performance` line gives Matom-step/s, so the wall time of a run is its
atoms times its steps divided by that rate. One core under a plain pair
potential does 2 to 3 Matom-step/s with the order parameter read every
100 steps. The bundled NPH stage sampled the order parameter every 10
steps and ran at 0.93 Matom-step/s, so 5120 atoms for 100 000 steps
took about 9 minutes on one core. The order-parameter computes cost two
to three times, and the three-run NPH route on a small cell still fits
a short job. A machine-learned potential is two orders of magnitude
slower per atom-step, and the same route then needs the GPU slice and
the hours that go with it. State the rate you measured next to the
plan.

## 2. What a hot crystal reads

A perfect crystal near T_m does not read as a crystal under an
instantaneous order parameter. Thermal displacements break the
neighbour signature, so `compute cna/atom` on one snapshot classifies a
large share of the atoms as unknown. How large depends on the potential,
the temperature, and the cutoff, and it is not predictable. The
Lennard-Jones crystal of the bundled cell reads 0.53 fcc a few kelvin
below its own T_m, at the cutoff taken from its own a(T). Its range
over 10 ps is 0.44 to 0.61, and its liquid reads zero. One copper
campaign read 0.31 down to 0.13 fcc between 1250 and 1450 K, the range
that brackets copper's T_m of 1358 K. Both are intact crystals. A gate
that compares either number to a cold count, or to one, fails on a
crystal that is working. That campaign diagnosed a failed freeze twice
when the freeze had worked. Measure the number, never assume it.

Prescribe one of two classifications and record which.

- Time-average the positions, then classify. `compute u all
  property/atom xu yu zu` and `fix pos all ave/atom 10 100 1000 c_u[1]
  c_u[2] c_u[3]` average the unwrapped positions over 1 to 2 ps, which
  is long enough to wash out the thermal displacement and short enough
  that the interface moves less than one lattice spacing. Dump the
  averaged positions with `dump avg all custom 1000 avg.dump id type
  f_pos[1] f_pos[2] f_pos[3]` and run `compute cna/atom` over the dump
  with `rerun avg.dump dump x y z box yes label x f_pos[1] label y
  f_pos[2] label z f_pos[3] wrapped no`. This is the recommended route.
  Averaging the per-atom CNA label instead does not remove the deficit,
  because the label is already wrong in each frame.
- Use `compute ptm/atom`, which fits a polyhedral template and tolerates
  thermal noise directly. Set its RMSD cutoff to 0.13 to 0.18 for a hot
  crystal; a tighter cutoff throws away good atoms and reintroduces the
  deficit. PTM needs the PTM package, so check the build's package list
  first.

Where neither is available, count solid-like bonds instead of
classifying atoms. `compute q6 all orientorder/atom components 6 nnn
NULL cutoff <first shell>`, `compute conn all coord/atom orientorder q6
0.5`, and `variable iscrystal atom "c_conn >= 6"` mark an atom whose q6
vector aligns with six or more neighbours. The count tolerates thermal
noise for the same reason PTM does, and the bundled runs use it.

Calibrate the classification you chose at every temperature you report.
Run the crystal-only cell and the liquid-only cell at that temperature
through the same classification, the same cutoff, and the same
averaging, and record the two numbers as the crystal baseline and the
liquid baseline. The calibration runs are short, they come first, and
their run ids travel with the result. The classification decides the
baseline, so the two travel together. The crystal of the bundled system
reads 1.00 under the connectivity count at 1445 K and 0.53 under
instantaneous CNA at the same temperature, and its liquid reads 0.045
and zero. One number is not comparable with the other.

Restate every gate against those baselines. The calibrated fraction is
(f − f_liquid) / (f_crystal − f_liquid), where f is the measured
fraction, and it is the only number a gate may test.

- Both phases present: the calibrated fraction sits between 0.05 and
  0.95. Outside that, a phase was consumed and the cell is no longer
  coexistence.
- The freeze worked: the frozen group's own fraction equals the crystal
  baseline at that temperature, within the baseline's own error. It does
  not equal one.
- The melt worked: the melted group's fraction equals the liquid
  baseline. A frozen half that reads 0.2 fcc on a raw count is not
  evidence of anything.
- `fix halt` thresholds are calibrated fractions, one guard per stage,
  never absolute fractions.

Set the CNA or the neighbour cutoff for the run temperature, not for
0 K. The cutoff is the midpoint of a(T)/sqrt(2) and a(T), with a(T)
from the crystal-only NPT run at that temperature, and you restate it at
every rung of the ladder. A cutoff left at the 0 K lattice constant adds
its own deficit on top of the thermal one.

## 3. Build the coexistence cell

- Make the cell long along the growth direction (L x L x 5L) with the
  crystal slab spanning the cross-section.
- Set the lateral cell to the crystal's own a(T) from a crystal-only NPT
  run at the same temperature. The lateral dimensions stay fixed from
  then on.
- The periodic cell holds *two* interfaces; both move. The velocity
  script divides by that count.
- For an anisotropic crystal, build one cell per growth direction; the
  anisotropy is the ratio of the fitted velocities, direction by
  direction. Expect (100) faster than (110) faster than (111) in fcc.

### The cross-section bound

A reportable T_m needs eight or more unit cells across the cross
section, and ten or more for a velocity. Below eight the interface
cannot roughen across the cell, and T_m carries a finite-size shift of
tens of kelvin. `coexistence_fraction.py` refuses a cross section under
eight cells. `--small-cell` accepts it and prints the caveat line, and
the report then carries that line next to the number. Repeat the
measurement at a larger cross-section once and state the change.

### Small cells and GPU builds

Assemble the two phases from their own equilibrated legs, not from one
cold crystal:

1. Equilibrate the crystal at the target temperature under NPT and take
   its a(T).
2. Equilibrate the liquid at the same temperature and the crystal's
   lateral cell, from a melt at 1.3 T_m or above brought back down.
3. Stack the two at the lateral cell of the crystal, with no gap at
   either join. A cell built with a gap at the periodic wrap opens a
   free liquid surface, and the run vaporises the liquid instead of
   melting the crystal.
4. Check the minimum interatomic distance with the atomsk-structures
   skill's `check_structure.py`. A merged interface has close contacts,
   so push off under a soft repulsion first, as the lammps-scripting
   skill gives, then minimise under the real potential with the crystal
   frozen, then check again.
5. Check the crystal's own calibrated fraction against its baseline
   before any dynamics. A crystal that is already damaged at the
   interface does not recover under the melt.

The barostat needs care on a two-phase cell.

- Write `dilate all`, not `dilate <group>`. A group dilate crashes under
  a KOKKOS build. `dilate all` remaps the frozen atoms with the box,
  which changes their positions but not their order, so the frozen
  crystal stays a crystal and the lateral cell stays the crystal's a(T).
- The barostat-free alternative is to pre-size the box to the two-phase
  volume from the pure legs and melt under NVT. It holds the volume
  fixed, so state the pressure the cell reached instead of setting it.
- Never barostat z to zero pressure on a cell with a free liquid
  surface. The liquid expands into the vacuum and the cell vaporises.
  Close the cell first.

## 4. Run the NPH plateau route

- Prepare the pair as section 3 says, then hold it at one temperature
  near the expected T_m under N P_z A T for 15 to 30 ps. This sets the
  enthalpy, and the fractions shift while it runs. The bundled run held
  15 ps.
- Release the thermostat. `fix nph all nph z 0.0 0.0 <Pdamp>` integrates
  with no temperature control, so the latent heat carries the cell to
  T_m and holds it there. Run 200 ps.
- Print thermo often enough for two windows of statistics, 100 steps or
  less, and write it as YAML with `thermo_modify line yaml`, because
  `run_lammps` reads the YAML documents. Write the fraction series with
  its own `fix ave/time`.
- Read the plateau with `coexistence_fraction.py --plateau`. It gives
  the mean temperature over the primary window and over each of its two
  disjoint halves, each with its block standard error. It also gives
  the drift across the primary window with the error of the fitted
  slope, and the verdict. The verdict is a plateau only when the drift stays inside
  three block errors, the two halves agree, and the calibrated fraction
  stays off both baselines. The halves agree when their gap is within
  two combined block errors, and the script prints the rule next to the
  gap.
- Read the block error before you read the drift. The script repeats
  the error over 16, 8, and 4 blocks, each level twice as long as the
  one before, and prints the series. The error is converged only when
  the last ratio is under 1.2. When the error still grows, the series
  correlates over the block. The error is then a lower bound, the drift
  in errors is an upper bound, and the run is too short for either to
  settle the question. The slope error assumes independent residuals,
  so the script scales it by the ratio of the residuals' block error to
  their naive error and prints the factor on the drift line.
- Repeat at three starting enthalpies, one run each, set by three
  preparation temperatures 30 K apart. T_m is the mean of the three
  plateaus, and its error is their spread, not one run's block error.
  Three plateaus that disagree by more than their block errors say the
  cell has not equilibrated, not that T_m is uncertain.
- A cell that drifts through the whole run has the wrong enthalpy. Reset
  the enthalpy at the plateau temperature the run reached and release
  again.

The bundled 70 ps log is a run on a 5120-atom Lennard-Jones coexistence
cell eight unit cells across, and it predates the recipe of section 3.
It melted the upper half at 1900 K under NVT for 15 ps with the lower
half frozen, then held the whole cell under N P_z A T at 1450 K for
15 ps, then released it to NPH for 70 ps. It has no separate legs, no
minimisation, and no close-contacts check, so it serves the plateau
analysis and not the build. The baselines were measured on that cell's
own pure phases at the same temperature:

    python <skill root>/scripts/coexistence_fraction.py \
        lammps-lj-coex-nph-70ps-yaml.log --cells 8 --plateau --timestep-fs 2 \
        --fraction lammps-lj-coex-70ps-fraction.dat --natoms 5120 \
        --crystal-baseline 1.00 --liquid-baseline 0.045

```text
lammps-lj-coex-nph-70ps-yaml.log: 3 thermo table(s), 351 rows in the one read; cross section 8.0 unit cells
primary        1442.58 +/- 3.48 (176 rows, 35.0 to 70.0 ps)
first half     1451.66 +/- 2.33 (88 rows, 35.0 to 52.4 ps)
second half    1433.50 +/- 2.91 (88 rows, 52.6 to 70.0 ps)
halves       gap 18.16 against a combined error of 3.73; they agree within 2 combined errors: False
block error  over 16, 8, 4 blocks: 2.97, 4.21, 5.85 (last ratio 1.39, converged under 1.2: False)
drift           -29.07 +/- 8.06 across the primary window (-8.3 block errors; the slope error is scaled 2.0x for correlated residuals)
fraction          0.57 calibrated over the window (0.53 to 0.61, baselines 0.045 and 1.000)
verdict: not a plateau; window mean = 1442.6 +/- 9.1
  because the temperature drifts -29.1 over the window, 8.3 block errors
  because the two halves of the window differ by 18.2 against a combined error of 3.7, more than 2 combined errors
warning: the block error has not converged: over 16, 8, 4 blocks it reads 2.97, 4.21, 5.85, and the last ratio is 1.39, not under 1.2. The series correlates over the block, so the error is a lower bound and the drift in errors is an upper bound. Lengthen the run, and settle it against the other enthalpies' plateaus
```

Read that verdict as it stands. Both phases survived the whole window,
so the cell is in coexistence near 1440 K. The temperature still
wanders 30 K over 35 ps, because the latent-heat reservoir of 5120
atoms is small. The two halves of the window read 1451.7 and 1433.5 K,
nearly five combined errors apart, so they do not agree. The block
error still grows at four blocks of 9 ps, so the drift of 8.3 errors is
an upper bound on how bad the drift is.

The same run continued to 200 ps is bundled beside it:

```text
lammps-lj-coex-nph-200ps-yaml.log: 3 thermo table(s), 1001 rows in the one read; cross section 8.0 unit cells
primary        1440.39 +/- 3.35 (500 rows, 100.2 to 200.0 ps)
first half     1434.92 +/- 3.53 (250 rows, 100.2 to 150.0 ps)
second half    1445.86 +/- 2.74 (250 rows, 150.2 to 200.0 ps)
halves       gap 10.94 against a combined error of 4.47; they agree within 2 combined errors: False
block error  over 16, 8, 4 blocks: 2.80, 3.90, 4.88 (last ratio 1.25, converged under 1.2: False)
drift           +20.16 +/- 9.40 across the primary window (+6.0 block errors; the slope error is scaled 3.6x for correlated residuals)
fraction          0.58 calibrated over the window (0.52 to 0.62, baselines 0.045 and 1.000)
verdict: not a plateau; window mean = 1440.4 +/- 5.5
  because the temperature drifts +20.2 over the window, 6.0 block errors
  because the two halves of the window differ by 10.9 against a combined error of 4.5, more than 2 combined errors
warning: the block error has not converged: over 16, 8, 4 blocks it reads 2.80, 3.90, 4.88, and the last ratio is 1.25, not under 1.2. The series correlates over the block, so the error is a lower bound and the drift in errors is an upper bound. Lengthen the run, and settle it against the other enthalpies' plateaus
```

The drift changed sign, and the block error still grows at four blocks
of 25 ps. The window mean is 1440.4 +/- 3.4 K against the shorter
window's 1442.6 +/- 3.5 K, and the two agree. The halves of each window
do not. The cell wanders on the timescale of the run rather than
trending, and neither window of this run passes the drift gate or the
halves gate. T_m comes from the agreement of the three enthalpies'
means, and the reported error is their spread. A wider gate is not the
fix. More atoms are, because the wander is a finite-size fluctuation,
and a cell of ten thousand atoms or more settles where this one cannot.
Do not report a plateau the script refused, and do not narrow the
window until it passes.

Interface pinning, a harmonic bias on the crystalline order parameter,
gives the chemical-potential difference at each temperature and is the
precise modern method. It is the route when the two above disagree.

## 5. Run the velocity ladder

- Hold each copy of the cell at one temperature under N P_z A T: the
  barostat acts only along the interface normal, the lateral cell is
  fixed at a(T). An isotropic barostat strains the crystal as the liquid
  fraction changes and shifts T_m.
- Take a(T) at each rung's own temperature, from a crystal-only NPT run
  there. A lateral cell set at another temperature strains the crystal
  and shifts the velocity at that rung.
- Write the barostat as `fix baro all nph z P P Pdamp` in production,
  where every atom moves. During the melt, where one half is frozen,
  integrate the mobile group and still remap the whole box: `fix baro
  mobile nph z P P Pdamp dilate all`. Never `dilate mobile`, per section
  3. `fix nph` integrates its group, so the layer thermostats must not
  integrate again. Use `fix langevin` or `fix temp/csvr` per layer,
  never `fix nvt`. Add `fix_modify baro temp tmove`, with `tmove` from
  the lammps-scripting skill, section 5, so the kinetic part of the
  pressure counts the moving atoms only. Give `Pdamp` in the time unit
  of the unit system. Under `units metal` 1 ps is `1.0`, not `1000`. A
  `Pdamp` of 1000 there is 1000 ps, longer than the run, so a starting
  pressure of twenty kilobar never relaxes.
- Keep the barostat on during the melt and during the equilibration.
  The melt expands, so a cell held at the crystal's volume compresses
  the liquid half to tens of kilobar.
- Give each stage (melt, equilibration, production) its own `fix halt`,
  and `unfix` it before the next stage starts. A guard written for one
  stage fires in another, because the melt and the equilibration pass
  through states the production never sees. Set each threshold as a
  calibrated fraction, per section 2.
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

## 6. Track the interface

Name the per-atom mask of section 2 `v_iscrystal`, whichever of the
three classifications produced it, so one name carries the concept
through the script and the report. Inside LAMMPS, `compute ncrystal all
reduce sum v_iscrystal` counts the marked atoms and `fix frac all
ave/time 10 10 100 c_ncrystal file fraction.dat` writes the series. The
file's rows come back through `series(result, "fraction.dat")` from
`foundation.tasks`, keyed by column, for the slope fit. `result["averages"]` holds only its
summary, and reading `rows` from that summary raises a `KeyError`. Never
parse the `.dat` file yourself. `coexistence_fraction.py --fraction`
reads the same file from disk when you want the calibrated series
without a plateau fit.

Report the whole-cell fraction against the calibrated baselines. The
crystal length along the axis is the cell length times the calibrated
fraction. The velocity is the slope of that length, so it follows the
phase length and not the raw count.

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

## 7. Fit and report

- Collect `{"T": kelvin, "value": velocity, "err": ...}` rows and use
  the kinetic-fits skill: `--mode crossing` fits v = k (T_m − T) over a
  window around the zero and gives T_m with an error and the kinetic
  coefficient k. Near T_m the law is linear; over wider undercooling
  the Wilson–Frenkel form v = v0 [1 − exp(−dG/kT)] applies, and VFT is
  a glass-former phenomenology, not a growth law.
- T_m from coexistence carries a finite-size shift; repeat at a larger
  cross-section once and state the change next to the number.
- Report T_m with the potential, the cell geometry, the direction, the
  route, the thermostat layout, and the run ids. Report v(T) as the
  table plus the fitted law, never the law alone. Name the order
  parameter, its cutoff, its averaging, and both baselines with their
  run ids, because the fraction means nothing without them.

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
