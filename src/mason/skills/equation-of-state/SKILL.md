---
name: equation-of-state
description: Compute an energy-volume curve and fit a Birch-Murnaghan
  equation of state to get the equilibrium volume, energy, and bulk modulus
  of a crystal. Use when asked for a lattice constant, an equilibrium
  volume, a bulk modulus, an E(V) curve, or to validate an engine against a
  known material.
license: MIT
metadata:
  mason-agents: "dft-expert analysis-expert"
---
# Equation of state

The procedure has two parts. A workflow computes single-point energies on a
ladder of scaled cells. A bundled script fits the Birch-Murnaghan form to
the recorded points.

## 1. Compute the energy-volume curve

Copy `assets/eos_scan.py` into the project directory and adapt the three
constants at the top:

- `STRUCTURE`: the bulk cell. Use a relaxed structure, or at least a
  reasonable experimental lattice constant.
- `ENGINE`: the engine for the single points. The template runs as-is
  under `emt` as a shakeout; switch to the production engine for a real
  number.
- `SCALES`: linear scale factors applied to the cell (volume scales as the
  cube). Seven points from 0.94 to 1.06 bracket most equilibria; widen the
  range only if the fit reports the minimum at an edge.

Run it with `launch_workflow` and give an intent. The workflow writes
`eos.json` (a list of `{"volume", "energy"}` points, in Å^3 and eV) and
records one traced task per volume, so every point has provenance.

Do not relax inside the scan. The scan holds each scaled cell fixed on
purpose, because relaxing the cell would walk every point back to the same
minimum.

## 2. Fit

Run the bundled fit script on the recorded points:

    python <skill root>/scripts/fit_eos.py eos.json

It prints V0 (Å^3), E0 (eV), and B (GPa), and `--json` gives the same
numbers machine-readable. Pass `--natoms N` to also get the volume per
atom. For a cubic cell, the lattice constant is `(V0 * cells_per_V0)**(1/3)`
with the multiplicity of your cell; state the conversion you used.

## 3. Verify and report

- The template's checks gate the run: energies finite, minimum interior to
  the scanned range. If `minimum_is_interior` fails, widen `SCALES` and
  rerun; do not fit a curve whose minimum sits on the boundary.
- Sanity-check B: most metals fall between roughly 20 and 400 GPa. A
  negative or wildly large B means the curve is not convex around the
  minimum — look at the points before trusting any number.
- Report V0, E0, and B with the run id of the scan, and say which engine
  and scale range produced them.

## When not to use this

- Molecules and isolated systems have no meaningful E(V) curve.
- Systems with internal degrees of freedom that relax strongly under
  strain (layered or molecular crystals) need constant-volume internal
  relaxation at each point; the plain single-point ladder understates
  their stiffness. Say so instead of reporting the naive fit.
