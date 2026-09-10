---
name: equation-of-state
description: Compute an energy-volume curve and fit a Birch-Murnaghan
  equation of state to get the equilibrium volume, energy, bulk modulus,
  and its pressure derivative for a crystal. Use when asked for a lattice
  constant, an equilibrium volume, a bulk modulus, an E(V) curve, or to
  validate an engine against a known material.
license: MIT
metadata:
  mason-agents: "dft-expert analysis-expert"
---
# Equation of state

The procedure has two parts. A workflow computes single-point energies on a
ladder of scaled cells. A bundled script fits the third-order
Birch-Murnaghan form to the recorded points.

## 1. Compute the energy-volume curve

Copy `assets/eos_scan.py` into the project directory and adapt the
constants at the top:

- `STRUCTURE`: the bulk cell. The experimental lattice constant is a
  reasonable start; `foundation.tasks.relax_cell` at the same engine is
  the honest one, and the scan then brackets a real minimum instead of a
  reference-strain artefact.
- `ENGINE`: the engine for the single points. The template runs as-is
  under `emt` as a shakeout; switch to the production engine for a real
  number.
- `VOLUME_SCALES`: fractions of the reference volume (the cell is scaled
  by the cube root). Seven points from 0.94 to 1.06 in volume is the
  Delta-protocol range, and the fit's bias is under 1 % there. A wider
  range (up to ±10 %) is less noise-sensitive but biases B by a few
  percent; state the range you used. Widen only if the fit reports the
  minimum at an edge.
- `RELAX_INTERNAL`: set `True` for cells with free internal coordinates
  (layered, molecular, or low-symmetry crystals). Clamped ions raise the
  energy away from V0 and overstate B; the relaxed ladder gives the
  physical curve. The cell itself is never relaxed inside the scan.

Keep the numerical settings identical at every volume: the same explicit
k-mesh (a spacing-derived mesh changes between points and puts steps in
E(V)), the same cutoff converged for energy differences, the same
smearing. Record the free energy the code reports with the smearing term
included. Run it with `launch_workflow` and give an intent. The workflow
writes `eos.json` (a list of `{"volume", "energy"}` points, in Å^3 and
eV) and records one traced task per volume, so every point has
provenance.

## 2. Fit

Run the bundled fit script on the recorded points:

    python <skill root>/scripts/fit_eos.py eos.json --natoms 1

It prints V0 (Å^3), E0 (eV), B (GPa), B', the RMS residual, and the
scanned range as fractions of V0; `--json` gives the same numbers
machine-readable, with the warnings in a list. Pass `--natoms N` to get
the volume and the residual per atom. For a cubic cell, the lattice
constant is `(V0 * cells_per_V0)**(1/3)` with the multiplicity of your
cell; state the conversion you used.

## 3. Verify and report

- The template's checks gate the run: energies finite, minimum interior to
  the scanned range. If `minimum_is_interior` fails, or the script warns
  that the fitted V0 lies outside the inner points, re-centre the scan on
  the fitted V0 and rerun; do not fit a curve whose minimum sits on the
  boundary.
- Heed the residual warning. More than 1 meV/atom means the points are
  noisy or the settings changed between them; fix the scan instead of
  reporting the fit.
- Sanity-check B: most metals fall between roughly 10 and 400 GPa, the
  alkali metals between 2 and 10. A negative or wildly large B means the
  curve is not convex around the minimum; look at the points before
  trusting any number. B' near 4 is typical; far from it, check the
  range.
- Report V0, E0, B, and B' with the run id of the scan, the engine, the
  volume range, and whether internal relaxation was on.

## When not to use this

- Molecules and isolated systems have no meaningful E(V) curve.
- Total energies from different pseudopotential families or codes are
  not comparable; compare V0 and B, never E0, across them.
