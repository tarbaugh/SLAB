---
name: elastic-constants
description: Compute elastic constants from energy-strain scans and average
  them into bulk and shear moduli, Young's modulus, and the Poisson ratio.
  Use when asked for C_ij, a stiffness, E, nu, or mechanical stability of a
  crystal or a glass.
license: MIT
metadata:
  mason-agents: "dft-expert analysis-expert"
---
# Elastic constants

The procedure has two parts. A workflow computes energy ladders along a
small set of strain patterns. A bundled script fits the ladders and
assembles C_ij, the Voigt-Reuss-Hill moduli, E, and nu.

## 1. Compute the energy-strain ladders

Copy `assets/strain_scan.py` into the project directory and adapt the
constants at the top:

- `STRUCTURE`: the *relaxed* cell. An unrelaxed reference leaves a linear
  term in every ladder, and the fit script warns about exactly that.
- `ENGINE`: the engine for the energies. The template runs as-is under
  `emt` as a shakeout.
- `SYMMETRY`: `isotropic` (2 modes; use for glasses and amorphous cells),
  `cubic` (3 modes), or `orthorhombic` (9 modes).
- `DELTAS`: strain amplitudes. Seven points over +/-1.5% suit most
  solids. Keep 0.0 in the list and keep the range symmetric.
- `RELAX_INTERNAL`: set `True` for materials with internal degrees of
  freedom (layered, molecular, or ribboned crystals). The plain
  single-point ladder holds the ions to affine positions and overstates
  their stiffness; the relaxed ladder gives the physical constants.

Run it with `launch_workflow` and give an intent. The workflow writes
`elastic.json` and records one traced task per strained cell.

## 2. Fit

    python <skill root>/scripts/fit_elastic.py elastic.json

It prints the C_ij in GPa, the Voigt/Reuss/Hill bulk and shear moduli,
and E and nu from the Hill averages; `--json` gives the same numbers
machine-readable.

## 3. Verify and report

- The template's checks gate the run: energies finite, every ladder's
  minimum interior. A minimum at a ladder's edge means the reference is
  strained; re-relax and rerun.
- Heed the script's warnings. A large linear term means the reference is
  not at its minimum. A non-positive-definite matrix means the structure
  is mechanically unstable or a ladder is noise; do not report averages
  from it.
- Sanity-check against the bulk modulus of an equation-of-state fit on
  the same engine; B_Hill and B(EOS) should agree within a few GPa.
- Report the C_ij with the symmetry, the strain range, whether internal
  relaxation was on, the engine, and the run id. For an amorphous cell,
  report E and nu as averages over at least two independent glasses and
  state the spread.

## When not to use this

- Liquids have no elastic constants; a flat or non-convex ladder is the
  signature.
- Finite-temperature (relaxed isothermal) constants need strained MD
  averages, not static ladders; this skill's numbers are athermal.
- Cells under residual stress need a stress-strain treatment; fix the
  reference first instead of fitting through the warning.
