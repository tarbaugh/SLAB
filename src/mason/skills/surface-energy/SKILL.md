---
name: surface-energy
description: Compute a surface energy from a slab and a bulk reference
  calculation. Use when asked for a surface energy, surface stability
  ordering, or which facet a crystal exposes.
license: MIT
metadata:
  mason-agents: "dft-expert"
---
# Surface energy

A surface energy compares a slab against the bulk it was cut from:

    gamma = (E_slab - N_slab / N_bulk * E_bulk) / (2 * A)

The factor 2 counts both faces of the slab, and A is the surface cell
area. This skill bundles no script - the arithmetic is one line - but the
procedure has traps the checklist below guards.

## 1. Compute the pair

- Build the slab with `ase.build.surface` (or `fcc111` and friends) from
  the *relaxed* bulk, with vacuum on both sides.
- Compute E_bulk and E_slab in one workflow, with the same engine, the
  same protocol expansion, the same pseudopotential family, and k-meshes
  that match in the in-plane directions (the slab's normal direction
  takes 1 k-point through the vacuum).
- Record N_bulk, N_slab, and the in-plane cell vectors in the run so the
  arithmetic is reproducible from `show_run` alone.

## 2. Converge what the number depends on

- Layers: gamma oscillates with slab thickness; grow the slab until
  successive thicknesses agree within your tolerance and add a check for
  that.
- Vacuum: 10 A is a starting point, not a result; confirm gamma is flat
  against vacuum thickness.
- The bulk reference must be converged per the convergence-study skill;
  an unconverged bulk poisons every facet's gamma equally and invisibly.

## 3. Report

- Report gamma in eV/A^2 and J/m^2 (1 eV/A^2 = 16.0218 J/m^2), with the
  facet, the termination, whether the slab was relaxed, and the run id.
- Unrelaxed (cleaved) gamma is a different quantity from relaxed gamma;
  never mix them in one ordering.

## When not to use this

- Polar terminations of ionic crystals: the naive formula diverges with
  thickness; those need dipole corrections or reconstructed
  terminations, and saying so is the correct answer.
- Alloys and compounds with off-stoichiometric slabs need chemical
  potentials; the two-energy formula above assumes stoichiometric slabs.
