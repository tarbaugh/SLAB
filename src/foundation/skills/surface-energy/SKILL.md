---
name: surface-energy
description: Compute a surface energy from a slab and a bulk reference
  that share the same in-plane cell and k-mesh, converge it in thickness
  without the linear drift a separate bulk energy causes, and rank facets
  by a Wulff construction. Use when asked for a surface energy, surface
  stability ordering, or which facet a crystal exposes.
license: MIT
metadata:
  mason-agents: "dft-expert"
---
# Surface energy

A surface energy compares a slab against the bulk it was cut from:

    gamma = (E_slab - N_slab / N_bulk * E_bulk) / (2 * A)

The factor 2 counts both faces of a symmetric slab, and A is the surface
cell area. This skill bundles no script; the arithmetic is one line, but
the procedure has traps the checklist below guards.

## 1. Compute the pair

- Relax the bulk first with `foundation.tasks.relax_cell` under the
  production engine (`isotropic` for cubic, `orthorhombic` for
  Pnma/Cmcm, `triclinic` otherwise; hexagonal and tetragonal cells need
  a and c free) so gamma is measured against a stress-free reference.
- Take E_bulk from the surface-oriented bulk cell, not from the
  primitive cell. Build the oriented cell with `ase.build.surface(...,
  vacuum=0)` (or pymatgen's `SlabGenerator.oriented_unit_cell`) and cut
  the slab from the same cell. Then the in-plane k-mesh matches exactly:
  k1 x k2 x k3 for the bulk, k1 x k2 x 1 for the slab. A primitive-cell
  E_bulk has no in-plane match for most facets, and any mismatch between
  E_bulk and the slab's energy per layer makes gamma drift linearly with
  thickness instead of converging.
- Build a symmetric slab: the same termination on both faces, both
  faces relaxed. `ase.build.surface` does not guarantee equal
  terminations for a compound; check the two faces. If the bottom layers
  are fixed, the two faces differ and the formula becomes
  gamma_relaxed = (E_slab - N E_bulk)/A - gamma_unrelaxed.
- Any asymmetric slab (different terminations, one face relaxed, an
  adsorbate) carries a dipole through the vacuum; apply the engine's
  dipole correction.
- Same engine, same protocol expansion, same pseudopotential family,
  same smearing, same spin setting for both cells. Use odd, Γ-centred
  meshes for hexagonal surface zones.
- Record N_bulk, N_slab (their ratio must be an integer for a
  stoichiometric slab), and the in-plane cell vectors in the run so the
  arithmetic is reproducible from `show_run` alone.

## 2. Converge what the number depends on

- Thickness: compute three or more thicknesses and fit
  E_slab(N) = N * E_bulk_fit + 2 gamma A (the Fiorentini–Methfessel
  fit). The slope must agree with the oriented-cell E_bulk; the
  intercept is gamma. Two adjacent thicknesses agreeing proves nothing
  when E_bulk is off. Metals converge in about five layers, oxides such
  as alpha-Al2O3(0001) need fifteen or more. Add a check that the fitted
  gamma stays within 0.02 J/m^2 across the thicknesses.
- Vacuum: 10 to 15 Å; confirm gamma is flat against vacuum thickness.
- The bulk reference must be converged per the convergence-study skill;
  an unconverged bulk poisons every facet's gamma equally and invisibly.

## 3. Rank facets

Which facet a crystal exposes is a Wulff construction over several
facets, not the single lowest gamma. Compute gamma for the low-index
facets and their terminations, then build the Wulff shape (pymatgen's
`WulffShape`) and report the facet areas it exposes.

## 4. Report

- Report gamma in eV/A^2 and J/m^2 (1 eV/A^2 = 16.0218 J/m^2), with the
  facet, the termination, whether the slab was relaxed, the thicknesses
  fitted, the vacuum, and the run id. Compare elemental values with the
  Materials Project surface database (PBE, with a standard error near
  0.27 J/m^2 against experiment).
- Unrelaxed (cleaved) gamma is a different quantity from relaxed gamma;
  never mix them in one ordering. Relaxation lowers gamma by up to about
  30%.

## When not to use this

- Polar (Tasker type 3) terminations of ionic crystals: the energy
  diverges with thickness, and a dipole correction does not fix it;
  those need charge compensation or reconstructed terminations, and
  saying so is the correct answer.
- Alloys and compounds with off-stoichiometric slabs need chemical
  potentials; the two-energy formula above assumes stoichiometric slabs.
