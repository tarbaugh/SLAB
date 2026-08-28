---
name: interface-adhesion
description: Work of adhesion from an interface supercell and its two
  isolated slabs, and the heterogeneous-nucleation potency f(theta) it
  implies. Use when asked how well two materials bind, for a work of
  adhesion, a wetting angle, or a substrate's nucleation potency.
license: MIT
metadata:
  mason-agents: "dft-expert analysis-expert"
---
# Interface adhesion

A work of adhesion compares an interface supercell against its two
halves, each isolated in the same footprint:

    W_adh = (E_slab_A + E_slab_B - E_interface) / area

The bundled script does the arithmetic and, given the nucleating phase's
interfacial free energy, converts W_adh into a Young-Dupre wetting angle
and the classical-nucleation-theory potency factor f(theta).

## 1. Compute the three energies

- Build the interface supercell with the two slabs in contact and vacuum
  above and below, so the cell holds exactly one A/B interface. Relax it.
- The isolated-slab references are the same cell with one slab deleted,
  each relaxed. Same engine, same settings expansion, same k-sampling
  footprint, same cell vectors.
- Strain bookkeeping: matching the two lattices strains at least one
  slab, and that strain energy must sit in the *reference* slab too, or
  it pollutes W_adh. Keep each reference slab at the strained in-plane
  lattice of the interface cell.
- Record the in-plane cell vectors and the three run ids so the
  arithmetic is reproducible from `show_run` alone.

## 2. Fit

    python <skill root>/scripts/adhesion.py --e-interface -1802.10 \
        --e-a -1204.62 --e-b -595.31 --area 187.4 --gamma 0.062

It prints W_adh in eV/A^2 and J/m^2; with `--gamma` (J/m^2) it adds
cos(theta) = W_adh/gamma - 1, the angle, and f(theta) =
(2+cos)(1-cos)^2/4, which multiplies the homogeneous nucleation barrier.
`--json` gives the numbers machine-readable.

## 3. Verify and report

- A non-positive W_adh means the slabs do not bind; check the relaxation
  and the shared settings before believing it.
- W_adh above 2*gamma means complete wetting (f = 0); the script clamps
  and warns. Real substrates rarely wet completely; treat a clamped
  result as a sign gamma or W_adh is off.
- Convergence: W_adh must be flat against slab thickness and vacuum, the
  same discipline as the surface-energy skill.
- Report W_adh with the two terminations, the in-plane strain state, the
  engine, and the run ids; report f(theta) together with the gamma used
  to derive it.

## When not to use this

- A crystalline approximant of an amorphous layer gives a crystalline
  answer; tag it as the approximant it is, and treat the amorphous
  interface as a different calculation.
- Charged or polar terminations need the same dipole care as polar
  surfaces; the plain three-energy arithmetic does not include it.
- f(theta) assumes an isotropic spherical-cap nucleus on a flat
  substrate. Faceted nuclei and rough or patterned substrates break the
  formula; say so rather than stretching it.
