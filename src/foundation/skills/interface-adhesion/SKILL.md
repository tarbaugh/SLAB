---
name: interface-adhesion
description: Work of adhesion from an interface supercell and its two
  isolated slabs, the interface energy it gives with the two surface
  energies, and the heterogeneous-nucleation potency f(theta) from the
  solid-liquid interfacial energies. Use when asked how well two
  materials bind, for a work of adhesion or separation, an interface
  energy, a wetting angle, or a substrate's nucleation potency.
license: MIT
metadata:
  mason-agents: "dft-expert analysis-expert"
---
# Interface adhesion

A work of adhesion compares an interface supercell against its two
halves, each isolated in the same footprint:

    W_adh = (E_slab_A + E_slab_B - E_interface) / area

The bundled script does the arithmetic, gives the interface energy from
the two surface energies, and converts the solid-liquid interfacial
energies into a wetting angle and the classical-nucleation-theory
potency factor f(theta).

## 1. Build and relax the interface

- Build the interface with the atomsk-interfaces skill. A cell with
  vacuum above and below holds exactly one A/B interface, but its two
  outer faces differ, so it carries a dipole: apply the engine's dipole
  correction, or build a symmetric B/A/B sandwich with two equivalent
  interfaces and divide by two areas.
- Scan the registry before relaxing. One relaxation from an arbitrary
  lateral offset finds a local minimum, and the registry moves W_adh by
  a factor of two or more (Al/Al2O3 from 0.4 to 1.1 J/m^2 between three
  stackings). Shift one slab over at least six in-plane offsets on the
  high-symmetry sites of the interface cell, relax each, and take the
  lowest; report the spread. Repeat for each termination when the
  compound has more than one; terminations move W_adh by an order of
  magnitude.
- Strain bookkeeping: matching the two lattices strains at least one
  slab. Strain the softer slab, or split the strain in inverse
  proportion to the bulk moduli, and keep each reference slab at the
  strained in-plane lattice of the interface cell, or the strain energy
  pollutes W_adh. Record the misfit in percent and which slab carries
  it. Pass the free-lattice slab energies as `--e-a-free`/`--e-b-free`
  and the script reports each strain energy per area separately.
- Same engine, same settings expansion, same k-sampling footprint, same
  cell vectors for all three cells. Record the in-plane cell vectors and
  the three run ids so the arithmetic is reproducible from `show_run`
  alone.

## 2. Compute

    python <skill root>/scripts/adhesion.py --e-interface -1802.10 \
        --e-a -1204.62 --e-b -595.31 --area 187.4 \
        --gamma-a 1.20 --gamma-b 0.95 --gamma-nl 0.40 --gamma-sl 0.25

It prints W_adh in eV/A^2 and J/m^2. With `--frozen-references` (the
slabs were not relaxed after separation) it names the number the work of
separation, which differs from W_adh by up to 0.6 J/m^2. With the two
free-surface energies from the surface-energy skill it prints the
interface energy gamma_int = gamma_a + gamma_b - W_adh (the Dupre
relation). With the substrate-liquid and nucleus-liquid interfacial
energies it prints cos(theta) = (gamma_NL - gamma_NS)/gamma_SL, the
angle, and f(theta) = (2+cos)(1-cos)^2/4, which multiplies the
homogeneous nucleation barrier. `--json` gives the numbers
machine-readable.

The vacuum W_adh alone gives no wetting angle. Dividing a 1 to 4 J/m^2
vacuum work by a 0.06 to 0.25 J/m^2 solid-liquid energy lands above
complete wetting for nearly every pair; the quantity nucleation compares
is the solid-solid interface energy gamma_NS against gamma_SL.

## 3. Verify and report

- A non-positive W_adh means the slabs do not bind; check the relaxation
  and the shared settings before believing it. Known interfaces sit
  between about 0.1 and 12 J/m^2 (metal/metal 0.5 to 3, Al-terminated
  metal/oxide about 1, O-terminated oxides 7 to 11); the script warns
  outside that band.
- Complete wetting (f = 0) means gamma_NS is below gamma_NL - gamma_SL.
  Real substrates rarely wet completely; treat a clamped result as a
  sign one of the four energies is off.
- Convergence: W_adh must be flat against slab thickness and vacuum, the
  same discipline as the surface-energy skill.
- Report W_adh (or W_sep) with the two terminations, the registry
  minimum and spread, the in-plane strain state and strain energies, the
  engine, and the run ids; report gamma_int with the surface energies
  used, and f(theta) with gamma_NL and gamma_SL and their sources.

## When not to use this

- A crystalline approximant of an amorphous layer gives a crystalline
  answer; tag it as the approximant it is, and treat the amorphous
  interface as a different calculation.
- Charged or polar terminations need the same dipole and charge care as
  polar surfaces; the plain three-energy arithmetic does not include it.
- f(theta) assumes a macroscopic spherical cap on a flat substrate with
  no line tension and no misfit strain in the nucleus. Faceted nuclei,
  nuclei of a few atomic spacings, and rough or patterned substrates
  break the formula; say so rather than stretching it.
