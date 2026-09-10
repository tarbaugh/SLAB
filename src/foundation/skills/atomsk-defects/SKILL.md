---
name: atomsk-defects
description: Insert point defects and dislocations with atomsk —
  vacancies, interstitials, substitutions, random disorder, screw and
  edge dislocations, dislocation loops — and compute formation energies
  with chemical potentials and finite-size control. Use when asked to
  build a defective crystal, a doped structure, or a dislocation
  configuration.
license: MIT
metadata:
  mason-agents: "dft-expert md-expert"
---
# Point defects and dislocations with atomsk

Atomsk inserts defects as options applied after an input structure. Run
it through `foundation.tasks.build_structure` (see the atomsk-structures
skill for the calling pattern); stage the perfect crystal with
`inputs=` and apply defect options in the argument list. Build the
perfect crystal at the engine's own relaxed lattice constant, because
every formation energy below is a difference against it.

## 1. Point defects

- Vacancy: select, then remove. `-select random 1 Al -rmatom select`
  removes one random Al atom; `-rmatom 42` removes atom index 42 (use a
  fixed index, not `random`, when the run must be reproducible).
- Substitution (doping): `-substitute Al Mg` replaces every Al with Mg;
  to dope a fraction, select first: `-select random 2% Al -substitute
  Al Mg`.
- Interstitial: `-add-atom H at 0.25*box 0.25*box 0.25*box` places an
  atom at a position; `-add-atom H near 12` places it in the largest
  cavity next to atom 12. The `relative` form takes an offset in Å, so
  `-add-atom H relative 12 0.5 0.5 0.5` puts H 0.87 Å from atom 12, on
  top of it; give a physical offset (about 1.5 Å or more) or use `near`.
- Small random displacements to break symmetry before relaxation:
  `-disturb 0.05`.

Build the supercell first, then insert the defect, in one invocation:
`-duplicate 4 4 4 -select random 1 Al -rmatom select`. A defect in the
unit cell duplicates into a defect array.

## 2. Formation energies

The formation energy of a neutral defect is

    E_f = E_defect - E_perfect + sum_i n_i mu_i

where n_i is the number of atoms of species i removed (+1) or added
(-1), and mu_i is that species' chemical potential. For a vacancy, mu is
the perfect crystal's energy per atom, E_perfect/N. For a dopant or an
interstitial, mu comes from the element's reference phase (or a competing
compound), computed with the same engine and settings; the two-cell
difference alone is not a formation energy for a substitution.

- Use the same supercell size, engine, protocol expansion, k-mesh, and
  cutoff for both cells, and say whether the defect cell relaxed at
  constant volume or constant pressure.
- Neutral defects interact with their images elastically, roughly as
  L^-3. Keep at least 10 Å between images in a near-cubic cell, and
  confirm E_f against one larger cell before reporting it.

## 3. Dislocations

- Screw: `-dislocation 0.501*box 0.501*box screw z y 2.86` inserts a
  screw dislocation with line along z, cut plane normal y, Burgers vector
  2.86 Å. Positions are in the plane normal to the line. Keep the centre
  off a lattice site (0.501, not 0.5); a centre on an atom gives that
  atom an unphysical displacement.
- Edge: `-dislocation 0.501*box 0.501*box edge_add z y 2.86 0.33` needs
  the Poisson ratio; `edge_add` inserts a half-plane and lengthens the
  box by b/2, `edge_rm` removes one and shortens it. Screw and mixed
  take no ratio.
- Loop: `-dislocation loop 0.5*box 0.5*box 0.5*box z 20 2.86 0 0 0.33`.
- Orient the cell so the intended line and glide directions lie along
  the cartesian axes first (`orient` in `--create`, or `-orient`); the
  Burgers vector magnitude comes from the oriented lattice. For bcc and
  other anisotropic metals, pass the elastic tensor with `-properties`
  so atomsk uses anisotropic elasticity for the displacement field.

Dislocation displacement fields are long-ranged, and a single
dislocation breaks periodicity in both in-plane directions; atomsk
applies no fix. Use a quadrupolar dipole array: four dislocations at
(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75) of the box with
alternating sign, which cancels the net Burgers vector and minimises the
image forces on each core. Edge dipoles also need a homogeneous strain
that cancels the plastic strain of the inserted half-planes. A core
energy is the array's energy minus its elastic energy, which needs an
anisotropic elasticity code; report the array's energy per line length
as such, not as a core energy.

## Verify and report

- Check the defective structure with the atomsk-structures skill's
  `check_structure.py --expect-atoms N`: the expected atom count (one
  less per vacancy), the formula after doping, and no close contacts from
  an interstitial placed on top of a host atom.
- Relax before measuring. Report E_f with the chemical potentials and
  their sources, the supercell size, the relaxation condition, and both
  run ids.
- Random selections (`-select random`) differ between invocations.
  Record the produced structure artifact as the source of truth, and
  prefer deterministic placements when a study must be repeatable.

## When not to use this

- Charged defects in insulators need the charge state's Fermi-level
  term q(E_VBM + E_F), an image-charge and potential-alignment correction
  (Freysoldt, Neugebauer, and Van de Walle; Kumagai and Oba for
  anisotropic dielectrics), and a band-filling check. The neutral
  formula above does not include them; say so rather than reporting an
  uncorrected number.
- Grain boundaries and interfaces are not point defects; use the
  atomsk-interfaces skill.
