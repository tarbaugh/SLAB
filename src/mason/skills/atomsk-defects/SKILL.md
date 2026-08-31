---
name: atomsk-defects
description: Insert point defects and dislocations with atomsk —
  vacancies, interstitials, substitutions, random disorder, screw and
  edge dislocations, dislocation loops. Use when asked to build a
  defective crystal, a doped structure, or a dislocation configuration.
license: MIT
metadata:
  mason-agents: "dft-expert md-expert"
---
# Point defects and dislocations with atomsk

Atomsk inserts defects as options applied after an input structure. Run
it through `foundation.tasks.build_structure` (see the atomsk-structures
skill for the calling pattern); stage the perfect crystal with
`inputs=` and apply defect options in the argument list.

## Point defects

- Vacancy: select, then remove. `-select random 1 Al -rmatom select`
  removes one random Al atom; `-rmatom 42` removes atom index 42 (use a
  fixed index, not `random`, when the run must be reproducible).
- Substitution (doping): `-substitute Al Mg` replaces every Al with Mg;
  to dope a fraction, select first: `-select random 2% Al -substitute
  Al Mg`.
- Interstitial: `-add-atom H at 0.25*box 0.25*box 0.25*box` places an
  atom at a position; `-add-atom H relative 12 0.5 0.5 0.5` places it
  relative to atom 12.
- Small random displacements to break symmetry before relaxation:
  `-disturb 0.05`.

Build the supercell first, then insert the defect, in one invocation:
`-duplicate 4 4 4 -select random 1 Al -rmatom select`. A defect in the
unit cell duplicates into a defect array.

## Dislocations

- Screw: `-dislocation 0.5*box 0.5*box screw z y 2.86` inserts a screw
  dislocation with line along z, cut plane normal y, Burgers vector
  2.86 Å. Positions are in the plane normal to the line.
- Edge: `-dislocation 0.5*box 0.5*box edge_add z y 2.86 0.33` needs the
  Poisson ratio; `edge_add` inserts a half-plane, `edge_rm` removes one.
- Loop: `-dislocation loop 0.5*box 0.5*box 0.5*box z 20 2.86 0 0 0.33`.
- Orient the cell so the intended line and glide directions lie along
  the cartesian axes first (`orient` in `--create`, or `-orient`); the
  Burgers vector magnitude comes from the oriented lattice.

Dislocation displacement fields are long-ranged. Use a large supercell,
and remember the periodic images: a single dislocation in a periodic
cell carries a net Burgers vector and strains the whole box. Dipole
configurations (two opposite dislocations) cancel it.

## Verify and report

- Check the defective structure with the atomsk-structures skill's
  `check_structure.py`: the expected atom count (one less per vacancy),
  the formula after doping, and no close contacts from an interstitial
  placed on top of a host atom.
- Relax before measuring. Report defect formation energies from total
  energies of defective and perfect supercells of the same size, same
  engine, same settings; record both run ids.
- Random selections (`-select random`) differ between invocations.
  Record the produced structure artifact as the source of truth, and
  prefer deterministic placements when a study must be repeatable.

## When not to use this

- Charged defects in insulators need charge compensation and finite-size
  corrections that plain total-energy differences do not include; say so
  rather than reporting an uncorrected number as the formation energy.
- Grain boundaries and interfaces are not point defects; use the
  atomsk-interfaces skill.
