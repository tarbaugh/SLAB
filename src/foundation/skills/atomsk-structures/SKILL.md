---
name: atomsk-structures
description: Build crystals with atomsk — unit cells, oriented cells,
  supercells, strained cells, and format conversions — as traced SLAB
  runs. Use when asked to create a bulk structure, a supercell, a
  crystal in a specific crystallographic orientation, a strained cell,
  or to convert a structure file between formats.
license: MIT
metadata:
  mason-agents: "dft-expert md-expert"
---
# Building structures with atomsk

Atomsk is a structure builder: it creates and transforms atomic
structures and computes no energies. In SLAB it is a *builder*, not an
engine. Run it through `foundation.tasks.build_structure` inside a run,
so the structure is a traced artifact with the atomsk version in its
cache identity:

    from foundation.tasks import build_structure

    supercell, info = build_structure(
        "--create fcc 4.046 Al -duplicate 4 4 4 al.xsf", label="al-444"
    )

The task runs atomsk in a private scratch directory. Use bare file
names only; the task refuses paths. It reads the produced file back as
an ASE `Atoms`, ready for `relax` or `single_point`, and keeps the file
and the atomsk log as artifacts. The binary comes from
`[builders.atomsk]` in `slab.toml` (default: `atomsk` on PATH). Check
it first with the shell: `atomsk --version`.

## 1. Fix the lattice constant first

Atomsk has no lattice-constant database; the number you give is the
number you get. For any energy difference (a defect, a surface, a
boundary, a strain ladder) build the cell at the engine's own
equilibrium: relax a unit cell with `foundation.tasks.relax_cell` under
the production engine, read its lattice constant, and pass that to
`--create`. The experimental value leaves residual stress in the
"perfect" reference and that stress enters every difference.

## 2. Recipes

- Unit cell: `--create <lattice> <a> [<c>] <species...> <file>` with
  lattices `sc bcc fcc diamond rocksalt perovskite st bct hcp wurtzite
  graphite` and more. Example: `--create rocksalt 4.21 Mg O mgo.xsf`
  (`rs` and `b1` are aliases; `rock-salt` is not a name).
- Oriented cell: append `orient [hkl] [hkl] [hkl]`, e.g. `--create fcc
  3.615 Cu orient [110] [-110] [001] cu110.xsf`. Write negative indices
  inside the brackets: `[1-10]`. Atomsk reduces the indices and takes the
  shortest lattice vector along each direction, so the atom count is not
  the unit-cell count times the index products; read it from the output.
  Hexagonal lattices accept `orient` too, with the first vector along x
  and a non-orthogonal result; add `-orthogonal-cell` to box it.
- Supercell: `-duplicate <Nx> <Ny> <Nz>` after the mode.
- Reorient an existing crystal: `-orient [100] [010] [001] [110] [-110]
  [001] -orthogonal-cell` rotates from the first triplet to the second.
  The rotation alone leaves the cell vectors off the Cartesian axes;
  `-orthogonal-cell` finds the equivalent axis-aligned box and changes the
  atom count. Put it before `-duplicate`.
- Strain: `-deform x 0.02 0.3` applies 2% uniaxial strain along x with
  Poisson ratio 0.3; `-deform xy 0.01` is a shear and takes no ratio.
- Convert: pass the input as a staged file and name the output with the
  wanted extension. `.xsf` and `POSCAR` round-trip well with ASE;
  `.lmp` writes a LAMMPS data file; `.pw` writes a Quantum ESPRESSO
  input skeleton whose cutoff is a placeholder, never a setting to run.
  Before a POSCAR add `-sort species pack -frac`; before `.lmp` add
  `-alignx -unskew`. Without them atomsk asks a question on stdin about
  the species order or the cell alignment, and a traced run hangs.

To transform a structure you already have as `Atoms`, stage it:

    slab_cell, info = build_structure(
        ["si.xsf", "-duplicate", "3", "3", "3", "si333.lmp"],
        inputs={"si.xsf": relaxed},
        output="si333.lmp",
    )

If atomsk writes more than one file, name the result with `output=`.

## 3. Verify and report

- Run the bundled check on the produced file before using it:

      python <skill root>/scripts/check_structure.py al.xsf --json --expect-atoms 256

  It reports atom count, formula, cell, density, and the minimum
  interatomic distance under periodic boundary conditions, and compares
  that minimum with the shortest bond the closest pair's covalent radii
  predict. It fails when the minimum is below 0.6 of that bond (set
  `--fail-below-fraction`, or an absolute `--fail-below` in Å), and when
  the atom count differs from `--expect-atoms`. Overlapping atoms mean
  rebuild, not relax.
- Compare the density against the known value for the phase. A wrong
  lattice constant or a doubled cell shows up here first.
- Record the run id and the exact atomsk argument list in the notebook;
  `info["args"]` and `info["version"]` hold them.

## When not to use this

- Amorphous or liquid starting states: atomsk places atoms on lattices.
  Build a crystal and melt it (see the melt-quench skill) instead.
- A relaxed structure is not built here: atomsk output is unrelaxed
  geometry. Relax before measuring anything.
- Defects and interfaces have their own skills (atomsk-defects,
  atomsk-interfaces); this one covers perfect crystals and conversions.
