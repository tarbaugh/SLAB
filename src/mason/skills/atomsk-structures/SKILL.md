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

## Recipes

- Unit cell: `--create <lattice> <a> [<c>] <species...> <file>` with
  lattices `sc bcc fcc diamond rock-salt perovskite st bct hcp wurtzite
  graphite` and more. Example: `--create rock-salt 4.21 Mg O mgo.xsf`.
- Oriented cell (cubic lattices): append `orient [hkl] [hkl] [hkl]`,
  e.g. `--create fcc 3.615 Cu orient [110] [-110] [001] cu110.xsf`.
  Write negative indices inside the brackets: `[1-10]`.
- Supercell: `-duplicate <Nx> <Ny> <Nz>` after the mode.
- Reorient an existing crystal: `-orient [100] [010] [001] [110] [-110]
  [001]` rotates from the first triplet to the second.
- Strain: `-deform x 0.02 0.3` applies 2% uniaxial strain along x with
  Poisson ratio 0.3; `-deform xy 0.01` is a shear.
- Convert: pass the input as a staged file and name the output with the
  wanted extension. `.xsf` and `POSCAR` round-trip well with ASE;
  `.lmp` writes a LAMMPS data file; `.pw` writes a Quantum ESPRESSO
  input skeleton.

To transform a structure you already have as `Atoms`, stage it:

    slab_cell, info = build_structure(
        ["si.xsf", "-duplicate", "3", "3", "3", "si333.lmp"],
        inputs={"si.xsf": relaxed},
        output="si333.lmp",
    )

If atomsk writes more than one file, name the result with `output=`.

## Verify and report

- Run the bundled check on the produced file before using it:

      python <skill root>/scripts/check_structure.py al.xsf --json

  It reports atom count, formula, cell, density, and the minimum
  interatomic distance under periodic boundary conditions. A minimum
  distance far below a bond length means overlapping atoms; rebuild
  instead of relaxing the overlap away.
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
