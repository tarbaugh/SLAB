---
name: atomsk-interfaces
description: Assemble interfaces, bicrystals, and polycrystals with
  atomsk — stack two slabs into an interface supercell, build grain
  boundaries from misoriented grains, generate Voronoi polycrystals.
  Use when asked to build an interface, a grain boundary, a bicrystal,
  or a polycrystalline cell.
license: MIT
metadata:
  mason-agents: "dft-expert md-expert"
---
# Interfaces and polycrystals with atomsk

Two atomsk modes assemble multi-grain systems: `--merge` stacks
existing structures, `--polycrystal` fills a box with misoriented
grains. Run both through `foundation.tasks.build_structure` (see the
atomsk-structures skill for the calling pattern).

## Interfaces and bicrystals: --merge

    atomsk --merge stack z 2 bottom.xsf top.xsf interface.xsf

stacks two files along z into one cell. The workflow:

1. Build each slab separately, oriented so the intended interface plane
   is normal to the stacking axis (atomsk-structures skill).
2. Match the in-plane lattices. The slabs keep their own cell vectors;
   strain one slab to the other's in-plane cell (`-deform`, or rebuild
   at the strained lattice constant) before merging. Record which slab
   carries the strain — the interface-adhesion skill needs the same
   strain in its reference slabs.
3. Merge with `--merge stack z 2 ...` and stage both slabs via
   `inputs=`. The merged cell is the sum of the stacked dimensions.
4. Set the interface separation by shifting one slab before the merge
   (`-shift`) or by editing the gap; a merge that leaves atoms
   overlapping at the seam is the common failure.
5. For a free-standing interface with vacuum, enlarge the cell along
   the stacking axis afterwards: `-cell z add 15` (older atomsk
   releases order it `-cell add 15 z`; on an error, atomsk prints the
   order your build expects).

`--merge scale x 2 ...` instead rescales structures to a common length
along x; use `stack` for interfaces.

## Grain boundaries and polycrystals: --polycrystal

    atomsk --polycrystal fcc-al.xsf poly.txt polycrystal.lmp

`fcc-al.xsf` is the seed (one unit cell) and `poly.txt` is a parameter
file; stage both through `inputs=` — an `Atoms` value becomes a
structure file, a string value is written verbatim:

    cell, info = build_structure(
        "--polycrystal fcc-al.xsf poly.txt polycrystal.lmp",
        inputs={"fcc-al.xsf": seed, "poly.txt": "box 200 200 200\nrandom 12\n"},
        output="polycrystal.lmp",
    )

The parameter file declares the box and the grains:

    box 200 200 200
    random 12

gives twelve randomly placed, randomly oriented grains in a 200 Å box.
`node <x> <y> <z> <α> <β> <γ>` places grains explicitly — two nodes
with chosen misorientation is a bicrystal with a defined grain
boundary. Polycrystals need MD-scale cells; target LAMMPS (`.lmp`
output), not DFT.

## Verify and report

- Check the assembled cell with the atomsk-structures skill's
  `check_structure.py --fail-below 1.4`: merges and polycrystal seams
  create overlapping atoms, and an optimizer given overlapping atoms
  diverges or welds artifacts. Delete or rebuild instead (a
  `-select ... -rmatom select` pass, or a larger separation).
- Relax the assembly before measuring anything across it.
- Report the orientation relationship (both grains' Miller indices or
  Euler angles), the in-plane strain state, and the run ids. For work
  of adhesion across the interface, hand off to the interface-adhesion
  skill; for the isolated-surface references, the surface-energy skill.

## When not to use this

- Coherent epitaxial interfaces at DFT scale are better built as one
  crystal with a composition change (atomsk-structures plus
  `-substitute` layers) when the lattices match; `--merge` shines when
  the two sides are different structures.
- Amorphous/crystal interfaces: atomsk builds crystals; make the
  amorphous side by melt-quench first, then merge the two files.
- Polycrystal grain-boundary energies from unrelaxed Voronoi cells are
  not physical; anneal before reporting.
