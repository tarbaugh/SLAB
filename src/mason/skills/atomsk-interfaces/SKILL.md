---
name: atomsk-interfaces
description: Assemble interfaces, bicrystals, and polycrystals with
  atomsk — stack two slabs into an interface supercell, build grain
  boundaries from oriented grains, generate Voronoi polycrystals — with
  lattice matching, overlap removal, and the translation scan a
  boundary energy needs. Use when asked to build an interface, a grain
  boundary, a bicrystal, or a polycrystalline cell.
license: MIT
metadata:
  mason-agents: "dft-expert md-expert"
---
# Interfaces and polycrystals with atomsk

Two atomsk modes assemble multi-grain systems: `--merge` stacks
existing structures, `--polycrystal` fills a box with misoriented
grains. Run both through `foundation.tasks.build_structure` (see the
atomsk-structures skill for the calling pattern).

Two option forms differ between atomsk builds. Release 0.13.1 writes
`--merge z 2 ...` and `-cell add 15 z`; the development branch writes
`--merge stack z 2 ...` and `-cell z add 15`. The release forms are
given below. On a wrong form atomsk prints the order your build
expects, so try once and switch.

## 1. Interfaces and bicrystals: --merge

    atomsk --merge z 2 bottom.xsf top.xsf interface.xsf

stacks two files along z into one cell. The workflow:

1. Build each slab separately, oriented so the intended interface plane
   is normal to the stacking axis (atomsk-structures skill).
2. Match the in-plane lattices. Search for a supercell pair with the
   Zur–McGill method (pymatgen's `ZSLGenerator`: area up to 400 Å^2,
   length mismatch 3%, angle mismatch 0.01 by default) and keep the
   residual strain under about 2% for a reported energy. Strain the
   softer slab, or split the strain in inverse proportion to the bulk
   moduli when both are thin (`-deform`, or rebuild at the strained
   lattice constant). Record which slab carries how much strain; the
   interface-adhesion skill needs the same strain in its reference slabs.
3. Merge with `--merge z 2 ...` and stage both slabs via `inputs=`.
   The merged cell is the sum of the two lengths along the stacking
   axis only; in-plane the box is the first file's. A second slab with a
   larger in-plane cell overlaps itself silently, so match the in-plane
   cells before merging.
4. Set the interface gap. `-shift` before the merge does not change the
   gap, because the separation is the bottom slab's cell length. Add
   the gap to the bottom slab first (`-cell add 2.0 z`), or after the
   merge shift the atoms atomsk tagged as the second system:
   `-select prop sysID 2 -shift 0 0 2.0`.
5. Remove overlapping atoms at the seam with `-remove-doubles <d>`,
   which deletes atoms of the second system closer than `d` to an atom
   of the first. Use about 0.6 of the nearest-neighbour distance
   (1.4 Å for Cu or Al).
6. For a free-standing interface with vacuum, enlarge the cell along
   the stacking axis afterwards: `-cell add 15 z`.

A periodic cell built this way holds two interfaces unless vacuum
separates the outer faces; count them in every energy per area.

## 2. Grain boundaries: two oriented crystals

A grain boundary with a defined misorientation is two oriented crystals
merged along the boundary normal, not two Voronoi nodes:

    atomsk --create fcc 4.046 Al orient [1-10] [11-2] [111] grain1.xsf
    atomsk --create fcc 4.046 Al orient [-110] [11-2] [111] grain2.xsf
    atomsk --merge y 2 grain1.xsf grain2.xsf gb.xsf

The mirrored first index makes a symmetric tilt boundary; `-rotate` and
`-orthogonal-cell` build an arbitrary angle. The periodic cell holds two
boundaries. A boundary energy needs the minimum over the rigid-body
translations: scan the in-plane shifts of one grain (a 20 by 20 grid
over the boundary's repeat cell is typical), remove doubles at two or
three cutoffs, minimise every candidate, and report

    gamma_GB = (E_bicrystal - N * E_bulk) / (2 * A)

for the lowest, with the grain thickness converged (grains of several
nanometres, and a check at one larger thickness).

## 3. Polycrystals: --polycrystal

    atomsk --polycrystal fcc-al.xsf poly.txt polycrystal.lmp

`fcc-al.xsf` is the seed (one unit cell) and `poly.txt` is a parameter
file; stage both through `inputs=` — an `Atoms` value becomes a
structure file, a string value is written verbatim:

    cell, info = build_structure(
        "--polycrystal fcc-al.xsf poly.txt polycrystal.lmp -wrap -remove-doubles 1.4",
        inputs={"fcc-al.xsf": seed, "poly.txt": "box 200 200 200\nrandom 12\n"},
        output="polycrystal.lmp",
    )

The parameter file declares the box and the grains:

    box 200 200 200
    random 12

gives twelve randomly placed, randomly oriented grains in a 200 Å box.
`node <x> <y> <z> <α> <β> <γ>` places grains explicitly. Voronoi seams
between arbitrary rotations are not periodic boundaries with a defined
structure; use section 2 for those. Polycrystals need MD-scale cells;
target LAMMPS (`.lmp` output), not DFT. Atomsk writes
`<output>_param.txt` with the nodes it generated; keep it as the
artifact that rebuilds the same cell, because `random` differs between
invocations.

## Verify and report

- Check the assembled cell with the atomsk-structures skill's
  `check_structure.py --expect-atoms N`: merges and polycrystal seams
  create overlapping atoms, and an optimizer given overlapping atoms
  diverges or welds artifacts. Remove doubles or rebuild instead.
- Relax the assembly before measuring anything across it.
- Report the orientation relationship (both grains' Miller indices or
  Euler angles), the in-plane strain state (which slab, how much), the
  number of interfaces in the cell, and the run ids. For work of
  adhesion across the interface, hand off to the interface-adhesion
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
