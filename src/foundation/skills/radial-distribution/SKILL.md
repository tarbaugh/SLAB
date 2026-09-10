---
name: radial-distribution
description: Compute the radial distribution function g(r) from a
  trajectory or structure file - peak positions, the first minimum, the
  coordination number, solid versus liquid character. Use when asked
  about local structure, nearest neighbor distances, coordination, or
  whether a system melted.
license: MIT
metadata:
  mason-agents: "md-expert analysis-expert"
---
# Radial distribution function

The bundled script computes g(r) with the minimum-image convention from
any ASE-readable file (trajectory or single structure). It needs no
workflow: it reads recorded frames.

## Run

    python <skill root>/scripts/rdf.py md.traj
    python <skill root>/scripts/rdf.py md.traj --skip 200 --species Cu Cu --json

- `--skip N` drops the first N frames. Always skip the unequilibrated
  span for a production g(r); ask the md-expert or the run record how
  long equilibration was. Frames closer than the correlation time add
  no statistics; saving every few hundred steps is enough.
- `--rmax` defaults to just under half the smallest perpendicular cell
  width over all frames, the largest radius the minimum-image convention
  supports. Do not raise it above that; enlarge the cell instead. An NPT
  frame that shrinks below the limit is refused by name.
- `--bins 200` sets the bin width, about 0.05 to 0.08 Å for a metal.
  Compare curves only at equal `--bins` and `--rmax`, because peak
  heights depend on the bin width.
- `--species A B` restricts to A-B pairs (order does not matter). The
  normalisation counts the ideal pairs exactly, self-pairs excluded, so
  g(r) tends to 1 for a same-species pair in a small cell too.

The output is a text table of r and g(r), then the first peak (the first
local maximum above 1, not the highest peak) with a block standard error
of its height, the first minimum after it, the coordination number up
to that minimum, and the mean of g over the last tenth of r; `--json`
returns the histogram and the same numbers machine-readable.

## Read the result

- A crystal shows sharp peaks separated by zeros; a liquid shows one
  broad first peak, damped oscillations, and g(r) -> 1; a gas is
  structureless. The contrast is the usual melted-or-not evidence, and
  the ratio of g at the first minimum to g at the first peak is the
  number to quote (near 0 for a crystal, 0.2 to 0.5 for a liquid).
- The first-peak position is the nearest-neighbor distance. For fcc it
  sits at a/sqrt(2) with coordination 12; checking that against the
  known lattice constant validates the whole pipeline.
- The tail must average 1. The script warns when it does not, which
  means too few atoms or frames for the bins, or a cell that is not in
  equilibrium. The first-peak height converges only with about a
  thousand uncorrelated pair samples per bin.

## When not to use this

- Fewer than a few dozen atoms per frame gives a noisy g(r); average
  over many frames or enlarge the cell.
- Strongly triclinic cells: the minimum-image distance used here is
  exact only up to half the smallest perpendicular width. The script
  caps `--rmax` accordingly and says so; a Niggli-reduced cell allows a
  larger radius.
- Cells above 2000 atoms take a neighbour-list path; the result is the
  same, the memory is not the full pair matrix.
