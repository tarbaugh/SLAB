---
name: radial-distribution
description: Compute the radial distribution function g(r) from a
  trajectory or structure file - peak positions, coordination, solid
  versus liquid character. Use when asked about local structure, nearest
  neighbor distances, or whether a system melted.
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
  long equilibration was.
- `--rmax` defaults to just under half the smallest cell width, the
  largest radius the minimum-image convention supports. Do not raise it
  above that; enlarge the cell instead.
- `--species A B` restricts to A-B pairs (order does not matter).

The output is a text table of r and g(r), with the first-peak position
and height summarized at the end; `--json` returns the histogram
machine-readable.

## Read the result

- A crystal shows sharp peaks separated by zeros; a liquid shows one
  broad first peak, damped oscillations, and g(r) -> 1; a gas is
  structureless. The contrast is the usual melted-or-not evidence.
- The first-peak position is the nearest-neighbor distance. For fcc it
  sits at a/sqrt(2); checking that against the known lattice constant
  validates the whole pipeline.
- Peak heights depend on the bin width. Compare curves only at equal
  `--bins` and `--rmax`.

## When not to use this

- Fewer than a few dozen atoms per frame gives a noisy g(r); average
  over many frames or enlarge the cell.
- Strongly triclinic cells: the minimum-image distance used here is
  exact only up to half the smallest perpendicular width. The script
  caps `--rmax` accordingly and says so.
