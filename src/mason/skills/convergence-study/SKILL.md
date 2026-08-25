---
name: convergence-study
description: Measure how a DFT result converges with one numerical
  parameter (k-point mesh, plane-wave cutoff) and pick the cheapest
  converged setting. Use before any production DFT number, or when asked
  whether a k-mesh or cutoff is sufficient.
license: MIT
metadata:
  mason-agents: "dft-expert"
---
# Convergence study

A convergence study varies exactly one numerical parameter along a ladder
while everything else stays fixed, and reads the result against the most
accurate rung. Run one workflow per ladder so the whole study is one
traced run.

## 1. Build the ladder

- Vary one parameter per study. A k-mesh ladder holds the cutoff fixed;
  a cutoff ladder holds the k-mesh fixed at a safely dense value.
- Use 4 to 7 rungs spanning cheap to clearly-more-than-enough, for
  example `ecutwfc` 30 to 90 Ry, or k-meshes 2x2x2 to 8x8x8.
- Every rung is a `single_point` on the same structure through the same
  protocol expansion, with only the studied key overridden in
  `calculator_options`.

Write the workflow so it appends one record per rung to a JSON list and
saves it as `conv.json`:

    {"value": 40, "energy": -215.834211}

Add checks: energies finite, and at least 4 rungs recorded.

## 2. Read the ladder

Run the bundled table script:

    python <skill root>/scripts/convergence_table.py conv.json --natoms 2

It prints each rung's difference to the final rung (in meV, or meV/atom
with `--natoms`) and names the first rung that stays within the threshold
(default 1 meV/atom). `--json` gives the same verdict machine-readable.

## 3. Verify and report

- The final rung is the reference, not the truth. If the last two rungs
  still differ by more than the threshold, the ladder is too short -
  extend it; do not report a converged value.
- Energy converges faster than forces, and forces faster than stress. A
  ladder converged for energies does not license production stress
  calculations; run the ladder on the quantity you will use.
- Report the chosen setting, the residual difference at that rung, and
  the run id of the ladder.

## When not to use this

- Comparing energies across different pseudopotential families or codes:
  convergence is per-family, and total energies are not comparable
  across families anyway - only differences are.
- Metals with very fine Fermi-surface features may need k-meshes beyond
  a naive ladder; watch the smearing width together with the mesh.
