---
name: convergence-study
description: Measure how a DFT result converges with one numerical
  parameter (k-point mesh, plane-wave cutoff) and pick the cheapest
  converged setting for the quantity you will publish. Use before any
  production DFT number, or when asked whether a k-mesh or cutoff is
  sufficient.
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
- Use 5 to 7 rungs spanning cheap to clearly more than enough, so the
  chosen rung has at least two later rungs confirming it.
- Cutoff ladders: `ecutwfc` 30 to 90 Ry for norm-conserving sets, and
  keep the `ecutrho/ecutwfc` ratio fixed (8 for ultrasoft and PAW), or
  the ladder varies two things.
- k-mesh ladders: state the mesh as a reciprocal spacing and stay in one
  grid family (all Γ-centred, or all shifted; do not mix odd and even
  divisions). Insulators converge near 0.30 Å^-1; metals need 0.15 to
  0.10 Å^-1, which for a primitive fcc cell is 20 to 30 divisions, not
  8. A k-ladder is not variational, so a rung can sit closer to the
  reference by accident; the confirming-rungs rule guards that.
- Metals: fix the smearing first (Marzari–Vanderbilt cold smearing at
  0.01 to 0.02 Ry), then converge the mesh at that width. The width and
  the mesh are coupled, and the ladder is valid at one width only.
- Set the SCF threshold well below the convergence threshold per cell
  (`conv_thr` 1e-8 Ry or tighter), or the noise floor sets the verdict.
- Every rung is a `single_point` on the same structure through the same
  protocol expansion, with only the studied key overridden in
  `calculator_options`.

Converge the quantity the production run will use. Energy converges
before forces, and forces before stress. When the product is an energy
difference (an equation of state, a formation energy, a strain ladder)
converge that difference, for example E(1.02 V0) − E(V0), rather than
the total energy, which converges earlier than the differences do.

Write the workflow so it appends one record per rung to a JSON list and
saves it as `conv.json`:

    {"value": 40, "energy": -215.834211, "fmax": 0.0123}

`single_point` returns `info["energy"]` and `info["fmax"]`. For a
pressure ladder, add `"pressure"` in kbar from the calculator's stress.
Add checks: energies finite, and at least 5 rungs recorded.

## 2. Read the ladder

Run the bundled table script:

    python <skill root>/scripts/convergence_table.py conv.json --natoms 2
    python <skill root>/scripts/convergence_table.py conv.json --quantity force

It prints each rung's difference to the final rung (meV, or meV/atom
with `--natoms`; meV/Å for `--quantity force`; kbar for `pressure`) and
names the first rung that stays within the threshold with at least two
later rungs confirming it. Default thresholds: 1 meV/atom, 5 meV/Å,
0.5 kbar; `--threshold` overrides. `--json` gives the same verdict
machine-readable.

## 3. Verify and report

- The final rung is the reference, not the truth. If no rung is
  confirmed, the ladder is too short. Extend it; do not report a
  converged value.
- A ladder converged for energies does not license production stress
  calculations; run the ladder on the quantity you will use.
- Report the chosen setting, the quantity it was converged for, the
  residual difference at that rung, the smearing width for a metal, and
  the run id of the ladder.

## When not to use this

- Comparing energies across different pseudopotential families or codes:
  convergence is per-family, and total energies are not comparable
  across families anyway. Only differences are.
- Metals with very fine Fermi-surface features may need k-meshes beyond
  a naive ladder; watch the smearing width together with the mesh.
