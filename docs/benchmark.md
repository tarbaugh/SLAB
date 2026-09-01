# The benchmark campaigns

This page fixes five research questions with known answers. Mason runs
each one as a campaign, per model and per machine, and the result is a
score. The score turns "the agent works" into a number, and it gives
every harness change a before and an after. The next builder or tool is
chosen by a measured gap in these campaigns, not by what is interesting
to build.

## The rule

Each campaign starts from one instruction, given to the PI card with
`slab mason sandbox launch` (or `slab mason run --auto` off-cluster). The
campaign ends when the agent finishes or the job's time limit stops it.
`slab mason report` produces the digest that is scored.

A campaign scores from 0 to 4, one point per criterion:

| Point | Criterion |
| --- | --- |
| Completed | The agent finished with a report that cites at least one run id. |
| Verified | Every run the report cites reached `verified`. Numbers from unverified or failed runs count as absent. |
| Correct | The reported value lies inside the tolerance band for the engine the agent used. |
| Honest | The report names the engine and its fidelity, and states the smoke-test caveat when the compute profile requires it. |

The tolerance band depends on the engine. A universal MLIP or a classical
potential is not expected to reproduce experiment, so the reference for
those engines is the accepted DFT-PBE value. The reference for a DFT run
is the same DFT-PBE value, and experiment is listed so the reader sees
the known distance between the two. A campaign that reports a number
with no run id scores 0 whatever the number is.

The set is deliberately small and all copper, so the same structures,
the same engines, and the same k-mesh conventions carry across the five
questions, and a failure isolates to the harness rather than to the
chemistry.

## The five questions

| # | Instruction to the agent | Reference (DFT-PBE) | Experiment | Tolerance | Skill exercised |
| --- | --- | --- | --- | --- | --- |
| 1 | "Determine the equilibrium lattice constant of fcc Cu from an equation of state." | 3.63 Å | 3.615 Å | ±0.03 Å (DFT), ±0.05 Å (MLIP or EMT) | equation-of-state |
| 2 | "Compute the Cu(111) surface energy." | 1.3 J/m² | 1.79 J/m² (polycrystalline average) | ±0.2 J/m² (DFT), ±0.4 J/m² (MLIP or EMT) | surface-energy |
| 3 | "Compute the monovacancy formation energy in fcc Cu." | 1.1 eV | 1.28 eV | ±0.15 eV (DFT), ±0.3 eV (MLIP or EMT) | atomsk-defects, convergence-study |
| 4 | "Estimate the melting point of Cu with the two-phase method under the served MLIP." | none: the reference is the engine's own two-phase result | 1358 K | ±100 K against experiment for an MLIP, reported with the caveat that classical potentials and EMT miss by more | two-phase-melting, melt-quench |
| 5 | "Fine-tune a GRACE potential on DFT labels for strained fcc Cu and validate it against held-out DFT single points." | none: the validation set is the reference | none | energy RMSE ≤ 5 meV/atom and force RMSE ≤ 0.1 eV/Å on the held-out set | mlip-training |

Reference values and their sources:

- Lattice constant. Experiment 3.615 Å at room temperature (Kittel,
  *Introduction to Solid State Physics*, 8th ed., Table 4). PBE 3.63 Å
  (Haas, Tran, and Blaha, *Phys. Rev. B* 79, 085104 (2009), Table I).
- Surface energy. Experiment 1.79 J/m² for the polycrystalline average
  (Tyson and Miller, *Surf. Sci.* 62, 267 (1977)). PBE Cu(111) 1.3 J/m²
  (Tran et al., *Sci. Data* 3, 160080 (2016), the Materials Project
  surface-energy set).
- Vacancy formation energy. Experiment 1.28 eV from positron annihilation
  (Landolt-Börnstein, Group III, Vol. 25, "Atomic defects in metals").
  PBE 1.1 eV with relaxation (Mattsson and Mattsson, *Phys. Rev. B* 66,
  214110 (2002), Table II).
- Melting point. Experiment 1357.77 K (*CRC Handbook of Chemistry and
  Physics*, "Properties of the elements").
- Fine-tuned potential. The held-out DFT set is generated in the
  campaign itself, so the reference is internal to the run record.

Confirm each value against the cited source before the first scored
campaign, and record the confirmation in this page. A reference nobody
has checked is not a reference.

## What a campaign must leave behind

- The `slab mason report --json` digest, saved beside the score.
- The promoted run for the final number, with its checks and artifacts.
- The scoring line: model, machine, engine, score out of 4, and the
  reported value beside the reference.

Score the same five questions again after every change to the prompt,
the tools, the roster, or the skills. A change that lowers the total is
a regression, whatever else it improves.
