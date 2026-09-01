# The benchmark campaigns

This page fixes five research questions with known answers. Mason runs
each one as a campaign, per model and per machine, and `slab benchmark`
scores the result. The score answers one question: did the agent achieve
a correct answer? That turns "the agent works" into a number, and it gives
every change to the prompt, the tools, the roster, or the skills a before
and an after. The next builder or tool is chosen by a measured gap in
these campaigns, not by what is interesting to build.

## The rule

A campaign is one Mason session that starts from one fixed instruction.
The instruction is the science question plus a reporting clause that
names the result key the agent must fill in its `finish` call. The
campaign ends when the agent finishes or the job's time limit stops it.

A campaign **passes** when both conditions hold:

1. The `finish` call carries the result key with a numeric value inside
   the tolerance band for the engine class the agent used.
2. Every run the `finish` call cites reached `verified` (or was promoted
   from there).

The second condition is what makes "correct" mean *computed*. A number
with no verified run behind it cannot pass, however close it lies to the
reference. A campaign that fails records why: no finish report, no
structured result, a cited run that does not exist or never verified, or
a value outside the band.

The band depends on the engine class, which the scorer reads from the
cited runs' task recipes, never from the report's prose. A run under
Quantum ESPRESSO (or a registry alias built on SLAB's `qe` factory) is
`dft`; everything else (served MLIP checkpoints, classical potentials,
EMT) is `mlip` and gets the wider band, because those engines are not
expected to reproduce DFT. The reference is the accepted DFT-PBE value
for both classes; experiment is listed so the reader sees the known
distance between the two.

## The questions

The table is rendered from the code (`slab benchmark render`), so the
instruction the agent receives and the band it is judged by cannot
drift from this page.

<!-- benchmark:questions:start -->
| # | Instruction to the agent | Reference (DFT-PBE) | Experiment | Passes when | Skills |
| --- | --- | --- | --- | --- | --- |
| 1 | Determine the equilibrium lattice constant of fcc Cu from an equation of state. Finish with results key `a0` in Å, and the run ids that produced it. | 3.632 Å | 3.615 Å | a0 within ±0.03 (DFT) or ±0.05 (MLIP, classical, EMT) Å | equation-of-state |
| 2 | Compute the Cu(111) surface energy. Finish with results key `gamma_111` in J/m^2, and the run ids that produced it. | 1.33 J/m^2 | 1.83 J/m^2 (polycrystalline average) | gamma_111 within ±0.2 (DFT) or ±0.4 (MLIP, classical, EMT) J/m^2 | surface-energy |
| 3 | Compute the monovacancy formation energy in fcc Cu. Finish with results key `e_vac` in eV, and the run ids that produced it. | 1.07 eV | 1.29 ± 0.02 eV | e_vac within ±0.15 (DFT) or ±0.3 (MLIP, classical, EMT) eV | atomsk-defects, convergence-study |
| 4 | Estimate the melting point of Cu with the two-phase method under the served MLIP. Finish with results key `t_melt` in K, and the run ids that produced it. | 1357.77 K | 1357.77 K | t_melt within ±100 (DFT) or ±100 (MLIP, classical, EMT) K | two-phase-melting, melt-quench |
| 5 | Fine-tune a GRACE potential on DFT labels for strained fcc Cu and validate it against held-out DFT single points. Finish with results key `energy_rmse` in meV/atom, `force_rmse` in eV/Å, and the run ids that produced them. | internal to the campaign | — | energy_rmse ≤ 5 meV/atom; force_rmse ≤ 0.1 eV/Å | mlip-training |
<!-- benchmark:questions:end -->

## How to run a campaign

Off-cluster, on a laptop or an interactive node, one command runs a
campaign in this process, scores it, and appends the record:

```bash
slab benchmark run 1 --machine laptop
```

Recorded from a real laptop campaign, Llama 3.1 8B served by Ollama:

<!-- no-verify -->
```text
session 20260901-234154-2590: stopped by answer after 54 step(s)
Q1 a0        llama3.1:8b              laptop       fail: no finish report
recorded in /private/tmp/you/demo/benchmarks/results.jsonl
```

The model answered in prose instead of calling `finish`, so the campaign
fails at the first condition, and the record says so. That is the row a
larger model has to beat.

On a cluster, submit the campaign as a sandbox job, then score it after
the job ends. `score` finds every session whose opening message is a
benchmark instruction and scores the ones not yet recorded:

```bash
slab benchmark launch 1 --partition gpu
slab hpc status <job id>
slab benchmark score --machine <label>
```

Rules for the record:

- `--machine` is a label you choose for the machine, such as `laptop` or
  `cluster-a`. Never a hostname. The default is the compute profile the
  session ran under.
- The model comes from the transcript, which names the model that
  answered. `--model` overrides it for transcripts written before that
  header existed.
- Records append to `benchmarks/results.jsonl` in the project directory,
  one JSON line per scored campaign. Commit that file to carry results
  from a cluster into the repository.
- `slab benchmark list` prints the questions with their result keys and
  bands. `slab benchmark score --rescore` scores a recorded session
  again; the renderer keeps the latest record per model, machine, and
  question.

Then render the tables on this page and the summary in the README:

```bash
slab benchmark render
```

## Results

<!-- benchmark:results:start -->
| Model | Machine | Q1 a0 | Q2 surface | Q3 vacancy | Q4 melting | Q5 finetune | Passed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama3.1:8b | laptop | fail (no finish report) | — | — | — | — | 0/5 |
<!-- benchmark:results:end -->

A cell shows the reported value beside the verdict, and a failure names
its reason. Score the same five questions again after every change to
the prompt, the tools, the roster, or the skills. A change that lowers a
model's total is a regression, whatever else it improves.

## Reference values and their sources

- Lattice constant. Experiment 3.615 Å at room temperature (Kittel,
  *Introduction to Solid State Physics*, 8th ed., Table 4, lists 3.61 Å;
  the zero-point-corrected value is 3.603 Å). PBE 3.632 Å (Haas, Tran,
  and Blaha, *Phys. Rev. B* 79, 085104 (2009), Table I).
- Surface energy. Experiment 1.83 J/m² for the polycrystalline average
  (de Boer et al., *Cohesion in Metals* (1988), as tabulated in Patra et
  al., *Proc. Natl. Acad. Sci.* 114, E9188 (2017), Table I; Tyson and
  Miller, *Surf. Sci.* 62, 267 (1977) is the other standard
  compilation). PBE Cu(111) 1.33 J/m² (Patra et al. 2017, Table II).
  Facet-resolved experimental values do not exist, so the reference is
  the PBE facet value and experiment is the average.
- Vacancy formation energy. Experiment 1.29 ± 0.02 eV from positron
  annihilation (Triftshäuser and McGervey, *Appl. Phys.* 6, 177 (1975)).
  PBE 1.07 eV with relaxation (Angsten et al., *New J. Phys.* 16, 015018
  (2014), Table A.1).
- Melting point. Experiment 1357.77 K, 1084.62 °C (NIST Chemistry
  WebBook; *CRC Handbook of Chemistry and Physics*, "Properties of the
  elements"). The reference is experiment for every engine class, so a
  classical potential or EMT fails this question, which is the truthful
  outcome.
- Fine-tuned potential. The held-out DFT set is generated in the
  campaign itself, so the reference is internal to the run record.

Confirmation record: every value above was checked against its cited
source on 2026-09-01 (the PNAS values from the arXiv 1702.08515 text;
the NJP value from the published Table A.1). A reference nobody has
checked is not a reference, so re-check before changing a value.
