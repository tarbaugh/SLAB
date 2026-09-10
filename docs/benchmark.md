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

The verdict is an outcome, not a defect. After the verdict, the review
reads the same evidence and raises flags: attributable defects, each
naming the skill, card, or tool a revision edits. The flags travel in the
record beside the verdict. [The science review](review.md) describes the
evaluators, the flag statuses, and the gate a skill revision must pass.

The band depends on the engine class, which the scorer reads from the
cited runs' task recipes, never from the report's prose. A run under
Quantum ESPRESSO (or a registry alias built on SLAB's `qe` factory) is
judged against the reference for the functional it used, `pbe` or
`pbesol`. The scorer reads the functional from the traced calculator
options: `input_dft` when set, else the pseudopotential family name,
else the pseudopotential file names. The SSSP families that SLAB installs
by default are PBEsol, and PBEsol binds copper about 2% tighter than PBE,
so one reference for both functionals would fail a correct answer.

Everything else (served MLIP checkpoints, classical potentials, EMT) is
`mlip`. It gets the wider band around the PBE value, because those engines
are not expected to reproduce DFT. Experiment is listed so the reader
sees the known distance between the functionals and the measurement.

A DFT run whose functional cannot be read, or cited runs that mix
functionals, fail with that reason. A class with no checked reference yet
(PBEsol for question 3) is refused rather than judged, and the refusal
says which value is missing.

## The questions

The table is rendered from the code (`slab benchmark tables`), so the
instruction the agent receives and the band it is judged by cannot
drift from this page.

<!-- benchmark:questions:start -->
| # | Instruction to the agent | Reference (PBE) | Reference (PBEsol) | Experiment | Passes when | Skills |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Determine the equilibrium lattice constant of fcc Cu from an equation of state. Finish with results key `a0` in Å, and the run ids that produced it. | 3.632 Å | 3.562 Å | 3.615 Å | a0 within ±0.03 (DFT, against its functional's reference) or ±0.05 (MLIP, classical, EMT, against PBE) Å | equation-of-state |
| 2 | Compute the Cu(111) surface energy. Finish with results key `gamma_111` in J/m^2, and the run ids that produced it. | 1.33 J/m^2 | 1.59 J/m^2 | 1.79 ± 0.19 J/m^2 (polycrystalline average) | gamma_111 within ±0.2 (DFT, against its functional's reference) or ±0.4 (MLIP, classical, EMT, against PBE) J/m^2 | surface-energy |
| 3 | Compute the monovacancy formation energy in fcc Cu. Finish with results key `e_vac` in eV, and the run ids that produced it. | 1.07 eV | no checked value yet | 1.29 ± 0.02 eV | e_vac within ±0.15 (DFT, against its functional's reference) or ±0.3 (MLIP, classical, EMT, against PBE) eV | atomsk-defects, convergence-study |
| 4 | Estimate the melting point of Cu with the two-phase method under the served MLIP. Finish with results key `t_melt` in K, and the run ids that produced it. | 1357.77 K | 1357.77 K | 1357.77 K | t_melt within ±100 (DFT, against its functional's reference) or ±100 (MLIP, classical, EMT, against PBE) K | two-phase-melting, melt-quench |
| 5 | Fine-tune a GRACE potential on DFT labels for strained fcc Cu and validate it against held-out DFT single points. Finish with results key `energy_rmse` in meV/atom, `force_rmse` in eV/Å, and the run ids that produced them. | internal to the campaign | internal to the campaign | — | energy_rmse ≤ 5 meV/atom; force_rmse ≤ 0.1 eV/Å | mlip-training |
<!-- benchmark:questions:end -->

## Three harness conditions

The same question runs under three harnesses on the same model, so the
gain from runtime-enforced verification is a measured number. A
condition is an agent card plus a set of mechanism switches.
`--condition` on `run`, `launch`, and `render` selects one, and the
record carries it. The table is rendered from the code.

<!-- benchmark:conditions:start -->
| Condition | Card | Mechanisms on | What it is |
| --- | --- | --- | --- |
| `slab` | `pi` | `adaptive-effort`, `budget-hint`, `check-gating`, `context-hygiene`, `critic-gate`, `delegation`, `failure-records`, `identical-result-annotation`, `looking-hint`, `machine-memory`, `skills` | Mason as it is: the PI card, every mechanism on, verification gated in code. |
| `protocol` | `protocol` | `adaptive-effort`, `budget-hint`, `context-hygiene`, `identical-result-annotation`, `skills` | The skill collection with a file protocol: scripts run with the shell, an append-only provenance log in the project, verification is what the agent writes down. No run tools, no failure records, no critic, no memory. |
| `bare` | `bare` | none | The model with read, write, shell, and finish, a one-paragraph prompt, and no mechanism at all. |
<!-- benchmark:conditions:end -->

The `slab` condition is Mason as it is. The `protocol` condition has the
shape of a skill collection with a file protocol, the shape of the AICC
control plane. The agent has the skill catalog, the file tools, and a
shell. It runs a calculation as `python script.py` with no run context,
and it keeps an append-only provenance log in the project. Verification
is what the agent writes down. The `bare` condition is the model with
read, write, shell, and finish, and a one-paragraph prompt.

The scorer does not change with the condition. A campaign passes only
when the finish call cites runs that reached `verified`. The `protocol`
and `bare` harnesses create no runs, so a campaign under them passes only
when the agent finds the traced path on its own, for example by running
its script with `slab run` and declaring a check. That asymmetry is the
measurement. A correct number in a provenance log fails, because nothing
but the agent's own note stands behind it. The record says which
condition ran and which mechanisms were on, so a table row never mixes
harnesses.

Run one question under one condition:

```bash
slab benchmark run 1 --condition protocol --machine laptop
```

Render the whole grid for a cluster without submitting anything:

```bash
slab benchmark matrix --partition gpu
```

`matrix` writes one directory per cell under `sandbox/matrix/`, three
conditions by five questions by default, and lists the cells in
`matrix.json`. `--condition` and `--question` narrow the grid. Each
`--without <mechanism>` adds one row of the `slab` condition with that
mechanism switched off, for every question in the grid. Submit each
script with `sbatch`, and score the campaigns when the jobs end. The
results table keys its rows by model, machine, and condition, and an
ablated row reads `slab -budget-hint`.

## The mechanism ledger

Every mechanism the harness runs is a switch, so the ablation grid can
measure it. The table is rendered from the code. The evidence column
names what the mechanism rests on now. The measured effect enters this
page when the grid has run.

<!-- benchmark:mechanisms:start -->
| Switch | What it does | Evidence |
| --- | --- | --- |
| `check-gating` | Calculations run as traced workflow scripts through launch_workflow, wait_for_run, list_runs, show_run, and read_artifact; a run whose checks pass is verified, and the prompt teaches that path. Off, the shell is the only way to run a script. | SLAB's verification gate (ARCHITECTURE.md); a number without a verified run is a rumor. |
| `failure-records` | A failed run's structured failure record (trimmed traceback and diagnostic notes) is returned with the run, and the prompt asks for a diagnosis before a retry. Off, a failed run reports its status only. | Reflexion (arXiv:2303.11366); Manus, keep failures in context. |
| `critic-gate` | The review tool reaches a read-only critic, and a card that reviews first spends no compute before the critic approves the plan. | Agent Laboratory (arXiv:2501.04227): fixed tool libraries with checkpoints beat full autonomy. |
| `machine-memory` | Facts a session learned about this machine persist through remember and enter later prompts through recall and the memory catalog. | MemGPT (arXiv:2310.08560); an overnight job that trips on a quirk at 03:00 should not trip on it twice. |
| `context-hygiene` | Old tool results are cleared to placeholders once the prompt is large, and superseded plan echoes are folded, before compaction. | SWE-agent and OpenHands: masking old observations matches summarization at half the cost; Anthropic's clear_tool_uses. |
| `identical-result-annotation` | A tool result identical to the same call's previous result carries a note the model reads as evidence, escalating with the repeat count. | A real 240-call session spent 82 calls on one byte-identical readelf pipeline. |
| `budget-hint` | An ephemeral step-of-budget line follows every request, stricter near the ceiling. | Transcripts stopped at the call budget mid-inquiry with nothing written down; the hint moved the finish earlier. |
| `looking-hint` | After fifteen consecutive steps that only read and listed, the budget line tells the model to step back, and again every five steps. Off, a look-only loop runs to the turn budget. | One campaign on 2026-09-03 spent 72 minutes and about a hundred model calls in a look-only run; the science review's no-progress-loop rule reads the same shape afterwards. |
| `skills` | The skill tool loads procedures and tested scripts from the catalog in the Agent Skills format, one line per skill until loaded. | Anthropic, Agent Skills; the skills audit of 2026-09-03. |
| `delegation` | A lead hands a separable task to a specialist card that runs its own loop one level down and returns a report. | Anthropic's multi-agent research system: context isolation pays for separable subtasks only. |
| `adaptive-effort` | A reply cut at the token budget is retried once at lower effort with a request for brevity before the turn ends. | Transcripts where a high-effort reply was cut twice and the turn ended with no report. |
<!-- benchmark:mechanisms:end -->

`[agent] mechanisms` in `slab.toml` lists the switches a session runs
with, and unset means every one. A name outside the ledger is refused.
The older flags `memory`, `delegation`, and `clear_tool_results` still
apply to their mechanisms. Plumbing has no switch: JSON repair,
truncation detection, and the effort field are necessary, and they are
not findings.

## How to run a campaign

Off-cluster, on a laptop or an interactive node, one command runs a
campaign in this process, scores it, and appends the record:

```bash
slab benchmark run 1 --machine laptop
```

Recorded from a real laptop campaign, Llama 3.1 8B served by Ollama with
a 32k-token context:

<!-- no-verify -->
```text
session 20260902-032612-15602: stopped by finish after 18 step(s)
Q1 a0        llama3.1:8b-32k          laptop       fail: the finish carried no structured results
recorded in /private/tmp/you/demo/benchmarks/results.jsonl
```

The model called `finish` without the `a0` result key, so the campaign
fails at the first condition, and the record says so. The `llama3.1:8b`
row in the table below is the same model under Ollama's default
2,048-token context, which truncates Mason's prefix before the model
sees its instructions; it answered in prose and never called `finish`.
Both rows stay, because both happened. Check the context length before a
laptop campaign (`slab mason doctor` reports it; the Mason tutorial gives
the fix). Those are the rows a larger model has to beat.

On a cluster, submit the campaign as a sandbox job, then score it after
the job ends. `score` finds every session whose opening message is a
benchmark instruction and scores the ones not yet recorded:

```bash
slab benchmark launch 1 --partition gpu
slab hpc status <job id>
slab benchmark score --machine <label>
```

`launch` always renders fresh, so a hand edit to the job script does not
survive it. For a tweak the config cannot express, render the job files
without submitting, edit the script, and submit it yourself:

```bash
slab benchmark render 1 --partition gpu
sbatch sandbox/mason-sandbox.sbatch
```

Rules for the record:

- `--machine` is a label you choose for the machine, such as `laptop` or
  `cluster-a`. Never a hostname. The default is the compute profile the
  session ran under.
- `--condition` names the harness arm: `slab`, `protocol`, or `bare`.
  The default is `slab`. `score` reads the condition from the transcript,
  and a transcript from before the switches existed scores as `slab`.
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

Then rewrite the tables on this page (the questions, the conditions, the
mechanism ledger, the results, and the flags) and the summary in the
README:

```bash
slab benchmark tables
```

## Results

<!-- benchmark:results:start -->
| Model | Machine | Condition | Q1 a0 | Q2 surface | Q3 vacancy | Q4 melting | Q5 finetune | Passed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama3.1:8b | laptop | slab | fail (no finish report) | — | — | — | — | 0/5 |
| llama3.1:8b-32k | laptop | slab | fail (the finish carried no structured results) | — | — | — | — | 0/5 |
<!-- benchmark:results:end -->

A row is one model on one machine under one condition. A cell shows
the reported value beside the verdict, a failure names its reason, and a
reviewed campaign shows how many flags it raised. Score the
same five questions again after every change to the prompt, the tools,
the roster, or the skills. A change that lowers a model's total is a
regression, whatever else it improves.

## Flags

The flags on the latest record per model, machine, and question. A flag
is `open` while its target is unchanged, `pending` when the skill has a
newer revision no campaign has run under, and `unknown` when the record
carried no digest. `slab benchmark gate <skill>` says whether the
benchmark validates a skill revision.

<!-- benchmark:flags:start -->
No flag has been raised on a recorded campaign.
<!-- benchmark:flags:end -->

## Reference values and their sources

- Lattice constant. Experiment 3.615 Å at room temperature (Kittel,
  *Introduction to Solid State Physics*, 8th ed., Table 4, lists 3.61 Å;
  the zero-point-corrected value is 3.603 Å). PBE 3.632 Å (Haas, Tran,
  and Blaha, *Phys. Rev. B* 79, 085104 (2009), Table I). PBEsol 3.562 Å
  (Csonka et al., *Phys. Rev. B* 79, 155107 (2009), Table II, the
  BAND/LCAO column; that table's PBE value is 3.628 Å, consistent with
  Haas et al.).
- Surface energy. Experiment 1.79 ± 0.19 J/m² for the polycrystalline
  average (Tyson and Miller, *Surf. Sci.* 62, 267 (1977), and Miedema,
  *Z. Metallkd.* 69, 287 (1978), as tabulated in Patra et al., *Proc.
  Natl. Acad. Sci.* 114, E9188 (2017), Table 1; de Boer et al.,
  *Cohesion in Metals* (1988), gives 1.83 J/m²). PBE Cu(111) 1.33 J/m²
  and PBEsol Cu(111) 1.59 J/m² (Patra et al. 2017, Table 2).
  Facet-resolved experimental values do not exist, so the reference is
  the facet value for each functional and experiment is the average.
- Vacancy formation energy. Experiment 1.29 ± 0.02 eV from positron
  annihilation (Triftshäuser and McGervey, *Appl. Phys.* 6, 177 (1975)).
  PBE 1.07 eV with relaxation (Angsten et al., *New J. Phys.* 16, 015018
  (2014), Table A.1). PBEsol: no checked value yet. Two papers tabulate
  one (Delczeg et al., *Phys. Rev. B* 80, 205121 (2009); Medasani et
  al., *Comput. Mater. Sci.* 101, 96 (2015)), but neither text was
  reachable when this page was checked. Until someone reads one of them
  and enters the value with its source, a PBEsol campaign on question 3
  is refused, not scored.
- Melting point. Experiment 1357.77 K, 1084.62 °C (NIST Chemistry
  WebBook; *CRC Handbook of Chemistry and Physics*, "Properties of the
  elements"). The reference is experiment for every engine class, so a
  classical potential or EMT fails this question, which is the truthful
  outcome.
- Fine-tuned potential. The held-out DFT set is generated in the
  campaign itself, so the reference is internal to the run record.

Confirmation record: every value above was checked against its cited
source on 2026-09-01 (the PNAS values from the published Tables 1 and 2;
the PBEsol lattice constant from the arXiv 0903.4037 text of Csonka et
al.; the NJP value from the published Table A.1). A reference nobody has
checked is not a reference, so re-check before changing a value.
