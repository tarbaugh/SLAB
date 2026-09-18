# The roster and skills

Mason is a research group, not a single agent. The group is the roster: a
set of agents defined as markdown files called agent cards. The default
agent is `pi`, the principal investigator. The PI can hand a scoped task
to a specialist with the `delegate` tool, and a plan to the critic with
the `review` tool. Every agent can load skills: reusable procedure
packages with tested analysis scripts.

This page shows how to use the roster, how to write a card and a skill,
and how to configure a different model per agent. The recorded outputs
are exact captures from real executions against a local Ollama.

## The roster

Ten cards ship built in: two leads, `pi` and `planner`, three
specialists, a `worker`, a helper, `coding-expert`, a `critic`, and two
condition cards, `protocol` and `bare`, that [the benchmark](../benchmark.md)
runs as harness arms. `slab mason roster` lists what is visible
from the current project, with the layer each card came from and the
model it would use:

```bash
slab mason roster
```

```text
pi                 built-in  llama3.1:8b                  20 skill(s)  [delegates]
analysis-expert    built-in  llama3.1:8b                  9 skill(s)
bare               built-in  llama3.1:8b                  0 skill(s)  [own prompt]
coding-expert      built-in  llama3.1:8b                  20 skill(s)  [helper]
critic             built-in  llama3.1:8b                  20 skill(s)  [reviews]
dft-expert         built-in  llama3.1:8b                  10 skill(s)
md-expert          built-in  llama3.1:8b                  15 skill(s)
planner            built-in  llama3.1:8b                  20 skill(s)  [delegates, review first]
protocol           built-in  llama3.1:8b                  20 skill(s)  [own prompt]
worker             built-in  llama3.1:8b                  20 skill(s)
```

Each agent runs the same harness with a different role prompt, its own
tool allowlist, and its own slice of the skill catalog. The
`analysis-expert` card, for example, cannot launch new physics: its
allowlist omits `launch_workflow` and the SLURM tools, so the doctrine
"work from recorded evidence" is enforced in code.

`slab mason chat` and `mason run` start as the PI. Pick another entry agent
with `--agent`:

```bash
slab mason run --agent dft-expert "expand the balanced protocol for this cell and explain each cutoff"
```

An unknown name fails and lists the roster.

## Delegation

The PI has the `delegate(agent, task, context?, continues?, steps?,
effort?)` tool. It runs the named specialist's own tool loop against the
shared workspace and returns the specialist's final report. The rules are
code, not prompt text:

- The tree is at most three levels deep: a lead, a specialist, a helper.
  A lead may brief anyone on its team, helpers included. A specialist
  may brief helpers only, one at a time, as [Helpers](#helpers)
  describes. A helper never has the `delegate` tool, whatever its card
  says.
- A card that delegates is a lead, not a hand. The `pi` and the
  `planner` never appear on each other's team, and a brief sent to one
  is refused with the team named.
- A card that reviews takes no briefs. A brief sent to the `critic` is
  refused, and the refusal names the `review` tool.
- A delegated agent also loses the `plan` tool. `PLAN.md` belongs to the
  turn owner.
- Every mutating tool call a specialist makes passes the same approval
  gate as the PI's calls, and the preview names who asks:
  `[dft-expert] write_file: ...`.
- The specialist appends to the same `NOTEBOOK.md`, with its entries
  attributed. The notebook is the group's shared memory, so briefs stay
  short.
- `[agent] delegation = false` removes the tool everywhere, at both
  depths.
- Independent briefs go out together with `delegate_many`, and their
  specialists run at the same time. The rules above hold for every
  brief in a wave.

A deliberately small errand, captured whole. The project directory holds
`data.txt`, which says `the secret word is perovskite`:

```bash
slab mason run --auto "Use the delegate tool to send analysis-expert this task: 'Use read_file on data.txt, then call finish reporting the secret word verbatim.' You may not call read_file yourself. After the specialist's report arrives, call finish reporting the word it returned."
```

```text
The secret word is perovskite.
[finish after 2 step(s); tokens 11778+230; transcript .slab/mason/sessions/20260825-040035-17009.jsonl]
```

The workspace now holds two transcripts. The conversation, and the
errand's own archive:

```text
20260825-040035-17009-analysis-expert-1.jsonl
20260825-040035-17009.jsonl
```

Inside the conversation transcript, the `delegate` call's result is the
specialist's report plus one bracketed harness line:

```text
The secret word is "perovskite".

[analysis-expert: answer after 3 step(s); tokens 6269+140; transcript 20260825-040035-17009-analysis-expert-1.jsonl]
```

The harness line is honest. When a specialist stops at its turn budget,
an error streak, or a server failure, the PI reads `turn budget`,
`error_streak`, or `error` there, not a confident report. The PI card's
doctrine is to read that line before trusting the text above it.

`slab mason chat --resume` replays only conversation transcripts. A
delegation archive is never resumed as a conversation.

Delegation quality is the served model's quality. The capture above is
`llama3.1:8b` at temperature 0, and a model of that size handles small,
explicit briefs; it also fails some attempts, so expect retries. Larger
served models handle larger briefs. The single loop remains the default
experience, and nothing requires you to delegate.

### A wave of independent briefs

`delegate_many(briefs)` hands out two or more briefs at once. Each brief
is an `{agent, task, context?}` object, the same three fields `delegate`
takes, and the same specialist may take two of them. The specialists run
their loops at the same time, in threads inside the lead's process and
under the lead's session lock, and the lead receives every report
together:

```text
## md-expert (brief 1 of 2)

The cell is molten at 1200 K (run aa11bb).

[md-expert-1: finish after 1 step(s); tokens 100+10; transcript 20260917-212159-12060-md-expert-1.jsonl; continue with continues="md-expert-1"]

## dft-expert (brief 2 of 2)

a = 3.601 A (run cc22dd).

[dft-expert-2: finish after 1 step(s); tokens 100+10; transcript 20260917-212159-12060-dft-expert-2.jsonl; continue with continues="dft-expert-2"]

wave: 2 briefs, 3 s; sequential would have been about 6 s
```

The last line is the wave's own accounting: the wall-clock it took, and
what the same briefs would have cost one after the other. The capture
above is a two-brief wave against a scripted client whose every call
takes three seconds, so the step counts, the token counts, and the spans
are the script's, not a campaign's.

Briefs in one wave must share no file and no run, because nothing
sequences them. A step that needs another step's result goes through
`delegate`, after it. The launches the briefs make draw on one budget, so
size every brief from `free_resources` before the wave goes out. One
brief that fails leaves the others' reports intact, and its section says
how it stopped; re-brief that one alone with `delegate`.

Each brief keeps its own share of the tool-result cap
(`[agent] max_tool_output_chars`), so a long report is shortened in place
and no section is dropped whole.

`[agent] parallel_delegations` caps a wave, and defaults to 3. A wave
larger than the cap is refused naming it, and `parallel_delegations = 1`
removes the tool and leaves `delegate` alone. `slab mason report` counts
the waves and the wall-clock they saved.

### Continuing a specialist

Every harness line names its specialist by a handle, `md-expert-1` in the
wave above, and ends with `continue with continues="md-expert-1"`. (The
single-delegation capture earlier on this page is older than handles and
shows the line without one.) The handle is the agent name and the ordinal
of the brief that created it, and it is also the tail of the specialist's
transcript name. Pass it back as `continues` to give that same specialist
another turn.

A continued specialist keeps the messages of its earlier turns, so it
still holds the failure record it read, the script it wrote, and the run
it launched. The lead pays for that reading once. Continue a specialist
when the follow-up needs what it already read or wrote, and brief a fresh
one when the step is new.

- The handle is unique in the conversation. An unknown handle is refused
  and the refusal lists the ones that exist.
- The `agent` must match the handle. A critic is never continued, because
  a review is a fresh reading of the text as it stands now.
- Each turn rebuilds the specialist's system message, so a notebook entry
  the lead wrote between the two briefs is in the second turn's prompt.
- The specialist writes on into its own transcript, which marks each turn
  with a `turn` event. The partial outcome and the memories-written list
  under a report cover that turn alone.
- The live specialists die with the process. After `slab mason chat
  --resume`, a continue replays the specialist's transcript into a fresh
  loop and records the resume in it. That reaches the specialists of the
  conversation that was resumed, one hop back.
- A wave's specialists are continued the same way, one at a time. A
  `continues` handle inside a `delegate_many` brief is refused, because
  one live specialist is never in two turns at once.

### Sizing a brief

`steps` is the model-call budget of one brief, and `effort` its reasoning
dial. A brief that reads a record and reports needs few calls at low
effort; a brief that writes a script, launches, and waits takes the
agent's default. Both only lower what the agent already runs under:

| Where the value comes from | Wins over |
|---|---|
| `[agent]` and `[agent.roster.<name>]` | nothing |
| the CLI flags `--max-turns` and `--effort` | the config |
| the brief's `steps` and `effort` | the config, and only downward |

`steps` above the agent's cap is refused naming the cap, and `effort`
above the agent's effort is refused the same way (the ladder is none,
low, medium, high, xhigh, max, and an unset effort counts as xhigh). A
flag outranks the brief: when `--max-turns` or `--effort` is set, the
flag's value stands and the report carries one harness note saying the
brief's was ignored.

The specialist's own budget hint reads `model call 1 of 8` under a brief
of eight steps, and the harness line says which budget stopped it: `turn
budget (8, set by the brief)` against `turn budget (60)`. The first is a
brief to continue with more steps. The second is a task to cut down.

### Helpers

A specialist may hand a script to a helper, one level further down. A
helper card sets `helper: true`. It takes briefs and never delegates, so
the tree stops there. The one built-in helper is `coding-expert`. It
writes, fixes, and checks scripts and inputs: analysis code, LAMMPS or QE
input files, and shell pipelines. It reads the failing file and its
failure record before it edits, changes one thing at a time, and returns
the path, the change, and the run or command that proves the fix. It does
not decide the science. When a fix needs a cutoff, an ensemble, or a
potential, it stops and reports the choice back.

The rules are code:

- A specialist briefed by a lead gets a `delegate` tool whose team is the
  helpers only. A brief to any other card is refused, and the refusal
  names only the helpers. A specialist has no `delegate_many`, so its
  helper briefs run one after the other.
- A helper never gets a `delegate` tool, at any depth.
- A lead's team takes the helper too, so the PI may brief `coding-expert`
  directly.
- A helper brief counts against the specialist's own brief. The helper
  runs at most the calls the specialist's brief has left, even when its
  own cap is larger. When the helper stops there, the specialist reads a
  harness note that says so.
- One specialist turn sends at most `[agent] helper_briefs` helper briefs,
  3 by default. The next one is refused with `this brief has used its 3
  helper calls; finish with what you have`. Raise it for one card in its
  `[agent.roster.<name>]` table.
- A helper's handle stays with the specialist that briefed it. The
  specialist may pass it as `continues`, and the lead may not.
- The helper's tokens count toward the specialist's total and then the
  lead's.

The PI, md-expert, dft-expert, analysis-expert, and worker brief
`coding-expert` on the first traceback a script raises, and when the
brief needs a script that does not exist yet. An engine error and a
failed check stay with the card whose domain they are in.

The harness enforces that division, so it does not rest on the cards
alone. Under the `script-bug-handoff` mechanism, every tool result that
reports a Python failure of a script the agent wrote ends with a line
naming the helper, the second failure of one script is briefed by the
harness itself inside that tool call, and a finish over an unfixed script
bug is refused once. The three tiers are described in
[Mason](mason.md#a-python-bug-in-the-agents-script). The automatic brief
is an ordinary helper brief: it takes a handle, it counts against
`[agent] helper_briefs`, and the lead reads it in the footer with the
rest.

The lead reads each helper brief under the specialist's own harness
line. This capture is the end of a real `delegate` result from a run
against `llama3.1:8b` on a local Ollama. The PI briefed md-expert about a
broken analysis script, and md-expert briefed `coding-expert` three
times:

```text
[md-expert-1: finish after 54 step(s); tokens 997038+2006; transcript 20260918-014119-55519-md-expert-1.jsonl; continue with continues="md-expert-1"]
[harness] helper coding-expert-1: 7 calls, error_streak
[harness] helper coding-expert-2: 7 calls, error_streak
[harness] helper coding-expert-3: 7 calls, error_streak
```

A model of that size did not fix the script. Each helper repeated one
failing call until the error streak stopped it, and md-expert then hit
its cap of three helper briefs. The lead reads that outcome in the
footer, next to what each helper cost, instead of reading a confident
report.

A helper's transcript is named after its specialist's:
`<stem>-<specialist handle>-<helper>-<n>.jsonl`, for example
`20260918-014119-55519-md-expert-1-coding-expert-1.jsonl`. Every delegated
transcript opens with a `session` header that names its `agent` and its
`parent`, the handle of the session that briefed it. `slab mason report`
reads the header and prints each helper under its specialist.
`slab mason read --live` follows the helper's transcript like any other
delegation, and `slab purge` sweeps it with its conversation.

## A critic before compute

A plan is cheapest to fix before the first run. The `critic` card reads
a plan, a brief, or a workflow script and returns a verdict with
numbered findings, each marked blocking or advisory. It runs nothing and
writes nothing. The rules are code, not prompt text:

- A card with `reviews: true` is read-only by construction. The toolbox
  keeps only the tools that observe: the file readers, `list_runs`,
  `show_run`, the engine and task catalogs, the Materials Project
  lookups, `job_status`, `skill`, `recall`, and `finish`. An allowlist
  that names anything else is refused when the card loads.
- A critic takes no briefs. `delegate` refuses it and names the `review`
  tool instead, so every review leaves a record.
- The critic's checklist asks whether the plan names each result exactly
  as the goal's reporting clause does, with the unit. The harness refuses
  a `finish` under another name, so a plan that renames the result leads
  to a refused finish.
- The `review(subject?, focus?, agent?)` tool belongs to the leads. It
  hands `PLAN.md` (the default subject) or a file path to the critic,
  runs the critic's own loop, and returns the findings under a verdict
  line, with the harness line after them. The critic passes its verdict
  in the `verdict` argument of `finish`: `approve` or `revise`. A review
  that ends without one is recorded as `none`, and `none` approves
  nothing. The brief also carries the lead's own `list_engines`,
  `list_tasks`, `describe_task`, and `get_material` results from the
  session, newest first, each cut to 2,000 characters and 8,000 in all,
  so the critic spends its steps on the observable and the contract
  instead of re-gathering the fingerprint. The brief says not to re-run
  those lookups. A second review of the same subject in one session is a
  re-review: the brief carries the prior findings and asks the critic to
  say, for each, whether the text now resolves it. When the critic's
  reply is cut at its reply-token ceiling on the retry too, the verdict
  line says so and names the lead's one move, which is to review again.
  The config change belongs to the operator.
- The findings persist. Each review is one markdown file under
  `.slab/mason/reviews/`, named after the session transcript, and the
  transcript records a `review` event that names it.
- The planner spends no compute before an approval. Its card sets
  `review_first: true`, so `delegate`, `launch_workflow`, and
  `submit_job` are refused until the critic has approved the plan. The
  refusal comes before any approval prompt.
- An approval belongs to one text. The record carries a digest of the
  reviewed plan. A session that starts with the same plan on disk starts
  approved, and an edited plan is a different plan.
- The lead's environment block shows the latest review of the plan, so
  the findings survive compaction. When the plan has changed since the
  review, the block says so.

The record stands alone. The harness wrote this one during a scripted
test run, so the findings are two lines rather than a served model's
review:

```text
---
subject: "plan"
digest: "bd787cb9c53a2802"
verdict: "revise"
reviewer: "critic"
session: "20260903-010916-8876"
transcript: "20260903-010916-8876-critic-1.jsonl"
at: "2026-09-03T01:09:16.131554+00:00"
---
# Findings

1. blocking. "report a in Å": the step has no check. Add a @check that the relaxed stress is below a stated tolerance, so the run can reach verified.
2. advisory. The plan does not say which cell (primitive or conventional) the value is read from; name it, because the two differ by a factor of sqrt(2).

# Reviewed text

# Goal

Lattice constant of fcc Cu under emt.

1. relax_cell the conventional cell; report a in Å.
```

The `pi` has the same tool, and its doctrine is to review a campaign
plan before the first launch. Nothing gates the PI, because a small
interactive task needs no critic. The critic takes `[agent] effort`
unless its roster table says otherwise. Set it there, and set it lower
than a planner's: a verdict over a few thousand characters of plan needs
medium reasoning, and a thinking model at `xhigh` can spend the whole
reply cap on its think block before it writes a word. One real critic
pass ran 78 minutes and 73,000 completion tokens over twelve steps and
returned no verdict.

```toml
[agent.roster.critic]
effort = "medium"
max_reply_tokens = 32000
```

## Skills

A skill is a directory with a `SKILL.md` file, in the
[Agent Skills format](https://agentskills.io/specification). Mason adds
no dialect, so skills written for other tools load unmodified.
Twenty-two skills ship built in:

```bash
slab mason skills
```

```text
atomsk-defects             built-in  dft-expert md-expert         0 script(s)
atomsk-interfaces          built-in  dft-expert md-expert         0 script(s)
atomsk-structures          built-in  dft-expert md-expert         1 script(s)
band-structure             built-in  dft-expert                   1 script(s)
convergence-study          built-in  dft-expert                   1 script(s)
density-of-states          built-in  dft-expert                   1 script(s)
elastic-constants          built-in  analysis-expert dft-expert   1 script(s)
equation-of-state          built-in  analysis-expert dft-expert   1 script(s)
interface-adhesion         built-in  analysis-expert dft-expert   1 script(s)
kinetic-fits               built-in  analysis-expert md-expert    1 script(s)
lammps-potentials          built-in  md-expert                    1 script(s)
lammps-scripting           built-in  md-expert                    1 script(s)
melt-quench                built-in  md-expert                    1 script(s)
mlip-training              built-in  dft-expert md-expert         0 script(s)
mp-screening               built-in  dft-expert md-expert         0 script(s)
msd-diffusion              built-in  analysis-expert md-expert    1 script(s)
nemd-transport             built-in  analysis-expert md-expert    1 script(s)
nucleation-cnt             built-in  analysis-expert md-expert    1 script(s)
radial-distribution        built-in  analysis-expert md-expert    1 script(s)
surface-energy             built-in  dft-expert                   0 script(s)
thermal-response           built-in  analysis-expert md-expert    1 script(s)
two-phase-melting          built-in  md-expert                    2 script(s)
```

The catalog covers structure building (the atomsk skills: crystals and
supercells, defects and dislocations, interfaces and polycrystals),
screening from the offline Materials Project snapshot (mp-screening),
potential training and fine-tuning with gracemaker, with the dataset
rules for each (mlip-training), LAMMPS potential
files, their pair styles, and the KOKKOS switches (lammps-potentials),
LAMMPS input scripts run whole (lammps-scripting), the
static side (equations of state, convergence, band structures, densities
of states, surfaces, elastic constants, interface adhesion), and the dynamic side (melt-quench
glasses, thermal response, two-phase melting by either the NPH plateau
or the interface-velocity ladder, NEMD transport, diffusion,
nucleation), with the fits and unit conversions in tested scripts.

The third column is the categorization: which agent cards see the skill.
The PI sees every skill, because its card sets `skills: all`. A
specialist sees its own slice:

```bash
slab mason skills --agent md-expert
```

```text
atomsk-defects             built-in  dft-expert md-expert         0 script(s)
atomsk-interfaces          built-in  dft-expert md-expert         0 script(s)
atomsk-structures          built-in  dft-expert md-expert         1 script(s)
kinetic-fits               built-in  analysis-expert md-expert    1 script(s)
lammps-potentials          built-in  md-expert                    1 script(s)
lammps-scripting           built-in  md-expert                    1 script(s)
melt-quench                built-in  md-expert                    1 script(s)
mlip-training              built-in  dft-expert md-expert         0 script(s)
mp-screening               built-in  dft-expert md-expert         0 script(s)
msd-diffusion              built-in  analysis-expert md-expert    1 script(s)
nemd-transport             built-in  analysis-expert md-expert    1 script(s)
nucleation-cnt             built-in  analysis-expert md-expert    1 script(s)
radial-distribution        built-in  analysis-expert md-expert    1 script(s)
thermal-response           built-in  analysis-expert md-expert    1 script(s)
two-phase-melting          built-in  md-expert                    2 script(s)
```

The DFT specialist sees the static side, band structures and densities
of states included:

```bash
slab mason skills --agent dft-expert
```

```text
atomsk-defects             built-in  dft-expert md-expert         0 script(s)
atomsk-interfaces          built-in  dft-expert md-expert         0 script(s)
atomsk-structures          built-in  dft-expert md-expert         1 script(s)
band-structure             built-in  dft-expert                   1 script(s)
convergence-study          built-in  dft-expert                   1 script(s)
density-of-states          built-in  dft-expert                   1 script(s)
elastic-constants          built-in  analysis-expert dft-expert   1 script(s)
equation-of-state          built-in  analysis-expert dft-expert   1 script(s)
interface-adhesion         built-in  analysis-expert dft-expert   1 script(s)
mlip-training              built-in  dft-expert md-expert         0 script(s)
mp-screening               built-in  dft-expert md-expert         0 script(s)
surface-energy             built-in  dft-expert                   0 script(s)
```

Skills load progressively. The system prompt carries one line per
visible skill, the name and the description. When a task matches, the
agent calls the `skill` tool, which returns the full instructions, the
skill's root path, and its bundled files. The agent then reads
references with `read_file` and runs scripts with `shell`. A skill
script therefore runs under exactly the approval gate and the
`shell_allowlist` that govern every other command. There is no separate
execution surface.

The bundled scripts are the point. Every fit script (`fit_eos.py`,
`fit_elastic.py`, `fit_rates.py`, `fit_nemd.py`, `msd.py`, and the rest)
is an argparse program with `--json` output and actionable errors, and
the test suite runs each one on real data. An agent that uses them does
not re-derive a Birch-Murnaghan fit or a Voigt-Reuss-Hill average in
every session, and the analysis itself has provenance: the skill names
the script, and the script version ships with the package. Some skills
also bundle an `assets/` workflow template (`eos_scan.py`,
`strain_scan.py`, `melt_quench.py`, `thermal_ramp.py`); the agent copies
the template into the project, edits the constants at the top, and
launches it as a traced run. The templates that run dynamics
(`md_nvt.py`, `melt_quench.py`, `thermal_ramp.py`) hand LAMMPS a whole
input script through `run_lammps`; none drives dynamics from Python.

## Discovery: three layers

Skills and cards are discovered the same way. A name in a higher layer
shadows the lower ones whole:

| Layer | Skills | Agent cards |
|---|---|---|
| project | `<project>/skills/` | `<project>/agents/` |
| user | `~/.config/slab/skills/` | `~/.config/slab/agents/` |
| built-in | inside the package | inside the package |

Project skills and cards are ordinary files in the project directory.
Commit them, like `NOTEBOOK.md` and `AGENTS.md`: they are part of the
project's provenance. A project card named `pi.md` replaces the default
PI entirely.

A malformed skill or card is a loud error naming the file and the rule.
A skill that silently vanished from the catalog would be undebuggable.

## Write a skill

The minimum is one directory and one file:

```text
skills/
  xrd-pattern/
    SKILL.md
    scripts/
      simulate_xrd.py
```

```markdown
---
name: xrd-pattern
description: Simulate a powder X-ray diffraction pattern from a recorded
  structure and compare peak positions against a reference. Use when
  asked about XRD, diffraction peaks, or phase identification.
metadata:
  mason-agents: "analysis-expert"
---
# XRD pattern

## 1. Simulate

Run the bundled script on a structure file:

    python <skill root>/scripts/simulate_xrd.py relaxed.cif --json

## 2. Compare
...
```

The rules come from the Agent Skills specification:

- `name` is required: 1 to 64 characters, lowercase letters, digits, and
  single hyphens, and it must equal the directory name.
- `description` is required, at most 1024 characters. Write what the
  skill does and when to use it. This line is the trigger the agent
  reads, so include the words a task would contain.
- Keep the body under 500 lines. Move long reference material to files
  in `references/`; agents read them on demand.
- `metadata` values must be strings. The `mason-agents` key is a
  space-separated list of card names; omit it to show the skill to every
  agent.

The spec's experimental `allowed-tools` field is accepted and ignored.
The toolbox already gates approval per call, and `slab mason skills` reports
the field as ignored so nothing is silent.

## Revise a skill

Every skill has a digest: a short hash of every file under its root. The
`skill` tool records it when the skill loads, and a benchmark campaign
carries it in its record. A flag the review raises on a skill is raised
against that revision.

Revise a skill from its flags, not from intuition:

1. Read the open flags on the skill.

    ```bash
    slab benchmark flags --target skill:equation-of-state --status open
    ```

2. Edit the description, the body, or the script the flag names, and run
   the script's test.
3. Run the campaigns for the questions that list the skill, and score
   them.
4. Check the gate. It refuses the revision until a campaign under it
   passes without regressing or raising the flag.

    ```bash
    slab benchmark gate equation-of-state
    ```

[The science review](../review.md) describes the flags, the evaluators,
and the gate.

## Write an agent card

A card is one markdown file whose body is the agent's role prompt:

```markdown
---
name: literature-scout
description: Finds and summarizes what the project's own notes and files
  already say about a topic. Delegate lookups into the project's recorded
  knowledge.
tools: read_file list_dir search skill notebook finish
---
You are the literature scout of a SLAB research group. You search the
project's files and notebook, quote exactly, and cite file paths for
every claim. You do not compute and you do not speculate.
```

- `name` and `description` follow the same rules as skills. The
  description is what the PI reads when deciding whom to delegate to, so
  write it as a delegation trigger.
- `tools` is optional. Absent means every tool the session offers.
  Present, it is validated against the full tool vocabulary, so a typo
  is refused even on a machine where the misspelled tool is absent.
  `finish` is always available.
- `skills: all` shows the full catalog (the PI uses this);
  `matching` (the default) shows the skills that name this card, plus
  the unrestricted ones.
- `delegates: true` grants the `delegate` tool, at depth zero only.
- `helper: true` makes the card a helper. It takes briefs from a lead or
  from a specialist, and it never delegates. It cannot be combined with
  `delegates`, `reviews`, or `review_first`.
- `reviews: true` makes the card a critic: read-only by construction,
  reached with the `review` tool, never briefed. It cannot be combined
  with `delegates` or `review_first`.
- `review_first: true` refuses the card's `delegate`, `launch_workflow`,
  and `submit_job` until a critic has approved the plan. A `tools`
  allowlist on such a card must name `review`.
- `core: false` makes the body the whole system prompt: no shared
  discipline, no compute budget, no software notes, and a minimal
  environment block. Such a card is never on a team and cannot delegate
  or review. The two condition cards use it.

The shared harness discipline (evidence, verification, honesty, tool
rules) is appended to every card automatically, unless the card sets
`core: false`. A card states identity and domain doctrine, nothing else,
so 20 to 60 lines is the normal size.

## A planner and a worker

Deep reasoning is expensive at every step, and most steps do not need
it. The `planner` card keeps the reasoning for the plan and hands the
steps to cheaper agents:

- The planner writes `PLAN.md` first, one step per brief, each with its
  success criterion and the evidence it must return.
- It hands the plan to the critic with the `review` tool and resolves
  the blocking findings. The harness refuses its briefs until the
  verdict is `approve`.
- It hands every step to a specialist or to the `worker` with the
  `delegate` tool, and revises the plan after each report.
- It confirms every cited run with `show_run` before a number enters the
  plan, and it owns the final report.

The rule that the planner runs nothing is code. Its tool allowlist has
no `shell`, no `launch_workflow`, and no file edits, so the card cannot
drift into doing the work itself at planner prices. The planner does read
evidence. It keeps `read_artifact` and `read_file`, so a check that two
lines of a run's averages table settle costs two calls and not a
delegation.

Four rules tie the planner's briefs to the record:

| rule | what the harness does |
|---|---|
| Size a wave to the free budget | Each request the planner receives ends with the free cpus and gpus at that step. The card and the environment's resource line state the rule. A wave has as many concurrent launches as free GPUs (or free CPU slices), the planner waits once on the wave, and the next wave starts when the first finishes. |
| Start from the prior result | The environment shows the notebook's earlier entries for this project with their dates (below). The `plan` tool refuses a plan whose Goal names a quantity the notebook reports, until the plan has a line `prior result: ...`. |
| Name the source of every gate | Every numeric gate in a brief names a skill's stated bound, a calibration run id, or the word "estimate" with the fallback action. The critic lists a gate without a source as an advisory finding. |
| Name a file as `run:<id>/<name>` | The `plan` tool checks each reference against the run store. A reference to a cache-hit run is rewritten to the run that produced the file, because a cache hit runs nothing and keeps none of the files its task wrote. A reference that no run answers refuses the plan. `delegate` rewrites a brief the same way and names the rewrite below the report. |

The prior-result check reads the notebook as it was when the session
started, so a result the session records itself does not trip it. A
measured line (a number with a unit) that repeats a quantity phrase of
the Goal, such as "melting point", or a result key such as `t_melt`,
refuses the plan. A line that shares only one word with the Goal adds a
note to the tool result and the plan is written. A planner started
with `[agent] delegation = false`, or on a roster where every other card
delegates, is refused before the model is called, because it would have
the tools of a reader and nobody to brief.

The `worker` is the executor for any step no specialist's domain names.
It sees every skill, takes one brief, does what the brief says and no
more, and returns numbers with units and run ids.

Give the planner the reasoning and the worker the economy in
`slab.toml`, then start the planner as the entry agent:

```toml
[agent.roster.planner]
effort = "xhigh"

[agent.roster.worker]
effort = "none"
```

```bash
slab mason run --agent planner "measure the lattice constant of fcc Cu with the balanced protocol"
```

`slab mason sandbox render`, `sandbox launch`, `slab benchmark render`,
and `benchmark launch` take the same `--agent` flag. The render records
it, so a later `launch` without arguments reuses it, and `slab doctor`
re-renders with it when it checks that the job files are fresh.

In a sandbox job the `[agent.roster.<name>]` tables travel with the
rendered config, so the effort split holds inside the job. The split is
only as real as the server: `effort` reaches an OpenAI-compatible server
verbatim as `reasoning_effort`, a server that does not know the field
ignores it, and a server that knows only part of the scale may treat the
rest as unset. On a server whose top level is the unset default, leave
the planner's `effort` out rather than writing `xhigh`. Check the
completion tokens per call in `slab mason report` before relying on
it. `slab doctor` notes a roster table that sets `effort` while `[agent]`
does not, because every card without a table then runs at the server's
default, and a table for a card that only `--agent` can start. Their
`provider`, `endpoint`, and `api_key_env` keys stay on the host, because
every agent in the job talks to the one bridged endpoint, and the render
warns when it drops one. A planner on a different provider than its
workers is possible outside the sandbox only.

## A model per agent

Cards are portable and never name models. Machine facts live in
`slab.toml`, in one table per agent:

```toml
[agent]
model = "qwen3-coder:30b"        # every agent's default

[agent.roster.pi]
provider = "anthropic"           # the PI orchestrates on a stronger model
model = "claude-opus-5"

[agent.roster.dft-expert]
temperature = 0.0                # the specialist executes deterministically
max_turns = 30
```

The table accepts the connection and budget fields: `provider`,
`endpoint`, `model`, `api_key_env`, `effort`, `temperature`,
`context_window`, `compact_at`, `max_turns`, `max_reply_tokens`,
`request_timeout_s`, `max_tool_output_chars`, `clear_tool_results`,
`keep_tool_results`, `clear_tool_results_at`, `shell_timeout_s`,
`table_nudge_rows`, and `tool_protocol`. A specialist that reads data can
run under a lower `max_reply_tokens` than its lead this way, so a cut
reply costs the session less. Session policy is deliberately not per-agent: one
`approval` mode, one `shell_allowlist`, one `[agent.serve]` section per
session.

Three rules keep the merge predictable:

- A table that names no card is refused, naming the roster. Config that
  silently does nothing is a trap.
- CLI flags outrank the tables, for the entry agent and for delegated
  agents alike. `--model X` means X for everyone.
- A table that sets `provider` without `endpoint` also clears the
  endpoint, so the new provider's default applies. A vLLM URL must not
  survive a switch to the Anthropic API.

`slab mason doctor` probes every distinct connection the roster produces. A
specialist pinned to an unserved model fails the doctor, not the first
delegation.
