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

Seven cards ship built in: two leads, `pi` and `planner`, three
specialists, a `worker`, and a `critic`. `slab mason roster` lists what is visible
from the current project, with the layer each card came from and the
model it would use:

```bash
slab mason roster
```

```text
pi                 built-in  llama3.1:8b                  18 skill(s)  [delegates]
analysis-expert    built-in  llama3.1:8b                  9 skill(s)
critic             built-in  llama3.1:8b                  18 skill(s)  [reviews]
dft-expert         built-in  llama3.1:8b                  10 skill(s)
md-expert          built-in  llama3.1:8b                  13 skill(s)
planner            built-in  llama3.1:8b                  18 skill(s)  [delegates, review first]
worker             built-in  llama3.1:8b                  18 skill(s)
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

The PI has one tool the specialists lack: `delegate(agent, task,
context?)`. It runs the named specialist's own tool loop against the
shared workspace and returns the specialist's final report. The rules are
code, not prompt text:

- Delegation goes one level down. A delegated agent never has the
  `delegate` tool, whatever its card says.
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
- `[agent] delegation = false` removes the tool everywhere.

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

The harness line is honest. When a specialist stops at its turn budget
or an error streak, the PI reads `max_turns` or `error_streak` there,
not a confident report. The PI card's doctrine is to read that line
before trusting the text above it.

`slab mason chat --resume` replays only conversation transcripts. A
delegation archive is never resumed as a conversation.

Delegation quality is the served model's quality. The capture above is
`llama3.1:8b` at temperature 0, and a model of that size handles small,
explicit briefs; it also fails some attempts, so expect retries. Larger
served models handle larger briefs. The single loop remains the default
experience, and nothing requires you to delegate.

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
- The `review(subject?, focus?, agent?)` tool belongs to the leads. It
  hands `PLAN.md` (the default subject) or a file path to the critic,
  runs the critic's own loop, and returns the findings under a verdict
  line, with the harness line after them. The critic passes its verdict
  in the `verdict` argument of `finish`: `approve` or `revise`. A review
  that ends without one is recorded as `none`, and `none` approves
  nothing.
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
interactive task needs no critic. Give the critic the reasoning a
review needs in `slab.toml`:

```toml
[agent.roster.critic]
effort = "xhigh"
```

## Skills

A skill is a directory with a `SKILL.md` file, in the
[Agent Skills format](https://agentskills.io/specification). Mason adds
no dialect, so skills written for other tools load unmodified. Eighteen
skills ship built in:

```bash
slab mason skills
```

```text
atomsk-defects             built-in  dft-expert md-expert         0 script(s)
atomsk-interfaces          built-in  dft-expert md-expert         0 script(s)
atomsk-structures          built-in  dft-expert md-expert         1 script(s)
convergence-study          built-in  dft-expert                   1 script(s)
elastic-constants          built-in  analysis-expert dft-expert   1 script(s)
equation-of-state          built-in  analysis-expert dft-expert   1 script(s)
interface-adhesion         built-in  analysis-expert dft-expert   1 script(s)
kinetic-fits               built-in  analysis-expert md-expert    1 script(s)
melt-quench                built-in  md-expert                    1 script(s)
mlip-training              built-in  dft-expert md-expert         0 script(s)
mp-screening               built-in  dft-expert md-expert         0 script(s)
msd-diffusion              built-in  analysis-expert md-expert    1 script(s)
nemd-transport             built-in  analysis-expert md-expert    1 script(s)
nucleation-cnt             built-in  analysis-expert md-expert    1 script(s)
radial-distribution        built-in  analysis-expert md-expert    1 script(s)
surface-energy             built-in  dft-expert                   0 script(s)
thermal-response           built-in  analysis-expert md-expert    1 script(s)
two-phase-melting          built-in  md-expert                    0 script(s)
```

The catalog covers structure building (the atomsk skills: crystals and
supercells, defects and dislocations, interfaces and polycrystals),
screening from the offline Materials Project snapshot (mp-screening),
potential training with gracemaker (mlip-training), the
static side (equations of state, convergence, surfaces, elastic
constants, interface adhesion), and the dynamic side (melt-quench
glasses, thermal response, two-phase melting, NEMD transport, diffusion,
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
melt-quench                built-in  md-expert                    1 script(s)
mlip-training              built-in  dft-expert md-expert         0 script(s)
mp-screening               built-in  dft-expert md-expert         0 script(s)
msd-diffusion              built-in  analysis-expert md-expert    1 script(s)
nemd-transport             built-in  analysis-expert md-expert    1 script(s)
nucleation-cnt             built-in  analysis-expert md-expert    1 script(s)
radial-distribution        built-in  analysis-expert md-expert    1 script(s)
thermal-response           built-in  analysis-expert md-expert    1 script(s)
two-phase-melting          built-in  md-expert                    0 script(s)
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
launches it as a traced run.

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
- `reviews: true` makes the card a critic: read-only by construction,
  reached with the `review` tool, never briefed. It cannot be combined
  with `delegates` or `review_first`.
- `review_first: true` refuses the card's `delegate`, `launch_workflow`,
  and `submit_job` until a critic has approved the plan. A `tools`
  allowlist on such a card must name `review`.

The shared harness discipline (evidence, verification, honesty, tool
rules) is appended to every card automatically. A card states identity
and domain doctrine, nothing else, so 20 to 60 lines is the normal size.

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
drift into doing the work itself at planner prices. A planner started
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
effort = "low"
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
as `reasoning_effort`, and a server that does not know the field ignores
it. Check the completion tokens per call in `slab mason report` before
relying on it. Their
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
`keep_tool_results`, `clear_tool_results_at`, `shell_timeout_s`, and
`tool_protocol`. Session policy is deliberately not per-agent: one
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
