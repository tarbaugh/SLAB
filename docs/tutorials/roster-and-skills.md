# The roster and skills

Mason is a research group, not a single agent. The group is the roster: a
set of agents defined as markdown files called agent cards. The default
agent is `pi`, the principal investigator. The PI can hand a scoped task
to a specialist with the `delegate` tool. Every agent can load skills:
reusable procedure packages with tested analysis scripts.

This page shows how to use the roster, how to write a card and a skill,
and how to configure a different model per agent. The recorded outputs
are exact captures from real executions against a local Ollama.

## The roster

Four cards ship built in. `slab mason roster` lists what is visible from the
current project, with the layer each card came from and the model it
would use:

```bash
slab mason roster
```

```text
pi                 built-in  llama3.1:8b                  13 skill(s)  [delegates]
analysis-expert    built-in  llama3.1:8b                  9 skill(s)
dft-expert         built-in  llama3.1:8b                  5 skill(s)
md-expert          built-in  llama3.1:8b                  8 skill(s)
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

## Skills

A skill is a directory with a `SKILL.md` file, in the
[Agent Skills format](https://agentskills.io/specification). Mason adds
no dialect, so skills written for other tools load unmodified. Sixteen
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
msd-diffusion              built-in  analysis-expert md-expert    1 script(s)
nemd-transport             built-in  analysis-expert md-expert    1 script(s)
nucleation-cnt             built-in  analysis-expert md-expert    1 script(s)
radial-distribution        built-in  analysis-expert md-expert    1 script(s)
surface-energy             built-in  dft-expert                   0 script(s)
thermal-response           built-in  analysis-expert md-expert    1 script(s)
two-phase-melting          built-in  md-expert                    0 script(s)
```

The catalog covers structure building (the atomsk skills: crystals and
supercells, defects and dislocations, interfaces and polycrystals), the
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

The shared harness discipline (evidence, verification, honesty, tool
rules) is appended to every card automatically. A card states identity
and domain doctrine, nothing else, so 20 to 60 lines is the normal size.

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
`request_timeout_s`, `max_tool_output_chars`, `shell_timeout_s`, and
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
