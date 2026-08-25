# The roster: multi-agent Mason and skills — implementation plan

Status: direction requested by the user 2026-08-24; technical decisions made
under liberal control and standing unless the user objects. Written for an
implementing agent. The decisions in §2 are settled once the user approves
this plan. Questions the plan does not answer go to the user.

Three taste calls are surfaced in §0.2 because they are the user's to
confirm; everything else is decided.

## 0. Read this first

**Goal.** Upgrade `mason` from one resident agent to a small research group,
and give every agent a skills library:

1. **The roster.** A set of agents defined as markdown *agent cards*:
   a Principal Investigator (`pi`, the default agent and orchestrator) and
   specialists (`dft-expert`, `md-expert`, `analysis-expert`). The PI can
   `delegate` a scoped task to a specialist, who runs their own ReAct loop
   with their own system prompt and toolset and returns a report. One level
   deep, sequential, never a swarm.
2. **Skills.** Reusable procedure packages in the [Agent Skills
   format](https://agentskills.io/specification) (`SKILL.md` + optional
   `scripts/`, `references/`, `assets/`), categorized per specialist,
   discovered from the package, the user config dir, and the project.
   Skills carry tested analysis scripts so agents stop re-deriving the same
   `fit_eos.py` in every session.

**Everything lands in `mason`.** No `slab` or `foundation` source change,
no new top-level config table, no workspace schema change.
`tests/test_layering.py` continues to prove the direction. The only files
outside `src/mason` that change are tests, docs, `pyproject.toml`, and
`CLAUDE.md`.

**This reverses a recorded decision, deliberately.** `mason/__init__.py`
and `docs/tutorials/mason.md` record "multi-agent orchestration
deliberately omitted", citing Anthropic's finding that orchestrator-worker
systems pay off for breadth-first search, not interdependent work, at ~15×
the tokens. That argument was about *parallel swarms for one question*. What
the roster adds is different and is the case the same literature endorses:
**context specialization and isolation for separable subtasks.** A k-mesh
convergence study is a self-contained errand: it needs a DFT-specific
system prompt and skills, it pollutes the PI's context with ladder tables,
and only its conclusion matters upstream. Domain evidence now exists on
both sides of this design: Agent Laboratory (arXiv:2501.04227, already
cited in mason.md) and the Virtual Lab (Swanson et al. 2024, PI agent +
specialist scientists with individual meetings, experimentally validated
nanobodies) both structure research as PI-plus-specialists; Google's AI
co-scientist (Gottweis et al. 2025, arXiv:2502.18864) and chemistry agents
(ChemCrow, Coscientist; El Agente and MDCrow for QC/MD specifically) show
domain-scoped agents outperforming one generalist prompt. The single loop
remains the unit of work and the default experience; the roster composes
loops. Phase 5 rewrites the provenance section to say exactly this — the
old argument is amended in the open, not silently deleted.

**Prior art the design borrows, precisely:**

- *Agent Skills* (agentskills.io spec; Anthropic engineering, Oct 2025):
  the SKILL.md format, progressive disclosure (~100-token metadata always
  in context, body on activation, resources on demand), scripts as
  reliable substitutes for regenerated code. Mason adopts the spec
  verbatim so third-party skills work unmodified (§2.2).
- *Claude Code subagents*: markdown agent definitions (frontmatter
  name/description/tools + body as system prompt), delegation as a tool,
  child context isolation, results returned as reports.
- *Blackboard architecture* (Hearsay-II): the shared `NOTEBOOK.md` is the
  blackboard. Every agent reads it at session start and writes attributed
  entries; delegation briefs stay short because durable state is in the
  file, not the handoff.
- *Anthropic multi-agent research system*: the cost warning is honored —
  delegation is optional per task, depth-limited to one, sequential, and
  switchable off in config (§2.7).

**How to work.**

- Work on a branch named `roster`. One commit per phase. Stop before
  merging or pushing to `main`; the user reviews first.
- Every phase ends with all four gates green (§5). Do not start the next
  phase on a red tree.
- Run tools from the venv explicitly (`.venv/bin/python`,
  `.venv/bin/pytest`, `.venv/bin/ruff`, `.venv/bin/mypy`,
  `.venv/bin/mkdocs`). The ambient shell resolves an anaconda Python ahead
  of the venv.
- Do not change behavior beyond what this plan lists. In particular the
  solo path (`mason run` with no roster content beyond built-ins) must
  keep working against an 8B model exactly as today, plus a skills list in
  its prompt.
- Prose in `README.md` and `docs/` follows `CLAUDE.md` (ASD-STE100
  principles). Captured outputs in docs are never hand-edited; phase 5
  re-executes them or marks them `<!-- no-verify -->` with a disclosure.
- If review agents are used later: they copy the tree before mutating it
  (lesson from the foundation-split review).

**Size, for orientation.** `src/mason` is 3,697 lines today. Expect
roughly +2,900: `skills.py` ~280, `roster.py` ~260, delegation (loop +
session + tools) ~220, prompts restructure ~120 delta, CLI ~170, built-in
cards ~240 of markdown, built-in skills ~900 (markdown + scripts), tests
~1,300.

### 0.1 Current architecture the plan builds on (verified 2026-08-24)

- `mason.loop.Mason`: one ReAct loop; `run_turn` until answer / `finish` /
  `max_turns` / 5-failure streak; compaction at `compact_at` ×
  `context_window` folding into `NOTEBOOK.md`; native + fenced tool
  protocols.
- `mason.tools.build_toolbox(session)`: 14–16 tools (file primitives,
  `shell` with allowlist gate, `list_runs`/`show_run`/`launch_workflow`,
  `list_engines`, SLURM trio when partitions exist, `notebook`, `plan`,
  `finish`). Approval on mutating tools via `session.allows`.
- `mason.session.MasonSession`: cwd + workspace + `[agent]`/`[hpc]` views,
  approval gate, `NOTEBOOK.md`/`PLAN.md`, JSONL transcripts under
  `<workspace>/mason/sessions/`, usage counters, endpoint discovery.
- `mason.prompts`: layered for prefix caching — static core, per-session
  environment block (AGENTS.md, PLAN.md, notebook tail), compaction brief.
- `mason.config.AgentConfig`: frozen, `extra="forbid"`; `[agent]` +
  `[agent.serve]`; CLI overrides go through validated rebuild
  (`_override_agent` in `cli.py`).
- Tests: `FakeClient` scripted replies drive every loop mechanism; a gated
  `SLAB_TEST_LLM` test runs the real loop against Ollama.

### 0.2 Taste calls for the user (recommendation first)

1. **Built-in roster membership.** Recommended: exactly four cards —
   `pi`, `dft-expert`, `md-expert`, `analysis-expert`. Anything else
   (structure building, literature) is a project-level card the user adds
   as a file. Confirm or name additions.
2. **Where project skills and cards live.** Recommended: visible
   `skills/` and `agents/` directories in the project root, committed like
   `NOTEBOOK.md`/`PLAN.md`/`AGENTS.md` (they are provenance, and `agents/`
   reads naturally next to `AGENTS.md`). Alternative: hidden under a dot
   directory. §2.3 assumes the visible form.
3. **Delegation default.** Recommended: on by default (`[agent]
   delegation = true`), because the PI card's own prompt tells small
   models to prefer working solo; the config switch exists for hard
   off. Alternative: default off, opt-in per project.

## 1. Target layout

```
src/mason/
  __init__.py        # docstring bullet rewritten; new exports
  roster.py          # NEW: agent cards — parse, validate, discover
  skills.py          # NEW: skills — parse, validate, discover, catalog
  loop.py            # Mason gains spec= and depth=; delegate wiring
  tools.py           # skill + delegate tools; per-card filtering
  session.py         # spawn() for child sessions; agent attribution
  prompts.py         # core/role split; skills + team blocks
  config.py          # [agent.roster.<name>] overrides; delegation switch
  cli.py             # --agent; `mason roster`; `mason skills`
  agents/            # NEW package data: built-in agent cards
    pi.md  dft-expert.md  md-expert.md  analysis-expert.md
  skills/            # NEW package data: built-in skills
    equation-of-state/       SKILL.md  scripts/fit_eos.py
    convergence-study/       SKILL.md  scripts/convergence_table.py
    radial-distribution/     SKILL.md  scripts/rdf.py
    msd-diffusion/           SKILL.md  scripts/msd.py
    surface-energy/          SKILL.md            # no script: shows optionality

tests/
  test_mason_skills.py    test_mason_roster.py    test_mason_delegate.py
  (+ additions to test_mason_cli.py, test_mason_real.py, test_layering.py)

docs/tutorials/roster-and-skills.md   # NEW page (+ mkdocs.yml nav entry)
```

Discovery precedence for both skills and cards, highest first:

| Layer | Skills | Agent cards |
|---|---|---|
| project | `<cwd>/skills/*/SKILL.md` | `<cwd>/agents/*.md` |
| user | `~/.config/slab/skills/*/SKILL.md` | `~/.config/slab/agents/*.md` |
| built-in | `src/mason/skills/` | `src/mason/agents/` |

Same name shadows lower layers whole (no merging), mirroring config-file
layering. A project card named `pi.md` replaces the built-in PI.

## 2. Decisions (settled on approval)

### 2.1 Names

One name per concept, used everywhere (docs, code, CLI, config):

| Concept | Name |
|---|---|
| One agent definition | agent card (a markdown file) |
| The set of agents | the roster |
| The default/orchestrating agent | `pi` (the product is still Mason; `pi` is Mason's default card) |
| A non-PI agent | specialist |
| Handing a task down | the `delegate` tool |
| A procedure package | skill |
| Loading a skill | the `skill` tool |
| Config overrides per agent | `[agent.roster.<name>]` |
| CLI listing verbs | `mason roster`, `mason skills` |
| Choosing the entry agent | `--agent <name>` on `chat` and `run` |
| Branch | `roster` |

On-disk names stay under the SLAB umbrella: nothing new in `$SLAB_*` or
`.slab/`; the user config dir gains `skills/` and `agents/` subdirectories
under the existing `~/.config/slab/`.

### 2.2 Skills are Agent Skills — the open spec, no dialect

A mason skill is a directory whose `SKILL.md` satisfies the Agent Skills
specification exactly (verified against the spec 2026-08-24):

- Frontmatter `name`: required, 1–64 chars, lowercase alphanumerics and
  hyphens, no leading/trailing/consecutive hyphen, **must equal the parent
  directory name**.
- Frontmatter `description`: required, 1–1024 chars. This line is what the
  agent sees before activation, so it states what the skill does *and*
  when to use it.
- Optional: `license`, `compatibility` (≤500 chars), `metadata`
  (string→string map), `allowed-tools` (accepted, ignored, and reported as
  ignored in `mason skills` output — the toolbox already gates approval).
- Body: instructions; loaded whole on activation; authors keep it under
  500 lines / ~5k tokens; `scripts/`, `references/`, `assets/` on demand.

**The mason extension rides in `metadata`, where the spec sends it:**

```yaml
metadata:
  mason-agents: "dft-expert analysis-expert"
```

A space-separated list of card names, mirroring the spec's own
`allowed-tools` convention (metadata values must be strings). Absent means
every agent sees the skill. Unknown names are kept (the card may exist in
another project) but reported by `mason skills`. Any Claude-style skill
drops in unmodified and is visible to all agents; any mason skill is valid
to every other Agent Skills consumer, which just ignores the metadata.

Parsing uses PyYAML (new core dependency, `pyyaml>=6`; `types-PyYAML` in
`dev` for mypy strict). Frontmatter in the wild uses folded scalars and
quoting; a hand-rolled subset parser is the classic trap. Refusals name
the file and the violated rule (`SkillError(MasonError)`).

### 2.3 Discovery and the `skill` tool

`mason/skills.py`:

```python
@dataclass(frozen=True)
class Skill:
    name: str; description: str; root: Path
    source: Literal["built-in", "user", "project"]
    agents: frozenset[str] | None      # None = all
    compatibility: str | None; license: str | None
    ignored_allowed_tools: bool

parse_skill(skill_dir: Path, source: str) -> Skill       # raises SkillError
discover_skills(cwd: Path) -> dict[str, Skill]           # precedence §1
skills_for(spec: AgentSpec, skills: ...) -> list[Skill]  # filter + card scope
catalog_block(skills: list[Skill]) -> str                # "- name: description"
listing(skill: Skill) -> str                             # body + file inventory
```

The built-in layer resolves through `importlib.resources.files("mason")`
so it works installed and editable. A malformed *project or user* skill is
a loud `SkillError` naming the file (a broken skill silently missing from
the roster would be undebuggable); built-ins failing validation is a bug
caught by tests.

**The `skill` tool** (read-only, no approval): `skill(name)` returns the
SKILL.md body followed by the skill's absolute root path and a sorted
inventory of its files (relative paths, capped at 50 entries). The agent
then uses the existing primitives — `read_file` for references, `shell`
for scripts — so **no new execution surface exists**: a skill script runs
under exactly the approval gate and allowlist that govern every other
shell command. An unknown name answers with the names available to *this*
agent. Activation is recorded in the transcript
(`{"type": "skill", "name": ...}`).

**Prompt integration.** The agent's available skills render as one line
each (`- name: description`) in a `# Skills` section of the system
message, session-stable so prefix caching holds (§2.8). The section
teaches the two-step: read the list, call `skill(name)` before doing a
task a skill covers, follow the loaded instructions, prefer bundled
scripts over rewriting them.

### 2.4 Agent cards

`mason/roster.py`. A card is one markdown file, `<name>.md`:

```markdown
---
name: dft-expert
description: Plans and runs DFT calculations with Quantum ESPRESSO —
  protocols, convergence, pseudopotentials. Delegate anything that needs
  cutoffs, k-meshes, or SCF diagnosis.
tools: read_file write_file edit_file list_dir search shell launch_workflow
  list_runs show_run list_engines submit_job job_status cancel_job notebook
skills: matching
delegates: false
---
You are the DFT specialist of a SLAB research group...
(body = the role section of the system prompt)
```

- `name`, `description`: required; same character rules as skill names;
  `name` must equal the file stem. The description is what the PI reads
  when deciding whom to delegate to — write it as a delegation trigger.
- `tools`: optional space-separated allowlist. Absent = every tool the
  session offers. Validation: every listed name must exist in the
  **universal tool vocabulary** (a frozenset exported by `mason.tools`);
  unknown names are a `RosterError` naming the card (catches typos), but a
  known tool absent from *this session* (e.g. `submit_job` with no
  partitions) is silently unavailable, exactly as today. `finish` is
  always present regardless of the list.
- `skills`: `matching` (default — skills whose `mason-agents` include this
  card or are unrestricted) or `all` (the full catalog; the PI card sets
  this so solo mode loses nothing).
- `delegates`: boolean, default false. Grants the `delegate` tool — but
  only at depth 0 (§2.7), so marking two cards `delegates: true` still
  cannot recurse.
- Body: the role block. The card supplies identity and domain doctrine;
  the shared core prompt (§2.8) supplies evidence discipline, so cards
  stay short (≤ ~60 lines).

`AgentSpec` (frozen dataclass: name, description, prompt, tools frozenset
| None, skills_scope, delegates, source, path), `parse_agent_card`,
`discover_roster(cwd)` with the §1 precedence. `RosterError(MasonError)`.
Built-in cards live at `src/mason/agents/*.md` and a test asserts all four
parse and that `pi` delegates while specialists do not.

### 2.5 Configuration: overrides only, no new tables

Model names are machine facts and belong in config; cards are portable
content and never name models. `AgentConfig` gains:

```python
delegation: bool = True
roster: Mapping[str, RosterOverride] = {}
```

`RosterOverride` = every connection/budget field of `AgentConfig` as
optionals (`provider`, `endpoint`, `model`, `api_key_env`, `effort`,
`temperature`, `context_window`, `compact_at`, `max_turns`,
`max_reply_tokens`, `request_timeout_s`, `tool_protocol`,
`max_tool_output_chars`, `shell_timeout_s`), `extra="forbid"`. Excluded on
purpose: `approval`, `shell_allowlist` (session security policy has one
owner), `serve` (one server lifecycle), `compute_profile` (machine truth),
`delegation`, `roster` (no nesting).

```toml
[agent]
model = "qwen3-coder:30b"

[agent.roster.pi]
provider = "anthropic"
model = "claude-opus-5"          # big model orchestrates,

[agent.roster.dft-expert]
temperature = 0.0                 # local model executes
```

Effective config for agent `<name>` = the session's `AgentConfig` (CLI
overrides already applied) overlaid with the set fields of
`[agent.roster.<name>]`, rebuilt through `model_validate` so every bound
still binds. To get that for free, move `_override_agent` from `cli.py`
to `mason/config.py` as public `override_agent(agent, updates)` raising
`ConfigError`-style messages; the CLI keeps its flag-naming wrapper. A
`[agent.roster.<name>]` whose name matches no discovered card is refused
when the roster is assembled, naming the known cards (same philosophy as
the moved-key refusal: config that silently does nothing is a trap).
`KNOWN_TOP_LEVEL_KEYS` is untouched; everything is inside `[agent]`.

### 2.6 Toolbox: filtering and the two injected tools

`build_toolbox(session, spec=None, *, depth=0, roster=None, skills=None)`:

1. Build today's box (unchanged construction, so existing tests hold).
2. If `spec.tools` is set, drop tools not named; keep `finish` always.
3. Drop `plan` when `depth > 0`: `PLAN.md` belongs to the turn owner; a
   specialist rewriting the PI's plan mid-delegation is a race, not
   collaboration. Specialists get the notebook (§2.7).
4. Add `skill` when the agent's skill list is non-empty.
5. Add `delegate` when `spec.delegates and depth == 0 and
   session.agent.delegation` and the roster holds at least one other card.

`Toolbox` gains `agent_name`; approval previews from children render as
`[dft-expert] write_file: ...` so the human always knows who is asking.
The universal vocabulary frozenset is defined next to the builders and a
test asserts it matches what an all-features session actually builds.

### 2.7 Delegation

**The tool.** `delegate(agent, task, context?)` — PI-only per §2.6. No
approval on the delegation itself: every mutating action the child takes
still passes the shared gate individually, and read-only errands should
run as freely as reads do today.

**The contract.** The handler:

1. Resolves the card (unknown → the roster's names; self-delegation and
   `delegates: true` targets are allowed but the child never gets the
   tool — depth does the guarding).
2. Builds the effective `AgentConfig` (§2.5) and a child session via
   `MasonSession.spawn(name, agent)`: shares cwd, workspace, hpc,
   approver, `auto_approve`, notebook and plan paths; fresh `read_files`
   (the staleness guard is per-loop — a child must read before editing
   even if the parent read); own transcript
   `<parent-stem>-<agent>-<n>.jsonl` in the same sessions dir; usage
   counters chain to the parent so `/status` and the CLI footer stay
   whole-session truths.
3. Reuses the parent's client object when the effective connection tuple
   (provider, endpoint, model, temperature, effort, max_reply_tokens,
   request_timeout_s) is identical; otherwise `client_from_config`, with
   endpoint discovery through the existing `discover_endpoint`.
4. Runs `Mason(child_session, spec=child_spec, depth=1).run_turn(brief)`
   where the brief is the task plus the optional context paragraph.
5. Returns the child's report with one harness footer line:
   `[dft-expert: finish after 14 steps; tokens 32k+4k; transcript <name>]`.
   Harness stops are returned honestly (`max_turns`, `error_streak`) —
   the PI reads the same truth a human would.

The parent transcript records
`{"type": "delegate", "agent", "task", "transcript", "stop", "steps"}`.

**Depth is a code rule, not a prompt rule:** the `delegate` tool is only
constructed at depth 0. Sequential only — one child at a time — because
the interactive approver cannot multiplex, a shared local server
serializes anyway, and determinism matters more than latency here.
Parallel delegation is future work (§4).

**Shared memory.** The notebook is the blackboard: children read the same
notebook tail at start and append attributed entries (headings gain
`— dft-expert`; compaction summaries likewise). Briefs therefore carry
intent, not state. `--resume` replays only parent-pattern transcripts —
`latest_transcript()` must filter to `\d{8}-\d{6}-\d+\.jsonl` so a child
transcript is never resumed as a conversation (pin with a test).

### 2.8 Prompt architecture

`SYSTEM_PROMPT` splits into layers, ordered by change frequency for
prefix caching, all inside the single system message as today:

1. **Role** — the card body (identity, domain doctrine, and for `pi` the
   orchestration section: delegate separable, context-heavy subtasks;
   brief with goal, constraints, engine/protocol, and what to return —
   numbers with run ids; work solo when the task is small or the model is
   small; never re-delegate a failed delegation unchanged).
2. **Core discipline** — today's evidence/verification/failures/
   scheduler/memory/tool-discipline/honesty sections, identity-free and
   shared by every card (`CORE_PROMPT`).
3. Compute profile block (unchanged).
4. Fenced-protocol block when applicable (unchanged).
5. **Environment** — today's block, plus `# Skills` (one line per visible
   skill) and, for delegating agents, `# Your team` (one line per other
   card: name, description). Session-stable, so caching holds.

The old single-string `SYSTEM_PROMPT` disappears; `system_messages`
gains the spec and precomputed skill/team blocks. `Mason(session)` with no
`spec` resolves the roster and uses `pi` — every existing caller and test
keeps working, now with the PI card's identity (whose body starts from
today's "You are Mason..." text, so the solo prompt is materially the
prompt Mason has today plus the new sections).

### 2.9 CLI

- `mason chat --agent <name>`, `mason run --agent <name>`: entry agent;
  unknown name fails listing the roster. Default `pi`.
- `mason roster`: one line per card — name, source layer, `delegates`
  marker, effective model (after `[agent.roster]` overlay), skill count.
  Configured-but-cardless roster names appear as the §2.5 refusal.
- `mason skills [--agent <name>]`: name, source layer, agents scope,
  script count; notes ignored `allowed-tools`. Both verbs read-only,
  wrapped in the standard three-base error handler.
- `mason doctor` gains a roster tail: the distinct effective
  (provider, endpoint, model) tuples across cards, each probed once —
  a specialist pinned to an unserved model should fail the doctor, not
  the first delegation.

### 2.10 Packaging

- `pyproject.toml`: add `pyyaml>=6` to dependencies; `types-PyYAML` to
  `dev`; add `"*-PLAN.md"` to the sdist excludes (the foundation-split
  review caught the sdist shipping the plan; this plan must not repeat
  it).
- Built-in cards and skills are package data under `src/mason/`;
  hatchling's `packages` setting ships whole directories (the `py.typed`
  precedent). Phase 6 verifies by building a wheel and asserting the
  paths, and `test_layering.py` gains disk-level assertions.
- `[tool.pytest.ini_options]` `addopts` gains
  `--ignore=src/mason/skills` — skill scripts are CLI programs, and
  `--doctest-modules` would import same-named files from different skill
  directories (collection collision). Script tests run them through
  `runpy.run_path` with patched `argv`, which keeps them under coverage.
  mypy strict and ruff still cover them (they are under `src`).

### 2.11 What does not change

The loop internals (compaction, error streak, protocols), the approval
model, `serve`, the Anthropic/OpenAI clients, `foundation` and `slab`
in full, all on-disk workspace formats, transcripts' existing event
types, and the solo default: a user who never adds a card or skill gets
today's Mason with a skills section and four built-in specialists a
`delegate` call away.

## 3. Phases

### Phase 0: baseline and guard rails

1. Branch `roster` from current `main`. Commit this plan file.
2. `pyproject.toml`: sdist excludes gain `"*-PLAN.md"`; deps gain
   `pyyaml>=6` (+ `types-PyYAML` in dev); pytest addopts gain
   `--ignore=src/mason/skills`. `.venv/bin/pip install -e '.[dev,mcp,docs]'`.
3. Gates green (nothing behavioral changed).

### Phase 1: skills

1. `mason/skills.py` per §2.2–2.3, with doctestable pure parsing
   (frontmatter split, name validation, `mason-agents` parsing).
2. `mason/tools.py`: the `skill` tool; universal-vocabulary frozenset.
3. `mason/prompts.py`: `# Skills` block in the environment layer (all
   skills for now; per-card filtering arrives in phase 2).
4. Transcript event `{"type": "skill", ...}`.
5. Seed exactly one built-in skill (`equation-of-state`, §Phase 4 shape)
   so discovery/packaging is exercised end to end from the first phase.
6. `tests/test_mason_skills.py`: spec validation table (name rules,
   dir-name match, description bounds, metadata string-map), precedence
   shadowing across the three layers, `skill` tool output (body + files
   + absolute root), unknown-skill answer names available skills, loud
   `SkillError` on a malformed project skill, `allowed-tools` ignored and
   reported, catalog rendering, prompt contains the section.

Verification: gates; a scripted `FakeClient` turn that activates a skill
and reads a bundled script path.

### Phase 2: the roster

1. `mason/roster.py` per §2.4; built-in cards as *placeholder-quality*
   bodies (phase 4 is the content pass): `pi.md` body starts from today's
   `SYSTEM_PROMPT` identity paragraph + a first orchestration section.
2. `mason/prompts.py`: core/role split per §2.8; team block; skills block
   becomes per-card.
3. `mason/config.py`: `RosterOverride`, `roster`, `delegation`,
   public `override_agent` (CLI rewired to it).
4. `mason/tools.py`: card `tools:` filtering, `finish` always, unknown
   names refused naming the card.
5. `mason/cli.py`: `--agent` on `chat`/`run`; `mason roster`;
   `mason skills`.
6. `tests/test_mason_roster.py` + CLI/config additions: card parsing and
   refusals, precedence (project `pi.md` shadows built-in), tools
   filtering, `skills: all` vs `matching`, `[agent.roster.x]` typo
   refused naming `agent.roster.x.<field>`, unknown roster name refused
   naming known cards, `--agent` unknown lists roster, solo-behavior
   regression: existing loop tests still pass unmodified.

### Phase 3: delegation

1. `MasonSession.spawn` per §2.7 (shared gate/notebook/usage-chain,
   fresh `read_files`, child transcript naming; `latest_transcript`
   parent-pattern filter).
2. `delegate` tool + handler in `tools.py`; `Mason` gains
   `spec`/`depth`; client reuse rule; footer format.
3. Approval preview attribution; notebook heading attribution.
4. `mason doctor` roster tail (§2.9).
5. `tests/test_mason_delegate.py`: scripted PI→specialist round trip
   (PI's second request contains the child's report); child toolbox has
   no `delegate` and no `plan`; `delegation = false` removes the tool;
   usage roll-up; both transcripts exist, parent records the delegate
   event; child harness stop surfaces in the footer; `[agent-name]`
   approval preview; fresh staleness guard; `--resume` ignores child
   transcripts; client reused when tuples match, fresh when model
   differs.
6. `tests/test_mason_real.py`: gated delegation smoke — PI told to
   delegate a one-file read to `analysis-expert` and report the word
   (same tolerant style as the existing real tests).

### Phase 4: built-in content (the domain pass)

The credibility phase: shipped prompts and skills must be true and their
scripts must run. Scripts use ase + numpy + stdlib only, argparse CLIs
with `--json` output, actionable nonzero-exit errors, mypy-strict clean.

1. Cards, full bodies (~40–60 lines each): `pi` (orchestration doctrine
   per §2.8), `dft-expert` (protocol expansion over invented cutoffs,
   convergence ladders, pseudopotential awareness, SCF failure taxonomy),
   `md-expert` (ensembles, timestep/thermostat discipline, equilibration
   before production, provenance of potentials), `analysis-expert`
   (works from recorded runs and artifacts, never re-runs physics to get
   a number it can read, states uncertainties and units).
2. Skills (each body ≤ ~150 lines, procedure + interpretation + the
   `@check`s to add + when *not* to use it):
   - `equation-of-state` (dft, analysis): E(V) workflow template in
     `assets/workflow.py` (runnable as-is under `emt`), and
     `scripts/fit_eos.py` — Birch–Murnaghan via `ase.eos`, in: JSON
     volume/energy pairs or a trajectory, out: V0, E0, B (GPa).
   - `convergence-study` (dft): ladder procedure;
     `scripts/convergence_table.py` renders results JSON into a table
     with a threshold verdict per rung.
   - `radial-distribution` (md, analysis): `scripts/rdf.py <traj>`
     via `ase.geometry.analysis`, bins/rmax flags, plain-text histogram
     plus `--json`.
   - `msd-diffusion` (md, analysis): `scripts/msd.py <traj> --dt-fs`,
     Einstein-relation D with the equilibration-window caveat printed
     next to the number.
   - `surface-energy` (dft): procedure + formula + checks; deliberately
     scriptless.
3. `tests`: every script exercised through `runpy` on tiny
   EMT/synthetic data (fit recovers known parameters within tolerance;
   bad input exits nonzero with the actionable message); the
   `equation-of-state` workflow template runs through `launch_script`
   in a tmp workspace and verifies; every built-in card and skill
   passes its own validator; `test_layering.py` asserts the built-in
   data files exist per package layout.

### Phase 5: documentation

1. New `docs/tutorials/roster-and-skills.md` (+ `mkdocs.yml` nav):
   concept, `--agent`, a delegation walkthrough, authoring a skill
   (spec-compatibility stated), authoring a card, `[agent.roster]`
   overrides, the listing verbs. Captures of `mason roster` and
   `mason skills` are real executions. A delegation transcript needs a
   served model: capture against local Ollama if a capable model is
   available, otherwise mark `<!-- no-verify -->` with the same
   disclosure pattern mason.md already uses — never hand-fake output.
2. `docs/tutorials/mason.md`: rewrite the single-loop provenance bullet
   per §0 (amended in the open, citing the multi-agent research post's
   own conditions, Virtual Lab, Agent Laboratory, AI co-scientist,
   Agent Skills); update "Limitations, honestly stated" (depth 1,
   sequential, no parallel delegation); add skills/roster to the
   harness-discipline section.
3. `mason/__init__.py`: docstring bullet rewritten ("one ReAct loop as
   the unit of work; the roster composes loops one level deep"); export
   `AgentSpec`, `Skill`, `RosterError`, `SkillError`, discovery
   functions.
4. `README.md`: roster + skills in the feature list and the Mason
   section, with a real `mason roster` capture.
5. `ARCHITECTURE.md`: mason section extended (theory register allowed).
6. `CLAUDE.md`: the package table's mason row gains "agent cards and
   the roster, skills, delegation"; one line noting built-in cards and
   skills are package data whose scripts are tested.
7. `docs/tutorials/agents-mcp.md`: one cross-reference sentence
   (external agents via MCP vs the resident roster).
8. Verify every URL added to the docs answers (curl sweep); re-run the
   docs runner on pages with executable blocks.

### Phase 6: verification sweep

1. All four gates on the branch tip.
2. Build the wheel and sdist; assert built-in `agents/*.md`,
   `skills/*/SKILL.md`, and `scripts/*.py` are in the wheel and the
   plan file is *not* in the sdist.
3. Cold-import and heavy-import checks still pass (`yaml` imports at
   `mason.skills` import time only — keep it out of `slab`/`foundation`
   and out of module scope in `mason/__init__`-reachable paths if the
   cold-import test complains).
4. Re-run the full real-model gate if a local model is available.
5. Self-review the diff against §2; then stop for the user's review.
   The user decides on merge and on running a formal `/code-review`.

## 4. Out of scope (recorded, not forgotten)

- **Parallel delegation** (needs a multiplexed approver and per-child
  budget arithmetic; sequential ships first).
- **Inter-specialist messaging or meetings** (Virtual-Lab-style group
  discussion; the notebook-as-blackboard carries shared state for now).
- **A foundation `actor` column** attributing runs to agents (workspace
  schema migration; today the intent text and transcripts carry
  attribution).
- **Serving multiple models at once** (`[agent.roster]` can point
  specialists at different endpoints; orchestrating several serve jobs
  is its own effort).
- **Exposing skills over the MCP server** (external agents bring their
  own skills; revisit if demand appears).
- **A skill scaffold command** (`mason skills new`) — cheap, but content
  before conveniences.
- Streaming, context-folding, and everything else on the existing
  roadmap.

## 5. Gates

Per phase, from the venv, all green before the phase commit:

```
.venv/bin/ruff check .
.venv/bin/mypy
.venv/bin/python -m pytest
.venv/bin/mkdocs build --strict
```

Coverage stays ≥ the configured floor (80) and should stay near the
current 95.7; new modules carry doctests per house style. The gated
`SLAB_TEST_LLM` suite is run whenever a local model is available, and
always before the branch is offered for merge.
