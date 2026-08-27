# Mason: the resident research agent

Mason is a Claude-Code-class agent harness built into SLAB, tuned for
long-running atomistic research projects on **your own hardware** and driven
by **open-weight models**. That means Meta's
[Muse Glimmer](https://huggingface.co/meta-models/Muse-Glimmer-30B) on a
cluster GPU node, Llama on a laptop via Ollama, or anything vLLM serves.
There is no SDK and no API subscription, because Mason uses a stdlib HTTP
client against the OpenAI-compatible surface that every serious open-model
server speaks.

That choice is deliberate, not merely frugal. HPC compute nodes are
frequently firewalled off the internet, where a hosted API is unreachable,
and the GPUs you already have an allocation on are free at the margin. An
agent loop that runs overnight is exactly the workload you do not want
metered. Mason also speaks the
[Anthropic Messages API](#claude-behind-the-same-harness), but that is the
alternative, not the plan.

What makes Mason a research agent rather than a generic coder is the floor
it stands on. Calculations run as SLAB workflow scripts through
`launch_workflow`, so every number Mason reports traces to a run id, its recipe,
and its `@check` assertions. An unverified number is a rumor, and the system
prompt says so.

## On the cluster, end to end

Describe the machine once, as in
[Configuring SLAB for your HPC](hpc-config.md):

```toml
schema_version = 1

[workspace]
root = "/scratch/${USER}/slab-workspace"

[hpc]
cluster = "delta"
account = "abc-123"
default_partition = "cpu"

[hpc.partitions.cpu]
time_limit = "24:00:00"
ntasks_per_node = 64
launcher = "srun"

[hpc.partitions.gpu]
time_limit = "12:00:00"
gres = "gpu:a100:4"

[agent]
model = "meta-models/Muse-Glimmer-30B"
context_window = 131072          # match the server's --max-model-len

[agent.serve]
partition = "gpu"
time_limit = "08:00:00"
tool_call_parser = "llama4_pythonic"
args = ["--tensor-parallel-size 4", "--max-model-len 131072"]
setup = [
  "source $SCRATCH/venvs/vllm/bin/activate",
  "export HF_HOME=$SCRATCH/hf-cache",   # filled by 'hf download' on the login node
  "export HF_HUB_OFFLINE=1",            # compute nodes are firewalled; serve from disk
]
```

The model itself is downloaded once, on the login node, with
`HF_HOME=$SCRATCH/hf-cache hf download meta-models/Muse-Glimmer-30B`,
because compute nodes rarely have internet. `HF_HUB_OFFLINE=1` makes that
arrangement explicit, so a checkpoint missing from the cache is a loud
startup error rather than a download attempt hanging inside a batch job.

The serve job's environment is deliberately its own. `[hpc]`-level `setup`
lines exist to load engine software, and those module stacks fight the
server's venv, so the serve script skips them by default. The partition's
own setup (GPU drivers) still applies, and you can set
`[agent.serve] include_hpc_setup = true` to opt back in. The script also
checks that `vllm` exists right after setup, before it announces an
endpoint. The endpoint record carries the cluster name, and `serve stop`
refuses to `scancel` a record that belongs to a different cluster, because
job ids are only meaningful on their own cluster.

Note what is not there: there is no `endpoint`. The GPU node is the
scheduler's choice, so the URL cannot be written down in advance. It is
discovered.

```bash
mason serve render     # read the batch script before trusting it
mason serve start --wait
mason doctor
mason chat
```

`serve start` submits the server as an ordinary batch job. Its first act on
the node is to write its own endpoint into `<workspace>/mason/endpoint.json`,
and it deletes that record when the server exits, so a dead node can never
keep answering for a live one. `--wait` follows the job to a live endpoint,
and it gives up early if the job dies instead of burning the whole timeout.
This capture and the `doctor` capture below are from a session recorded
before the package split; only the command names were updated:

<!-- no-verify -->
```text
$ mason serve start --wait
submitted job 4242314 (mason-serve) to gpu
script: /scratch/tom/slab-workspace/mason/mason-serve-4242314.sbatch
waiting for the endpoint (up to 1800s)...
node gpu-07.delta.internal announced http://gpu-07.delta.internal:8000/v1; loading...
[+] http://gpu-07.delta.internal:8000/v1 answers; serving: meta-models/Muse-Glimmer-30B
```

Everything downstream then finds the model without being told, and says
where it found it:

<!-- no-verify -->
```text
$ mason doctor
provider: openai
endpoint: http://gpu-07.delta.internal:8000/v1  [job 4242314 on gpu-07.delta.internal]
model:    meta-models/Muse-Glimmer-30B
[+] endpoint answers; 1 model(s) served
[+] model 'meta-models/Muse-Glimmer-30B' is served
[+] native tool calls work
```

That last line is the one to care about. `--tool-call-parser` names a
model-specific parser that your vLLM build either registers or does not, and
a wrong name produces a server that runs and answers but never calls a tool.
Mason does not guess it for you. Rendering a default vLLM command without
`tool_call_parser` set is refused, and the refusal names all three ways out:
set the parser, switch to the fenced text protocol, or give an explicit
`command`. The doctor's probe is the empirical check that the name was
right.

`mason serve status` reports the record, the job's state, and a live
probe, and `mason serve stop` cancels the job and clears the record.
When you want to point Mason at a server you started yourself,
`[agent] endpoint` or `--endpoint` outranks any discovered record, so a
written-down endpoint is never overridden by a background job.

### Where the endpoint comes from

Four sources, highest first:

| source | when |
|---|---|
| `--endpoint` | one command, overriding everything |
| `[agent] endpoint` | you run your own server, or a persistent one exists |
| the serve record | a `mason serve` job is running for this workspace |
| the provider default | `http://localhost:11434/v1` (Ollama), or the Claude API |

The record is the only coupling between the login node and the compute
node, which assumes the workspace sits on a shared filesystem, the normal
arrangement (`/scratch/$USER/...`). When that does not hold, the symptom is
honest: there is no record, and the endpoint falls back to the default
rather than silently pointing somewhere wrong.

A serve job is a job. It has a wall clock, and when the wall clock expires
the agent's model disappears mid-session, so give it a generous
`[agent.serve] time_limit`. Remember, too, that Mason's memory is files.
`NOTEBOOK.md`, `PLAN.md`, and the transcript survive the server, so
`mason chat --resume` after you restart the server picks the project
back up.

## A smaller loop, on a laptop

The same harness runs against Ollama with no cluster in sight, which is how
to try it in two minutes:

```bash
ollama pull llama3.1:8b        # or any tool-calling model
```

```toml
[agent]
endpoint = "http://localhost:11434/v1"
model = "llama3.1:8b"
```

```bash
mason doctor
mason run "..." --auto
```

A real session, with Llama 3.1 8B, locally. The transcript is unedited
apart from the tool and command names, which the package split changed:

<!-- no-verify -->
```text
$ mason run "Relax bulk Cu (fcc, a=3.6) with the emt engine: write a SLAB
  workflow script with a @check that fmax converged below 0.05, run it with
  launch_workflow, then finish with the verified total energy in eV and the run id." --auto

The total energy of bulk Cu is verified to be -0.0067 eV, run id: 01m06tadrfzwg799ecqevcwff7
[finish after 6 step(s); tokens 11016+723; transcript .slab/mason/sessions/20260817-025503-51252.jsonl]
```

The claim survives auditing, which is the point:

<!-- no-verify -->
```text
$ foundation show 01m06tadrfzwg799ecqevcwff7
run 01m06tadrfzwg799ecqevcwff7  workflow
  state:   verified    status: completed
  intent:  relax bulk Cu with emt engine
  checks:  1/1 passed
    [+] forces_converged: residual=4.46589e-15 < 0.05
```

The script it wrote is in the project directory, the energy matches an
independent EMT evaluation to machine precision, and the run's provenance is
ordinary SLAB provenance. Nothing about the result knows an LLM was
involved.

## Compute budget: sizing the physics to the machine

`[agent] compute_profile` tells the agent how big a calculation it may run,
and it is one of `laptop`, `workstation`, or `cluster`. When it is unset,
SLAB derives it: a config that declares SLURM partitions is a cluster, and
anything else is treated as a laptop. That is the conservative guess,
because over-sizing a calculation wastes hours while under-sizing wastes
minutes.

The `cluster` profile allows production settings. It uses the `balanced`
protocol by default, and `stringent` when a result must be publishable, and
it submits anything past a few minutes through `submit_job` rather than
running it on the login node. The `laptop` profile puts hard limits in the
prompt instead. It prefers `emt`, `lj`, and small MACE models, keeps DFT to
single-digit atoms and the `fast` protocol, keeps MD to picoseconds, and
asks before anything expected to run past ten minutes. It also adds an
honesty requirement, because this is where a fast loop can quietly produce
a wrong impression. **Laptop settings are smoke-test settings, and saying so
is part of the result.** Mason records that caveat in the run's `intent`, in
the notebook entry, and in its final report.

The profile shapes what the agent chooses, and it changes no physics on its
own. Every choice it leads to still lands in explicit, traced
`calculator_options` that the run records, so an audit sees the actual
cutoffs and k-mesh rather than a profile name.

The parallelism budget is stated and enforced separately. The environment
block tells the agent how many CPUs the session may use and at how many
MPI ranks the configured engines launch, so scripts and delegated tasks
get sized to real numbers instead of guesses. The enforcement matches the
statement: a `shell` command or `launch_workflow` script that spells out
an `mpirun`, `mpiexec`, or `srun` with more ranks than the CPU budget is
refused as a tool result that names the limit. `submit_job` is exempt,
because its payload runs in its own allocation with its own budget.

## Software notes: curated context for the engines

Without context, a model spends its first steps searching the filesystem for
what `slab` already knows. The system prompt therefore carries a curated
note for each engine this machine enables. A note says what the engine is
for, how to invoke it correctly, and which mistakes it invites. The `mace`
note names the two routes to a served checkpoint, the `qe` note points at
the named protocols instead of hand-picked cutoffs, and the `lammps` note
says where the real error message hides.

Selection follows the configuration. The always-available engines (`emt`,
`lj`, `mace`) load their notes unconditionally. The engines that need a
table (`qe`, `lammps`, `rootstock`) load their notes when `slab.toml`
configures `[engines.<name>]`. Set `[agent] software_notes = false` to turn
the block off. The notes are starting context only: they grant no
capability, and `list_engines` stays the live inventory.

If your software has machine-local quirks, you can replace a note: a file at
`~/.config/slab/notes/<engine>.md` wins over the packaged note, whole-file.
This is an escape hatch for local tweaks, not a content system. Keep a
replacement short, because the note rides in every request's context.

## The tool surface

Mason has few, orthogonal tools with crisp machine-checkable failure modes.
That is the SWE-agent lesson, that interface design drives agent performance
more than model choice.

| tool | contract |
|---|---|
| `read_file` | line-numbered, windowed; refuses binary; a file must be read before it may be edited |
| `write_file` / `edit_file` | edit is exact-string replacement, unique match or `replace_all`; Python files get an immediate syntax check after every write |
| `list_dir`, `search` | listing and recursive regex search, output-capped |
| `shell` | one command, merged output + exit code, timeout-capped; **not** for long calculations |
| `launch_workflow` | run a workflow script as a traced, check-gated run; this is how physics happens |
| `list_runs`, `show_run`, `list_engines` | the workspace's evidence surface: runs, checks with observed/expected values, failure records, capabilities |
| `submit_job`, `job_status`, `cancel_job` | SLURM plumbing, present only when the config declares partitions |
| `notebook`, `plan` | the memory instruments (below) |
| `skill` | load a skill: its instructions, root path, and bundled files; the catalog is per-agent |
| `delegate` | hand one scoped task to a specialist's own loop; the PI only, one level deep, sequential |
| `finish` | end the task with a report citing run ids; honored only as the sole call of its reply, and only with a report |

The last three belong to the roster: Mason is a research group of agent
cards with per-specialist skills, described in
[The roster and skills](roster-and-skills.md).

Every tool failure is returned as the tool result, as evidence the model
reads, and never as an exception that kills the loop. Mutating tools pass
through an approval gate. Interactively, Mason asks, while `--auto` (or
`[agent] approval = "auto"`) trusts them. `shell_allowlist` prefixes
auto-approve at word boundaries, but a command that contains shell control
operators (`;`, `|`, `&`, redirection, and so on) never auto-approves.

Plan the gate before an interactive session. `write_file`, `edit_file`,
`shell`, `launch_workflow`, `submit_job`, and `cancel_job` ask, and everything
else (reads, `list_engines`, `job_status`, the memory instruments) never
does. A multi-step goal therefore prompts several times, every shell probe
included, because the default allowlist is empty. At the prompt, **Enter
refuses**, because the default is "no", and five straight refusals abort
the turn. Either answer `y` deliberately, run with `--auto`, or set
`[agent] shell_allowlist` to your read-only probes so only the real
mutations ask. `mason run` without `--auto` refuses every mutating
tool, so batch use needs `--auto`.

In chat, Mason prints the model's reasoning between tool calls, dimmed
and prefixed `[reasoning]`, so each approval prompt arrives with its
rationale above it. Interim assistant text prints the same way, and a
delegated specialist's output carries its name. The reasoning stream
needs a reasoning parser on the server, which the `mason serve` template
names next to the tool-call parser. Set `[agent] show_reasoning = false`
to hide the display. The transcript records every reasoning trace as its
own event either way, and `--resume` never replays reasoning into the
model's context. `mason run` prints only the final report.

The file fence bounds where the file tools work. With the default
`[agent] file_scope = "project"`, the reading tools (`read_file`,
`list_dir`, `search`, and `launch_workflow`) reach the project directory,
the workspace, and the skill directories the session advertises. The
writing tools (`write_file`, `edit_file`) reach only the project and the
workspace. Paths are compared after symlinks resolve, so a link that
points out of the fence counts as outside it. A refused path comes back
as a tool result that names the fence and the setting, and
`file_scope = "anywhere"` lifts it.

While the fence is on, the prompt also tells the model to treat it as the
working area: stay inside the project directory and the workspace, do not
probe other locations with the shell, and ask before touching a path
outside. The fence blocks the file tools mechanically. The prompt guidance
covers the shell, which the fence never bounded, so an approval prompt for
an out-of-bounds `ls` should now arrive with a reason or not at all.

Job scripts and SLURM output files from `submit_job` land in
`<workspace>/jobs/`, not in the project directory. The job itself still
runs in the project directory. The workspace sits inside the file fence,
so the agent reads its own `.out` files with `read_file`. When a job is
finished, `slab-stack purge` sweeps its files from there (see
[Lifecycle & retention](lifecycle-and-retention.md)).

The session lock keeps one running loop per workspace. With the default
`[agent] session_lock = true`, a second Mason loop in the same workspace
is refused with a message that names the holding process, because two
loops interleave `NOTEBOOK.md` and race the plan. Delegated specialists
run inside the parent's lock. On a filesystem that cannot hold an
advisory lock, the lock degrades to a warning, and
`session_lock = false` turns it off.

## The sandbox: autonomous runs without a network

`--auto` removes the approval gate, so the boundary for an unattended run
must come from the operating system. `mason sandbox render` writes a batch
job that provides that boundary. The job runs `mason run --auto` inside an
Apptainer container with an empty network namespace, no home directory, a
clean environment, and file access limited to explicit bind mounts. The
shell tool then reaches only what the fence was always meant to bound.

The model stays reachable through exactly one path. On the host side of the
job, `mason sandbox bridge` relays a unix socket to the recorded serve
endpoint. Inside the container, `mason sandbox forward` relays that socket
to `127.0.0.1:8000`, and the agent talks to it as a normal endpoint. The
destination is fixed at job start, so the agent cannot redirect it. Both
halves are plain Python from the installed package, so the host needs no
relay tool.

The job fails closed. Before the agent starts, `mason sandbox verify` runs
inside the container and proves two things: a public URL is unreachable,
and the bridged endpoint lists its models. Either proof failing aborts the
job before the first turn.

The render derives the bind mounts from the configuration it already has:
the project directory and the workspace read-write, the scratch root
read-write, the pseudopotential root, the engine registry, the rootstock
install, and the Python environment read-only. `[agent.sandbox]` holds
only what derivation cannot see: the container `image`, and extra `binds`
for engine installs and their library closures (run `ldd` on the engine
binary to find them). Because everything specific to a machine comes from
that machine's own config, nothing site-specific ever needs to enter a
repository.

Run `mason sandbox check` first. It reports whether the container runtime
and unprivileged network namespaces exist here, and whether the image and
the serve record are in place. Then render, read both files, and submit:

```
mason sandbox render "the goal" --partition cpu
sbatch sandbox/mason-sandbox.sbatch
```

Engine `setup` lines get snapshotted. A `module load` works on the host
and means nothing inside the container, so the render runs each engine's
setup once, on the host, and records what it did: the resolved binaries —
the payload and the launcher its command references, since `mpirun` must
be bound as surely as `pw.x` — the environment it changed, and each
binary's library closure from `ldd`. The
snapshot becomes bind mounts in the script and explicit `export` lines in
the rendered `slab.toml`. List variables such as `PATH` keep only the
components the setup added, so the container's own base value survives
underneath. Your real config keeps its module loads as the source of
truth. The snapshot is frozen at render time, so re-render after a module
changes, and check the `[=]` line the render prints for each snapshot.
A setup that fails to snapshot keeps a warning naming the cause.

Two consequences to plan for. The rendered `slab.toml` has no `[hpc]`
table, because the namespace has no route to the scheduler — so the
scheduler tools do not exist, and calculations run inside the job's own
allocation. Size the job for its engine legs, and name QE by its install
(`[engines.qe] bin`), which sizes `mpirun` to the allocation and binds the
install automatically; the render warns when a hand-written command uses
`srun`. And because no `[hpc]` would otherwise derive a `laptop` compute
profile, the rendered config pins `compute_profile = "workstation"` — the
honest size for one owned node — unless your own config sets a profile.

## Memory that outlives the context window

Long projects die of context, not of model quality. Models degrade well
before their window fills (Chroma's "context rot" measurements), and no
window survives weeks. Mason's memory is **files in the project directory**,
under version control, readable by humans:

* **`NOTEBOOK.md`** is an append-only lab notebook that holds decisions,
  verified results with run ids, and diagnosed failures, written for a
  colleague who has read none of the conversation.
* **`PLAN.md`** is the living plan, rewritten by the `plan` tool as
  understanding changes. The tool echoes the full plan back into context,
  which is the "recitation" trick that holds long goals stable.
* **`.slab/mason/sessions/*.jsonl`** are append-only transcripts of every
  message, tool result, compaction, and token count, and
  `mason chat --resume` replays the newest one.
* **`AGENTS.md`** is the cross-tool conventions standard, and if the project
  has one, it enters the system prompt every session.

When the conversation approaches the budget (`compact_at` × `context_window`,
default 70%), the middle of the history is folded into a structured summary
of state, verified results, failures observed, decisions, and open
questions. The summary is also written to the notebook, and the system
context is rebuilt fresh so the current plan and notebook re-enter updated.
A context-overflow answer from the server forces the same compaction
immediately. Failures are deliberately carried forward, because evidence of
what went wrong is what keeps a model from repeating it.

## Open-model realism

Open-weight tool calling is uneven, and the harness plans for it:

* **Native tool calls** are the default, which means
  `vllm serve ... --enable-auto-tool-choice --tool-call-parser <family>`,
  the command that `mason serve` renders. Ollama parses natively.
* **`tool_protocol = "fenced"`** switches to a plain-text protocol, with one
  fenced ````tool```` block per message, for servers with no tool-call parser
  at all. That is the mini-swe-agent lesson, that a text protocol is the
  great equalizer. Set it, and `serve` stops passing the parser flags.
* Either way, the loop also catches the llama-style
  `{"name": ..., "parameters": {...}}` that models leak into message text
  even when served with a parser, and runs it. A well-shaped call that names
  a hallucinated tool gets the tool catalog back as its answer instead of a
  dead end, and malformed JSON arguments come back as a repair prompt.

Every one of those recoveries was exercised by a real model during
development, not imagined. `tool_choice` is never sent, because Ollama
ignores it, and nothing is constrained-decoded, because harness-side
validation with repair re-prompts costs less than the documented tool-call
suppression that strict constrained decoding causes on open models.

One failure mode is worth naming because it is invisible. A small model can
run a calculation correctly, verify it, cite the right run id, and then
**mistype the number in its prose**. That happened here, with Llama 3.1 8B,
when a run whose recorded energy was `-0.0015020475862299598` eV was
reported as "-0.15 eV". The run was genuinely `verified`, and its value
matched an independent EMT evaluation exactly, so only the sentence was
wrong. This is why traceability, not model accuracy, is what SLAB
guarantees, because `foundation show <run id>` had the right number the whole
time. The prompt now requires numbers to be copied from run output rather
than retyped, which fixed it on the same model and goal, but the audit
trail is the actual defense. Prefer the largest model your hardware serves
for anything you intend to quote.

Model sizing, honestly stated. 8B-class models (the laptop loop above)
handle scripted, well-specified goals and need explicit sequencing in
prompts, while research judgment needs the strongest model your allocation
serves. That means Muse-Glimmer-30B-class on a single GPU node, or
`gpt-oss-120b`, GLM/Qwen3-class MoEs, or DeepSeek V4-Flash across several.

Mason's prompts are layered for prefix caching (static core, then
per-session environment, then the append-only conversation), so a long
session reuses the server's KV cache turn after turn.

## Claude behind the same harness

The loop is provider-agnostic, so the same tools, notebook, and verification
gates run against the Anthropic Messages API:

```toml
[agent]
provider = "anthropic"
model = "claude-opus-5"
api_key_env = "ANTHROPIC_API_KEY"   # the default for this provider
effort = "medium"                   # low | medium | high | xhigh | max
compute_profile = "laptop"          # keep the physics small while iterating
```

Three things differ from the open-model path, and Mason handles all three.
Sampling parameters are never sent, because current Claude models reject
`temperature` outright, so **`[agent] temperature` applies to the
OpenAI-compatible provider only**, and `effort` is the equivalent knob.
`max_tokens` is required, and it bounds thinking plus reply together, so a
truncated turn is reported rather than passed off as a finished answer. And
the stable system prompt is sent with a cache breakpoint.

Two caveats, both load-bearing:

**This path needs billed API access, which a Claude subscription does not
include.** Claude.ai and Claude Code subscriptions are separate products from
API credit, so `$ANTHROPIC_API_KEY` has to come from an API account with
billing enabled. If `mason doctor --provider anthropic` reports an
authentication failure on a valid-looking key, that is usually why.

**Developing exclusively against Claude lets the open-model path rot
silently.** Every open-model bug fixed so far is something Claude would never
have done, from llama-style tool JSON leaking into message text, to useless
missing-argument errors, to Python written with literal `\n`, to `finish`
emitted in the same reply as the tool calls whose results its report
pretends to cite (the harness now honors `finish` only alone, and only
with a report). Keep the gated `SLAB_TEST_LLM` test as the acceptance
gate before Mason changes land; the `finish` rule was caught exactly
there, when a model update started batching its calls.

## Harness discipline, in code

The limits live in the harness, because prompts do not enforce invariants.
There is a `max_turns` model-call budget per goal, and an abort after five
consecutive harness-level tool failures, with the evidence left in place.
The prompt sets bounded diagnose-then-retry expectations, and
required-argument validation answers with the tool's schema instead of a
stack trace. Token usage is accounted per turn from the server's own numbers
and recorded in the transcript.

## Design provenance

Mason is a deliberate distillation of the 2024–2026 agent-harness literature
onto SLAB's philosophy. The load-bearing choices and their sources:

* **One ReAct-style loop as the unit of work** ([Yao et al. 2022,
  arXiv:2210.03629](https://arxiv.org/abs/2210.03629)). The first release
  omitted multi-agent orchestration entirely, citing the finding that
  orchestrator-worker systems pay off on breadth-first search, not tightly
  interdependent work, at ~15× the tokens ([Anthropic, *Building a
  multi-agent research system*](https://www.anthropic.com/engineering/multi-agent-research-system)).
  [The roster](roster-and-skills.md) amends that decision in the open, for
  the case the same report endorses: context isolation for *separable*
  subtasks — a convergence ladder fills a context with tables when only its
  conclusion matters upstream. The composition stays deliberately austere
  (one level deep, sequential, delegation always optional, a config
  switch to off), and the PI-plus-specialists shape is now the domain
  norm: Agent Laboratory below, the Virtual Lab's PI agent and specialist
  scientists with experimentally validated designs ([Swanson et al.
  2024](https://doi.org/10.1101/2024.11.11.623004)), and
  Google's AI co-scientist ([Gottweis et al. 2025,
  arXiv:2502.18864](https://arxiv.org/abs/2502.18864)).
* **Skills instead of regenerated code** ([Anthropic, *Equipping agents
  for the real world with Agent Skills*](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)),
  in the open [Agent Skills format](https://agentskills.io/specification),
  adopted verbatim: procedures and tested scripts ship as files an agent
  loads on demand, with the spec's progressive disclosure keeping the
  always-loaded cost to one line per skill. The shared notebook doubles as
  the group's blackboard, the oldest multi-agent pattern there is
  (Hearsay-II, Erman et al. 1980).
* **Compaction + structured note-taking + file memory** as the three
  context-pollution countermeasures ([Anthropic, *Effective context engineering
  for AI agents*](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)).
  Compaction triggers well below the window because degradation starts early
  ([Chroma, *Context Rot*](https://research.trychroma.com/context-rot)), and
  condensation measurably improves task success, not just cost
  ([OpenHands condenser](https://openhands.dev/blog/openhands-context-condensensation-for-more-efficient-ai-agents)).
* **Progress files, plans, and git as the recovery substrate** for
  multi-session work ([Anthropic, *Effective harnesses for long-running
  agents*](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)).
  OS-style externalized memory tiers trace to MemGPT
  ([arXiv:2310.08560](https://arxiv.org/abs/2310.08560)).
* **Append-only, cache-stable context; recitation; keep failures in context**
  ([Manus, *Context engineering for AI
  agents*](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)).
* **Small tool set with crisp failure modes**, because agent-computer
  interface design drives performance ([SWE-agent,
  arXiv:2405.15793](https://arxiv.org/abs/2405.15793)). The fenced text
  protocol follows
  [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)'s result that a
  ~100-line bash-only loop stays competitive.
* **Verbal self-reflection after failures** (Reflexion,
  [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)) is Mason's
  diagnose-before-retry rule. Deterministic checks come before any
  model-judged verification, which is SLAB's `@check` philosophy anyway.
* **Fixed tool libraries + human checkpoints** outperform full autonomy in
  scientific-agent studies (Agent Laboratory,
  [arXiv:2501.04227](https://arxiv.org/abs/2501.04227)), hence the approval
  gate, and the promotion step staying human.

The roadmap direction, folding sub-trajectories at workflow-step boundaries
rather than summarizing linearly, follows the context-folding line
([arXiv:2510.11967](https://arxiv.org/abs/2510.11967),
[arXiv:2510.24699](https://arxiv.org/abs/2510.24699)).

## Limitations, honestly stated

Mason's roster delegates one level deep and sequentially: no parallel
delegation, no specialist-to-specialist messaging, no recursive teams.
Delegation quality is bounded by the served model, and an 8B-class model
handles only small, explicit briefs. Mason does not stream tokens,
because an agent loop consumes whole turns. Its judgment is the served
model's, so SLAB guarantees that what Mason reports is traceable and
verified, not that its research taste is good. And the approval gate is a
workflow control for your own account on your own machine, not a security
sandbox, so `--auto` means what it says — for every agent on the roster.
The file fence and the session lock share that caveat: they shrink the
blast radius of a confused agent, and the `shell` tool remains the honest
escape, behind its own gate. When you want a real boundary, use the
sandbox: `mason sandbox render` writes a batch job that runs the session
in a container with no network, no home, and only the configured
directories bound — see
[The sandbox](#the-sandbox-autonomous-runs-without-a-network).

What has and has not been exercised against reality, precisely:

* **Verified against a real model.** The full loop ran against Llama 3.1 8B
  served by Ollama, and an autonomous Cu relaxation reached `verified` with
  a reported energy that matches an independent EMT evaluation exactly.
* **Verified without a cluster.** The serve path's rendered script is
  executed by `bash` in the test suite with a stub server on PATH, which
  proves the record it writes is readable and that the exit trap clears it.
  Discovery, waiting, `status`/`stop`, and a complete goal driven through a
  discovered endpoint all run against a live local server. Discovery was
  also exercised hand-to-hand against a real Ollama, with a record written
  where a serve job would write one, after which `doctor`, `serve status`,
  and two autonomous Al relaxations reached `verified` with no `endpoint`
  configured anywhere. What no test here can cover is a real `sbatch` on a
  real GPU node, so the first `mason serve start` on your cluster is
  that test, which is why `serve render` exists.
* **Not verified against the live API.** The Anthropic provider is tested
  against a mock that reproduces the documented Messages wire shape,
  including a Cu relaxation driven end to end through it. But no live call
  has been made, because the development machine has no billed API access.
  The gated test is written and waiting:

<!-- no-verify -->
```bash
ANTHROPIC_API_KEY=sk-ant-... pytest tests/test_mason_real.py
```
