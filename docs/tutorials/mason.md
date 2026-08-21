# Mason: the resident research agent

Mason is a Claude-Code-class agent harness built into SLAB. It is tuned for
long-running atomistic research projects on **your own hardware**, driven by
**open-weight models**. That means Meta's
[Muse Glimmer](https://huggingface.co/meta-models/Muse-Glimmer-30B) on a
cluster GPU node, Llama on a laptop via Ollama, or anything vLLM serves.
There is no SDK and no API subscription. Mason uses a stdlib HTTP client
against the OpenAI-compatible surface that every serious open-model server
speaks.

That choice is deliberate, not merely frugal. HPC compute nodes are
frequently firewalled off the internet, where a hosted API is unreachable.
The GPUs you already have an allocation on are free at the margin. And an
agent loop that runs overnight is exactly the workload you do not want
metered. Mason also speaks the
[Anthropic Messages API](#claude-behind-the-same-harness), but that is the
alternative, not the plan.

What makes Mason a research agent rather than a generic coder is the floor
it stands on. Calculations run as SLAB workflow scripts through
`slab_launch`, so every number Mason reports traces to a run id, its recipe,
and its `@check` assertions. An unverified number is a rumor, and the system
prompt says so.

## On the cluster, end to end

Describe the machine once. See
[Configuring SLAB for your HPC](hpc-config.md):

```toml
schema_version = 1

[paths]
workspace = "/scratch/${USER}/slab-workspace"

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
`HF_HOME=$SCRATCH/hf-cache hf download meta-models/Muse-Glimmer-30B`.
Compute nodes rarely have internet. `HF_HUB_OFFLINE=1` makes that
arrangement explicit. A checkpoint missing from the cache is a loud startup
error, not a download attempt hanging inside a batch job.

The serve job's environment is deliberately its own. `[hpc]`-level `setup`
lines exist to load engine software, and those module stacks fight the
server's venv. So the serve script skips them by default. The partition's
own setup (GPU drivers) still applies. Set
`[agent.serve] include_hpc_setup = true` to opt back in. The script also
checks that `vllm` exists right after setup, before it announces an
endpoint. The endpoint record carries the cluster name. `serve stop` refuses to
`scancel` a record that belongs to a different cluster, because job ids are
only meaningful on their own cluster.

Note what is not there. There is no `endpoint`. The GPU node is the scheduler's
choice, so the URL cannot be written down in advance. It is discovered.

```bash
slab mason serve render     # read the batch script before trusting it
slab mason serve start --wait
slab mason doctor
slab mason chat
```

`serve start` submits the server as an ordinary batch job. Its first act on
the node is to write its own endpoint into `<workspace>/mason/endpoint.json`.
It deletes that record when the server exits, so a dead node can never keep
answering for a live one. `--wait` follows the job to a live endpoint. It
gives up early if the job dies, instead of burning the whole timeout:

<!-- no-verify -->
```text
$ slab mason serve start --wait
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
$ slab mason doctor
provider: openai
endpoint: http://gpu-07.delta.internal:8000/v1  [job 4242314 on gpu-07.delta.internal]
model:    meta-models/Muse-Glimmer-30B
[+] endpoint answers; 1 model(s) served
[+] model 'meta-models/Muse-Glimmer-30B' is served
[+] native tool calls work
```

That last line is the one to care about. `--tool-call-parser` names a
model-specific parser that your vLLM build either registers or does not. A
wrong name produces a server that runs and answers but never calls a tool.
Mason does not guess it for you. Rendering a default vLLM command without
`tool_call_parser` set is refused, and the refusal names all three ways out.
Set the parser, switch to the fenced text protocol, or give an explicit
`command`. The doctor's probe is the empirical check that the name was
right.

`slab mason serve status` reports the record, the job's state, and a live
probe. `slab mason serve stop` cancels the job and clears the record. When
you want to point Mason at a server you started yourself, `[agent] endpoint`
or `--endpoint` outranks any discovered record. A written-down endpoint is
never overridden by a background job.

### Where the endpoint comes from

Four sources, highest first:

| source | when |
|---|---|
| `--endpoint` | one command, overriding everything |
| `[agent] endpoint` | you run your own server, or a persistent one exists |
| the serve record | a `slab mason serve` job is running for this workspace |
| the provider default | `http://localhost:11434/v1` (Ollama), or the Claude API |

The record is the only coupling between the login node and the compute
node. That assumes the workspace sits on a shared filesystem, which is the
normal arrangement (`/scratch/$USER/...`). When it does not hold, the
symptom is honest. There is no record, and the endpoint falls back to the
default rather than silently pointing somewhere wrong.

A serve job is a job. It has a wall clock, and when the wall clock expires
the agent's model disappears mid-session. Give it a generous
`[agent.serve] time_limit`. And remember that Mason's memory is files.
`NOTEBOOK.md`, `PLAN.md`, and the transcript survive the server, so
`slab mason chat --resume` after you restart the server picks the project
back up.

## A smaller loop, on a laptop

The same harness runs against Ollama with no cluster in sight. That is how
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
slab mason doctor
slab mason run "..." --auto
```

A real session, with Llama 3.1 8B, locally, unedited:

<!-- no-verify -->
```text
$ slab mason run "Relax bulk Cu (fcc, a=3.6) with the emt engine: write a SLAB
  workflow script with a @check that fmax converged below 0.05, run it with
  slab_launch, then finish with the verified total energy in eV and the run id." --auto

The total energy of bulk Cu is verified to be -0.0067 eV, run id: 01m06tadrfzwg799ecqevcwff7
[finish after 6 step(s); tokens 11016+723; transcript .slab/mason/sessions/20260817-025503-51252.jsonl]
```

The claim survives auditing, which is the point:

<!-- no-verify -->
```text
$ slab show 01m06tadrfzwg799ecqevcwff7
run 01m06tadrfzwg799ecqevcwff7  workflow
  state:   verified    status: completed
  intent:  relax bulk Cu with emt engine
  checks:  1/1 passed
    [+] forces_converged: residual=4.46589e-15 < 0.05
```

The script it wrote is in the project directory. The energy matches an
independent EMT evaluation to machine precision. And the run's provenance is
ordinary SLAB provenance. Nothing about the result knows an LLM was
involved.

## Compute budget: sizing the physics to the machine

`[agent] compute_profile` tells the agent how big a calculation it may run.
It is one of `laptop`, `workstation`, or `cluster`. When it is unset, SLAB
derives it. A config that declares SLURM partitions is a cluster. Anything
else is treated as a laptop. That is the conservative guess, because
over-sizing a calculation wastes hours while under-sizing wastes minutes.

The `cluster` profile allows production settings. It uses the `balanced`
protocol by default, and `stringent` when a result must be publishable. It
submits anything past a few minutes through `submit_job` rather than running
it on the login node. The `laptop` profile puts hard limits in the prompt
instead. It prefers `emt`, `lj`, and small MACE models. It keeps DFT to
single-digit atoms and the `fast` protocol, and MD to picoseconds. It asks
before anything expected to run past ten minutes. It also adds an honesty
requirement, because this is where a fast loop can quietly produce a wrong
impression. **Laptop settings are smoke-test settings, and saying so is part
of the result.** Mason records that caveat in the run's `intent`, in the
notebook entry, and in its final report.

The profile shapes what the agent chooses. It changes no physics on its own.
Every choice it leads to still lands in explicit, traced
`calculator_options` that the run records, so an audit sees the actual
cutoffs and k-mesh rather than a profile name.

## The tool surface

Mason has few, orthogonal tools with crisp machine-checkable failure modes.
That is the SWE-agent lesson. Interface design drives agent performance more
than model choice.

| tool | contract |
|---|---|
| `read_file` | line-numbered, windowed; refuses binary; a file must be read before it may be edited |
| `write_file` / `edit_file` | edit is exact-string replacement, unique match or `replace_all`; Python files get an immediate syntax check after every write |
| `list_dir`, `search` | listing and recursive regex search, output-capped |
| `shell` | one command, merged output + exit code, timeout-capped; **not** for long calculations |
| `slab_launch` | run a workflow script as a traced, check-gated run; this is how physics happens |
| `slab_runs`, `slab_show`, `slab_engines` | the workspace's evidence surface: runs, checks with observed/expected values, failure records, capabilities |
| `submit_job`, `job_status`, `cancel_job` | SLURM plumbing, present only when the config declares partitions |
| `notebook`, `plan` | the memory instruments (below) |
| `finish` | end the task with a report citing run ids |

Every tool failure is returned as the tool result, as evidence the model
reads. It is never an exception that kills the loop. Mutating tools pass
through an approval gate. Interactively, Mason asks. `--auto` (or
`[agent] approval = "auto"`) trusts them. `shell_allowlist` prefixes
auto-approve at word boundaries. A command that contains shell control
operators (`;`, `|`, `&`, redirection, and so on) never auto-approves.

Plan the gate before an interactive session. `write_file`, `edit_file`,
`shell`, `slab_launch`, `submit_job`, and `cancel_job` ask. Everything else
(reads, `slab_engines`, `job_status`, the memory instruments) never does. A
multi-step goal therefore prompts several times, every shell probe included,
because the default allowlist is empty. At the prompt, **Enter refuses**.
The default is "no", and five straight refusals abort the turn. Either
answer `y` deliberately, run with `--auto`, or set `[agent] shell_allowlist`
to your read-only probes so only the real mutations ask. `slab mason run`
without `--auto` refuses every mutating tool. Batch use needs `--auto`.

## Memory that outlives the context window

Long projects die of context, not of model quality. Models degrade well
before their window fills (Chroma's "context rot" measurements), and no
window survives weeks. Mason's memory is **files in the project directory**,
under version control, readable by humans:

* **`NOTEBOOK.md`** is an append-only lab notebook. It holds decisions,
  verified results with run ids, and diagnosed failures. It is written for a
  colleague who has read none of the conversation.
* **`PLAN.md`** is the living plan, rewritten by the `plan` tool as
  understanding changes. The tool echoes the full plan back into context.
  That is the "recitation" trick that holds long goals stable.
* **`.slab/mason/sessions/*.jsonl`** are append-only transcripts of every
  message, tool result, compaction, and token count.
  `slab mason chat --resume` replays the newest one.
* **`AGENTS.md`** is the cross-tool conventions standard. If the project has
  one, it enters the system prompt every session.

When the conversation approaches the budget (`compact_at` × `context_window`,
default 70%), the middle of the history is folded into a structured summary.
The summary holds state, verified results, failures observed, decisions, and
open questions. The summary is also written to the notebook, and the system
context is rebuilt fresh so the current plan and notebook re-enter updated.
A context-overflow answer from the server forces the same compaction
immediately. Failures are deliberately carried forward. Evidence of what
went wrong is what keeps a model from repeating it.

## Open-model realism

Open-weight tool calling is uneven, and the harness plans for it:

* **Native tool calls** are the default. That is
  `vllm serve ... --enable-auto-tool-choice --tool-call-parser <family>`,
  which is what `slab mason serve` renders. Ollama parses natively.
* **`tool_protocol = "fenced"`** switches to a plain-text protocol, with one
  fenced ````tool```` block per message, for servers with no tool-call parser
  at all. That is the mini-swe-agent lesson. A text protocol is the great
  equalizer. Set it, and `serve` stops passing the parser flags.
* Either way, the loop also catches the llama-style
  `{"name": ..., "parameters": {...}}` that models leak into message text
  even when served with a parser, and runs it. A well-shaped call that names
  a hallucinated tool gets the tool catalog back as its answer instead of a
  dead end. Malformed JSON arguments come back as a repair prompt.

Every one of those recoveries was exercised by a real model during
development, not imagined. `tool_choice` is never sent, because Ollama
ignores it. Nothing is constrained-decoded. Harness-side validation with
repair re-prompts costs less than the documented tool-call suppression that
strict constrained decoding causes on open models.

One failure mode is worth naming because it is invisible. A small model can
run a calculation correctly, verify it, cite the right run id, and then
**mistype the number in its prose**. That happened here, with Llama 3.1 8B.
A run whose recorded energy was `-0.0015020475862299598` eV was reported as
"-0.15 eV". The run was genuinely `verified`, and its value matched an
independent EMT evaluation exactly. Only the sentence was wrong. This is why
traceability, not model accuracy, is what SLAB guarantees. `slab show <run
id>` had the right number the whole time. The prompt now requires numbers to
be copied from run output rather than retyped, which fixed it on the same
model and goal. But the audit trail is the actual defense. Prefer the
largest model your hardware serves for anything you intend to quote.

Model sizing, honestly stated. 8B-class models (the laptop loop above) handle
scripted, well-specified goals and need explicit sequencing in prompts.
Research judgment needs the strongest model your allocation serves. That
means Muse-Glimmer-30B-class on a single GPU node, or `gpt-oss-120b`,
GLM/Qwen3-class MoEs, or DeepSeek V4-Flash across several.

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
`temperature` outright. So **`[agent] temperature` applies to the
OpenAI-compatible provider only**, and `effort` is the equivalent knob.
`max_tokens` is required, and it bounds thinking plus reply together, so a
truncated turn is reported rather than passed off as a finished answer. And
the stable system prompt is sent with a cache breakpoint.

Two caveats, both load-bearing:

**This path needs billed API access, which a Claude subscription does not
include.** Claude.ai and Claude Code subscriptions are separate products from
API credit. `$ANTHROPIC_API_KEY` has to come from an API account with billing
enabled. If `slab mason doctor --provider anthropic` reports an
authentication failure on a valid-looking key, that is usually why.

**Developing exclusively against Claude lets the open-model path rot
silently.** Every open-model bug fixed so far is something Claude would never
have done. The list so far: llama-style tool JSON leaking into message text,
useless missing-argument errors, and Python written with literal `\n`. Keep the
gated `SLAB_TEST_LLM` test as the acceptance gate before Mason changes land.

## Harness discipline, in code

The limits live in the harness, because prompts do not enforce invariants.
There is a `max_turns` model-call budget per goal. There is an abort after
five consecutive harness-level tool failures, with the evidence left in
place. The prompt sets bounded diagnose-then-retry expectations. And
required-argument validation answers with the tool's schema instead of a
stack trace. Token usage is accounted per turn from the server's own numbers
and recorded in the transcript.

## Design provenance

Mason is a deliberate distillation of the 2024–2026 agent-harness literature
onto SLAB's philosophy. The load-bearing choices and their sources:

* **Single ReAct-style loop** ([Yao et al. 2022, arXiv:2210.03629](https://arxiv.org/abs/2210.03629)),
  with no default multi-agent orchestration. Orchestrator-worker systems pay
  off on breadth-first search, not tightly interdependent research work, and
  they cost ~15× the tokens ([Anthropic, *Building a multi-agent research
  system*](https://www.anthropic.com/engineering/multi-agent-research-system)).
* **Compaction + structured note-taking + file memory** as the three
  context-pollution countermeasures ([Anthropic, *Effective context engineering
  for AI agents*](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)).
  Compaction triggers well below the window because degradation starts early
  ([Chroma, *Context Rot*](https://research.trychroma.com/context-rot)).
  Condensation measurably improves task success, not just cost
  ([OpenHands condenser](https://openhands.dev/blog/openhands-context-condensensation-for-more-efficient-ai-agents)).
* **Progress files, plans, and git as the recovery substrate** for
  multi-session work ([Anthropic, *Effective harnesses for long-running
  agents*](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)).
  OS-style externalized memory tiers trace to MemGPT
  ([arXiv:2310.08560](https://arxiv.org/abs/2310.08560)).
* **Append-only, cache-stable context; recitation; keep failures in context**
  ([Manus, *Context engineering for AI
  agents*](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)).
* **Small tool set with crisp failure modes.** Agent-computer interface design
  drives performance ([SWE-agent,
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
  [arXiv:2501.04227](https://arxiv.org/abs/2501.04227)). Hence the approval
  gate, and the promotion step staying human.

The roadmap direction, folding sub-trajectories at workflow-step boundaries
rather than summarizing linearly, follows the context-folding line
([arXiv:2510.11967](https://arxiv.org/abs/2510.11967),
[arXiv:2510.24699](https://arxiv.org/abs/2510.24699)).

## Limitations, honestly stated

Mason is single-loop, with no subagents yet. It does not stream tokens,
because an agent loop consumes whole turns. Its judgment is the served
model's. SLAB guarantees that what Mason reports is traceable and verified,
not that its research taste is good. And the approval gate is a workflow
control for your own account on your own machine, not a security sandbox.
`--auto` means what it says.

What has and has not been exercised against reality, precisely:

* **Verified against a real model.** The full loop ran against Llama 3.1 8B
  served by Ollama. An autonomous Cu relaxation reached `verified`, and its
  reported energy matches an independent EMT evaluation exactly.
* **Verified without a cluster.** The serve path's rendered script is
  executed by `bash` in the test suite with a stub server on PATH. That
  proves the record it writes is readable and that the exit trap clears it.
  Discovery, waiting, `status`/`stop`, and a complete goal driven through a
  discovered endpoint all run against a live local server. Discovery was
  also exercised hand-to-hand against a real Ollama. A record was written
  where a serve job would write one, then `doctor`, `serve status`, and two
  autonomous Al relaxations reached `verified`, with no `endpoint` configured
  anywhere. What no test here can cover is a real `sbatch` on a real GPU
  node. The first `slab mason serve start` on your cluster is that test,
  which is why `serve render` exists.
* **Not verified against the live API.** The Anthropic provider is tested
  against a mock that reproduces the documented Messages wire shape,
  including a Cu relaxation driven end to end through it. But no live call
  has been made, because the development machine has no billed API access.
  The gated test is written and waiting:

<!-- no-verify -->
```bash
ANTHROPIC_API_KEY=sk-ant-... pytest tests/test_mason_real.py
```
