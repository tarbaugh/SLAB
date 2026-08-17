# Mason: the resident research agent

*The mason works the slab.* Mason is a Claude-Code-class agent harness
built into SLAB and tuned for one job: long-running atomistic research
projects on your own hardware, driven by **open-weight models** — Meta's
[Muse Glimmer](https://huggingface.co/meta-models/Muse-Glimmer-30B) or
Llama on a laptop via Ollama, anything vLLM serves on a compute node. No
SDK, no API subscription: a stdlib HTTP client against the
OpenAI-compatible surface every serious open-model server speaks.

What makes it a *research* agent rather than a generic coder is the floor
it stands on: calculations run as SLAB workflow scripts through
`slab_launch`, so every number Mason reports traces to a run id, its
recipe, and its `@check` assertions. An unverified number is a rumor, and
the system prompt says so.

## Quickstart (laptop, Ollama)

```bash
ollama pull llama3.1:8b        # or any tool-calling model
```

Point `[agent]` at it in `slab.toml` (see
[Configuring SLAB for your HPC](hpc-config.md)):

```toml
[agent]
endpoint = "http://localhost:11434/v1"
model = "llama3.1:8b"
```

Check the plumbing, then talk to it:

```bash
slab mason doctor
slab mason chat
```

`doctor` verifies the three things that actually go wrong: the endpoint
answers, the model is served, and tool calls come back parsed. Captured
against a real local Ollama:

<!-- no-verify -->
```text
endpoint: http://localhost:11434/v1
model:    llama3.1:8b
[+] endpoint answers; 2 model(s) served
[+] model 'llama3.1:8b' is served
[+] native tool calls work
```

For an autonomous goal, `slab mason run` loops until the model calls
`finish`, answers in text, or a harness limit stops it. A real session —
Llama 3.1 8B, locally, unedited:

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

The script it wrote is in the project directory, the energy matches an
independent EMT evaluation to machine precision, and the run's provenance
is ordinary SLAB provenance — nothing about the result knows an LLM was
involved.

## The tool surface

Few, orthogonal tools with crisp machine-checkable failure modes (the
SWE-agent lesson: interface design drives agent performance more than
model choice):

| tool | contract |
|---|---|
| `read_file` | line-numbered, windowed; refuses binary; a file must be read before it may be edited |
| `write_file` / `edit_file` | edit is exact-string replacement, unique match or `replace_all`; Python files get an immediate syntax check after every write |
| `list_dir`, `search` | listing and recursive regex search, output-capped |
| `shell` | one command, merged output + exit code, timeout-capped; **not** for long calculations |
| `slab_launch` | run a workflow script as a traced, check-gated run — how physics happens |
| `slab_runs`, `slab_show`, `slab_engines` | the workspace's evidence surface: runs, checks with observed/expected values, failure records, capabilities |
| `submit_job`, `job_status`, `cancel_job` | SLURM plumbing, present only when the config declares partitions |
| `notebook`, `plan` | the memory instruments (below) |
| `finish` | end the task with a report citing run ids |

Every tool failure is returned as the tool *result* — evidence the model
reads — never an exception that kills the loop. Mutating tools pass
through an approval gate: interactively Mason asks; `--auto` (or
`[agent] approval = "auto"`) trusts them; `shell_allowlist` prefixes
auto-approve at word boundaries, and a command containing shell control
operators (`;`, `|`, `&`, redirection...) never auto-approves.

## Memory that outlives the context window

Long projects die of context, not of model quality — models degrade well
before their window fills (Chroma's "context rot" measurements), and no
window survives weeks. Mason's memory is **files in the project
directory**, under version control, readable by humans:

* **`NOTEBOOK.md`** — an append-only lab notebook. Decisions, verified
  results with run ids, diagnosed failures; written for a colleague who
  has read none of the conversation.
* **`PLAN.md`** — the living plan, rewritten by the `plan` tool as
  understanding changes. The tool echoes the full plan back into context
  (the "recitation" trick that holds long goals stable).
* **`.slab/mason/sessions/*.jsonl`** — append-only transcripts: every
  message, tool result, compaction, and token count. `slab mason chat
  --resume` replays the newest one.
* **`AGENTS.md`** — the cross-tool conventions standard; if the project
  has one, it enters the system prompt every session.

When the conversation approaches the budget (`compact_at` ×
`context_window`, default 70%), the middle of the history is folded into a
structured summary — state, verified results, failures observed,
decisions, open questions — the summary is *also written to the notebook*,
and the system context is rebuilt fresh so the current plan and notebook
re-enter updated. A context-overflow answer from the server forces the
same compaction immediately. Failures are deliberately carried forward:
evidence of what went wrong is what keeps a model from repeating it.

## Open-model realism

Open-weight tool calling is uneven, and the harness plans for it:

* **Native tool calls** are the default (`vllm serve ... --enable-auto-tool-choice
  --tool-call-parser <family>`; Ollama parses natively).
* **`tool_protocol = "fenced"`** switches to a plain-text protocol — one
  fenced ````tool`` block per message — for servers with no tool-call
  parser at all (the mini-swe-agent lesson: a text protocol is the great
  equalizer).
* Either way, the loop also catches the llama-style
  `{"name": ..., "parameters": {...}}` that models leak into message text
  even when served with a parser, and runs it. A well-shaped call naming a
  hallucinated tool gets the tool catalog back as its answer instead of a
  dead end. Malformed JSON arguments come back as a repair prompt.

Every one of those recoveries was exercised by a real model during
development, not imagined. `tool_choice` is never sent (Ollama ignores
it), and nothing is constrained-decoded — harness-side validation with
repair re-prompts costs less than the documented tool-call suppression
that strict constrained decoding causes on open models.

Model sizing, honestly: 8B-class models (the quickstart) handle scripted,
well-specified goals and need explicit sequencing in prompts; research
judgment wants the strongest model your hardware serves —
Muse-Glimmer-30B-class on a workstation GPU, `gpt-oss-120b`,
GLM/Qwen3-class MoEs, or DeepSeek V4-Flash on a multi-GPU node.

## On the cluster

Serve a model on a GPU node, point the config at it, and let long
calculations go through the scheduler:

```bash
vllm serve meta-models/Muse-Glimmer-30B \
    --enable-auto-tool-choice --tool-call-parser llama4_pythonic \
    --enable-prefix-caching
```

```toml
[agent]
endpoint = "http://gpu-node-01:8000/v1"
model = "meta-models/Muse-Glimmer-30B"
```

Mason's prompts are layered for prefix caching (static core → per-session
environment → append-only conversation), so a long session reuses the
server's KV cache turn after turn. With `[hpc]` partitions configured, the
`submit_job` tool appears and the system prompt teaches the split: quick
things in-process, anything long as `slab run workflow.py` under sbatch,
polled with `job_status` — never busy-waited.

## Harness discipline, in code

The limits live in the harness because prompts don't enforce invariants:
a `max_turns` model-call budget per goal, an abort after five consecutive
harness-level tool failures (with the evidence left in place), bounded
diagnose-then-retry expectations in the prompt, and required-argument
validation that answers with the tool's schema instead of a stack trace.
Token usage is accounted per turn from the server's own numbers and
recorded in the transcript.

## Design provenance

Mason is a deliberate distillation of the 2024–2026 agent-harness
literature onto SLAB's philosophy; the load-bearing choices and their
sources:

* **Single ReAct-style loop** ([Yao et al. 2022, arXiv:2210.03629](https://arxiv.org/abs/2210.03629)),
  no default multi-agent orchestration — orchestrator-worker systems pay
  off on breadth-first search, not tightly interdependent research work,
  at ~15× token cost ([Anthropic, *Building a multi-agent research
  system*](https://www.anthropic.com/engineering/multi-agent-research-system)).
* **Compaction + structured note-taking + file memory** as the three
  context-pollution countermeasures ([Anthropic, *Effective context
  engineering for AI agents*](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents));
  compaction triggers well below the window because degradation starts
  early ([Chroma, *Context Rot*](https://research.trychroma.com/context-rot));
  condensation measurably *improves* task success, not just cost
  ([OpenHands condenser](https://openhands.dev/blog/openhands-context-condensensation-for-more-efficient-ai-agents)).
* **Progress files, plans, and git as the recovery substrate** for
  multi-session work ([Anthropic, *Effective harnesses for long-running
  agents*](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents));
  OS-style externalized memory tiers trace to MemGPT
  ([arXiv:2310.08560](https://arxiv.org/abs/2310.08560)).
* **Append-only, cache-stable context; recitation; keep failures in
  context** ([Manus, *Context engineering for AI
  agents*](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)).
* **Small tool set with crisp failure modes** — agent-computer interface
  design drives performance ([SWE-agent,
  arXiv:2405.15793](https://arxiv.org/abs/2405.15793)); the fenced text
  protocol follows
  [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)'s result
  that a ~100-line bash-only loop stays competitive.
* **Verbal self-reflection after failures** (Reflexion,
  [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)) — Mason's
  diagnose-before-retry rule; deterministic checks before any
  model-judged verification, which is SLAB's `@check` philosophy anyway.
* **Fixed tool libraries + human checkpoints** outperform full autonomy
  in scientific-agent studies (Agent Laboratory,
  [arXiv:2501.04227](https://arxiv.org/abs/2501.04227)) — hence the
  approval gate and the promotion step staying human.

The roadmap direction — folding sub-trajectories at workflow-step
boundaries rather than summarizing linearly — follows the context-folding
line ([arXiv:2510.11967](https://arxiv.org/abs/2510.11967),
[arXiv:2510.24699](https://arxiv.org/abs/2510.24699)).

## Limitations, honestly stated

Mason is single-loop: no subagents yet. It does not stream tokens (an
agent loop consumes whole turns). Its judgment is the served model's —
SLAB guarantees that what Mason *reports* is traceable and verified, not
that its research taste is good. And the approval gate is a workflow
control for your own account on your own machine, not a security sandbox:
`--auto` means what it says.
