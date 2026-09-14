# Machine memory

Mason records facts about the machine it runs on. A package behaves
unlike its documentation, a flag turns out to matter, a workaround takes
an afternoon to find. The agent writes the fact once, and every later
session on that machine reads it.

This page covers what belongs in memory, the file format, the evidence a
memory needs, the agent tools, what a delegate hands back, the sandbox,
and the commands you use to read, review, and prune the store.

## Four places knowledge lives

SLAB keeps four kinds of knowledge, and each has one job.

| Surface | Scope | Written by | Holds |
|---|---|---|---|
| Software notes | machine | the package | curated engine knowledge |
| `NOTEBOOK.md` | project | agents | the scientific record |
| Runs and artifacts | project | workflows | results with provenance |
| Memory | machine | agents | learned quirks of this machine |

Memory is the machine-scoped one an agent may write. Put a fact in memory
when it is true of the machine and not of the project, so a different
project on the same cluster still needs it. Keep results in runs, keep
project decisions in the notebook, and keep credentials out of all four.

## Where memories live

Memories live in `~/.config/slab/memory/`, beside the user's skills and
agent cards. `$XDG_CONFIG_HOME` is honored, and `$SLAB_MEMORY_DIR`
overrides both. The directory appears on the first write.

```bash
slab memory path
```

```text
/Users/you/.config/slab/memory
```

Each memory is one markdown file named for the memory. The frontmatter
carries the description and the provenance, and the body carries the
fact:

```bash
slab memory show vllm-mamba-cache
```

```text
---
description: vLLM refuses to start a hybrid-Mamba model at the default max_num_seqs on one
  80 GB GPU. Read before serving one.
created: 2026-09-14
updated: 2026-09-14
agent: pi
model: qwen3-30b
evidence: run 01m2gex59fs1yxcw5qvy4zstqa failed at the default; 32 served
against:
  slab-stack: 0.1.0
---
A hybrid attention/Mamba model reserves one fixed Mamba cache block per concurrent decode sequence, allocated at startup. After the weights load, the memory left under gpu_memory_utilization fits far fewer blocks than the default max_num_seqs of 1024, and CUDA graph capture aborts naming the number that fits.

Mason needs a handful of concurrent sequences, not a thousand. Set [agent.serve] args = ["--max-num-seqs", "32"], which also leaves more memory for the KV cache and so for longer contexts.
```

The rules the store enforces:

- The file name is the memory's name. Use lowercase letters, digits, and
  single hyphens, at most 64 characters.
- `description` is required, and at most 1024 characters. Write the fact
  and the condition it applies under, because this line is what a later
  session reads first.
- The body is required, and at most 4000 characters. A memory states one
  fact. Split a longer one, or fold it into an existing memory.
- One machine holds at most 100 memories.
- `evidence` is what confirmed the fact: a run id with one line, a dry
  run, or a failure record, at most 500 characters. A write without it is
  refused unless the writer marks the fact `unverified: true`. See
  [Evidence](#evidence).
- `created`, `updated`, `agent`, and `model` are provenance the store
  writes. You may edit any file by hand. A file you write yourself needs
  only the description and the body, and it reads as unverified until it
  carries evidence.
- `against` is the version stamp: the software the memory names, at the
  versions present when it was written. The `remember` tool writes it.
  See [Version stamps](#version-stamps).
- Replacing a memory keeps the old file under `.history/<name>/`, up to
  ten versions. The catalog never reads that directory.

A malformed file is an error that names the file and the rule. It is
never skipped, because a memory that vanished from the catalog would be
undebuggable.

## How the agent uses memory

The system prompt carries one line per memory: the name and the
description. The fact itself stays on disk until the agent asks for it:

```text
# Memory

Facts earlier sessions recorded about this machine and its software. Call the recall tool with a name before you rely on one: the line below is a summary, and the memory itself holds the detail. Each memory is stamped with the versions of the software it names. A line that reports a change since, a newer version or a tool not found now, is a memory you must confirm before you build on it. A line that reports none names software that is unchanged, so rely on that memory without probing. A line marked [unverified] is a claim no run confirmed; test it before you rely on it. When you find a quirk of this machine or its software worth keeping, record it with remember once a run has confirmed it, and cite that run as the evidence. Machine facts only: results belong in runs, project decisions in the notebook, and credentials nowhere.

- mace-model-inside-the-fence: The sandbox cannot reach ~/.cache, so a MACE model file must live in the project directory. [unverified]
- vllm-mamba-cache: vLLM refuses to start a hybrid-Mamba model at the default max_num_seqs on one 80 GB GPU. Read before serving one.
```

The `recall` tool returns the fact, who recorded it, and the evidence.
When the evidence names a run, `recall` also reads that run's state now
from the workspace:

```text
A hybrid attention/Mamba model reserves one fixed Mamba cache block per concurrent decode sequence, allocated at startup. After the weights load, the memory left under gpu_memory_utilization fits far fewer blocks than the default max_num_seqs of 1024, and CUDA graph capture aborts naming the number that fits.

Mason needs a handful of concurrent sequences, not a thousand. Set [agent.serve] args = ["--max-num-seqs", "32"], which also leaves more memory for the KV cache and so for longer contexts.

[recorded by pi on 2026-09-14, model qwen3-30b, against slab-stack 0.1.0, evidence: run 01m2gex59fs1yxcw5qvy4zstqa failed at the default; 32 served]

[evidence runs now: 01m2gex59fs1yxcw5qvy4zstqa: serve-probe, failed, quarantined, error: CUDA graph capture aborted: max_num_seqs 1024 exceeds the 212 Mamba cache blocks that fit]
```

The `remember` tool writes one. It takes the name, the description, the
body, and the evidence, and it answers with where the fact landed:

```text
recorded as memory 'vllm-mamba-cache' in /Users/you/.config/slab/memory/vllm-mamba-cache.md; every later session on this machine reads it (stamped against slab-stack 0.1.0); the evidence runs now: 01m2gex59fs1yxcw5qvy4zstqa: serve-probe, failed, quarantined, error: CUDA graph capture aborted: max_num_seqs 1024 exceeds the 212 Mamba cache blocks that fit
```

`remember` is a mutating tool, so it passes the approval gate like every
other one. The preview shows the whole text and names the agent that
asks, because what it writes enters every later session's prompt.

Re-using a name replaces that memory. The creation date survives, the
update date moves, and the new writer, evidence, and version stamp are
recorded. The replaced file goes into the memory's history. When the
body changed, `recall` shows the previous body with its date and its
evidence, so a contradiction is visible to the next reader:

```text
Export SLURM_CONF in the job script; srun then starts ranks inside the container.

[recorded by md-expert on 2026-09-14, model qwen3-30b, evidence: run 01m2geyzf31q6zq6kk74wr0z3e completed under srun]

[evidence runs now: 01m2geyzf31q6zq6kk74wr0z3e: srun-probe, completed, quarantined]

[this memory was replaced on 2026-09-14; the version before it, from 2026-09-14 (no evidence), said:]
Use mpirun inside the container.
```

The `forget` tool undoes a memory written in the current session, by the
agent or by an agent it briefed. A memory that is new this session is
removed. A memory that existed before goes back to the version it had
then, and the rejected version stays in the history. `forget` refuses a
memory from an earlier session, because that one is the person's to
remove.

Set `[agent] memory = false` to run a session that neither reads nor
writes memories. The block and the three tools disappear, and the store
keeps what it holds.

## Evidence

A memory enters every later session's prompt, so a wrong one misleads
every session after it. Most wrong memories share one cause. The agent
wrote the fact before any run confirmed it. So `remember` requires
evidence:

- a run id, with one line saying what the run showed
- a dry run of the script that showed it
- a failure record, when the failure is the fact

Without evidence the write is refused:

```text
not recorded: a memory needs evidence: the run id, the dry run, or the failure record that confirmed the fact. Pass evidence, or pass unverified=true to record it as an unverified claim that recall flags and 'slab memory review' lists
```

An agent that has a lead worth keeping but no confirmation passes
`unverified=true`. The store writes `unverified: true`, the catalog line
ends with `[unverified]`, and `recall` says so first:

```text
unverified: no run confirmed this memory. Test it before you rely on it, then remember it again with the run as its evidence.

The container runs with --no-home, so ~/.cache is not visible inside it. Copy the model file into the project directory and pass that path.

[recorded by md-expert on 2026-08-28, model qwen3-30b, no evidence recorded]
```

A memory with no `evidence` field reads as unverified, whatever else its
frontmatter says. That covers every memory written before this rule and
every file written by hand without the field.

The store keeps the evidence as the writer gave it. It does not check
the claim. The readers check it. `recall`, the delegate list, and the MCP
`recall` read each run id in the evidence and report that run's state
now. A cited run that failed, or one the workspace does not hold, is a
reason to doubt the memory.

## What a delegate hands back

A delegated specialist can write memories that its lead never sees in
the report. So the harness appends a list to the delegate result. The
list names every memory the specialist and its own delegates wrote, with
the evidence and the state of each cited run:

```text
The order is settled: masses, then pair_style grace.

[memories written: read each one, and forget any its evidence does not support]
- masses-before-grace (md-expert): masses must come before pair_style grace in this build. evidence: run 01m2gey0s4fcb88yd9vpba7vab [runs now: 01m2gey0s4fcb88yd9vpba7vab: grace-order-probe, failed, quarantined, error: ERROR: Invalid atom type in probe.data]
- newton-before-read-data (md-expert): newton on must precede read_data. evidence: none [unverified]

[md-expert: finish after 3 step(s); tokens 300+30; transcript 20260914-172013-40505-md-expert-1.jsonl]
```

The planner and the PI cards tell the lead to read each entry before the
next brief, and to forget a memory whose evidence is missing, names a
failed run, or does not show the fact. Here the cited run failed on a
hand-written probe file, so the lead forgets that memory:

```text
forgot 'masses-before-grace'; it did not exist before this session
```

The delegate's transcript record also lists the names under `memories`.

## Memory in the sandbox

A sandbox job runs the container with `--containall --no-home`, so
`~/.config` is not visible inside it. `slab mason sandbox render` therefore
binds the memory directory read-write and names it in the environment:

```text
--bind /Users/you/.config/slab/memory:/Users/you/.config/slab/memory:rw \
--env SLAB_MEMORY_DIR=/Users/you/.config/slab/memory \
```

The render creates the directory first, because Apptainer refuses a bind
whose source is missing.

This is what memory is for on a cluster. An overnight job hits a quirk at
03:00, records it, and finishes. The next job starts knowing it. A
session rendered with `[agent] memory = false` binds nothing.

## Version stamps

A memory states what was true when it was written. Software changes
between one session and the next, and the fact can stop holding without
anyone noticing. Version stamps let a later session tell which memories
to re-check, and trust the rest without probing.

When `remember` writes a memory, it stamps it with the software the
text names and the versions present at that moment. The software it
knows is `slab-stack` itself, every engine whose version can be told,
the `mp` snapshot by its release, and a configured `atomsk` or
`gracemaker`. Names are matched as whole words, with a few aliases:
`pw.x` counts as `qe`, `lmp` as `lammps`, `grace` and `tensorpotential`
as `gracemaker`, and `slab`, `foundation`, and `mason` as `slab-stack`.
A memory about vLLM gets no stamp, so a gracemaker upgrade never flags
it.

```bash
slab memory show grace-gpu-growth
```

```text
---
description: gracemaker fits on the GPU nodes need TF_FORCE_GPU_ALLOW_GROWTH=true. Set it
  before a second fit on one node.
created: 2026-09-14
updated: 2026-09-14
agent: pi
model: qwen3-30b
evidence: 'run 01m2gf1cshv2npt1vv0nr1fm8z: two fits on one node completed with the export
  set'
against:
  gracemaker: 0.6.0
  slab-stack: 0.1.0
---
Without it TensorFlow reserves the whole card at startup, and a second fit on the same node fails to allocate. Put the export in [builders.gracemaker] setup in slab.toml so every fit gets it.
```

At the start of every session the catalog compares each stamp with the
machine. The versions are probed once per session, and only when some
memory carries a stamp. A memory whose software changed since gets a
note on its catalog line:

```text
- grace-gpu-growth: gracemaker fits on the GPU nodes need TF_FORCE_GPU_ALLOW_GROWTH=true. Set it before a second fit on one node. [changed since: gracemaker was 0.6.0, now 0.7.0]
```

`recall` repeats the note after the provenance, and asks the agent to
confirm the fact and record it again once it has:

```text
Without it TensorFlow reserves the whole card at startup, and a second fit on the same node fails to allocate. Put the export in [builders.gracemaker] setup in slab.toml so every fit gets it.

[recorded by pi on 2026-09-14, model qwen3-30b, against gracemaker 0.6.0, slab-stack 0.1.0, evidence: run 01m2gf1cshv2npt1vv0nr1fm8z: two fits on one node completed with the export set]

[evidence runs now: 01m2gf1cshv2npt1vv0nr1fm8z: grace-fit-pair, completed, quarantined]

[changed since: gracemaker was 0.6.0, now 0.7.0. Confirm the fact before you build on it; remember it again once you have.]
```

A stamp that names software the machine no longer reports reads as
"not found now". A failed probe and a removed tool look the same from
the stamp, and either is a reason to re-check.

`slab memory list` prints the same note, so you can see which memories
an upgrade has put in question before you prune. A memory you write by
hand carries no stamp unless you add one. Without a stamp the catalog
makes no claim about it either way.

## Reading and pruning the store

`slab memory list` prints the catalog: the name, the date, the
agent that recorded it, and the description. An unverified memory ends
with `[unverified]`.

```bash
slab memory list
```

```text
mace-model-inside-the-fence  2026-08-28  md-expert         The sandbox cannot reach ~/.cache, so a MACE model file must live in the project directory. [unverified]
newton-before-read-data      2026-09-14  md-expert         newton on must precede read_data. [unverified]
vllm-mamba-cache             2026-09-14  pi                vLLM refuses to start a hybrid-Mamba model at the default max_num_seqs on one 80 GB GPU. Read before serving one.
3 memory(s) in /Users/you/.config/slab/memory
```

Add `--json` for the same catalog with the full provenance of each
memory, the evidence included.

`slab memory review` lists only the memories a person should check. A
memory is listed when it is unverified, or when software it is stamped
against has changed since it was written. The reasons follow each line:

```bash
slab memory review
```

```text
mace-model-inside-the-fence  2026-08-28  md-expert         The sandbox cannot reach ~/.cache, so a MACE model file must live in the project directory. [unverified]
newton-before-read-data      2026-09-14  md-expert         newton on must precede read_data. [unverified]
2 of 3 memory(s) to review: 'slab memory confirm <name> --evidence ...' or 'slab memory forget <name>'
```

Test each fact, then take one of two actions:

- If the fact holds, run `slab memory confirm <name> --evidence ...`.
  The body stays as it is. The memory gets the evidence, loses the
  unverified mark, and is stamped against the software present now.
- If the fact does not hold, run `slab memory forget <name>`.

```bash
slab memory confirm mace-model-inside-the-fence --evidence "a relax with the model under the project dir completed; the ~/.cache path failed"
```

```text
confirmed mace-model-inside-the-fence: evidence a relax with the model under the project dir completed; the ~/.cache path failed
```

`slab memory add <name> <description> <body>` writes a memory yourself,
under the same rules as `remember`. Give `--evidence`, or `--unverified`
for a claim. A body of `-` is read from standard input.

```bash
slab memory add srun-in-sandbox "srun cannot reach the controller inside the sandbox." "Use mpirun inside the container."
```

```text
Error: a memory needs evidence: the run id, the dry run, or the failure record that confirmed the fact. Pass --evidence, or pass --unverified to record it as a claim that recall flags and 'slab memory review' lists
```

`slab memory forget <name>` deletes one memory. It prints what it
is about to delete and asks first, and `--yes` skips the question.

```bash
slab memory forget mace-model-inside-the-fence
```

```text
mace-model-inside-the-fence: The sandbox cannot reach ~/.cache, so a MACE model file must live in the project directory.
permanently delete /Users/you/.config/slab/memory/mace-model-inside-the-fence.md? [y/N]:
```

`slab memory purge` deletes every memory that matches, in one
confirmed step. Give it shell-style globs against the names, or no
pattern to select everything. Add `--before YYYY-MM-DD` to keep recent
memories, and `--yes` to skip the question. Use it after a change that
makes a family of memories stale, such as a SLAB fix that retires the
workarounds agents recorded.

```bash
slab memory purge 'rootstock-*' 'foundation-relax-cell-*'
```

```text
foundation-relax-cell-ase329-exp-cell-factor: relax_cell crashes on ASE 3.29 for cells with more than 1 atom.
rootstock-missing-grace-smax-checkpoint: The rootstock install does not declare a grace checkpoint.
permanently delete these 2 of 3 memory(s)? [y/N]: y
purged 2 memory(s) from /Users/you/.config/slab/memory
```

`forget` and `purge` are the only ways a memory from an earlier session
leaves the machine. Both remove the memory's history too.
`slab purge` (without `memory`) deletes project state that nobody
promoted, and it does not touch memories, which are durable machine
state that outlives every project.

Read the store when a machine changes. A memory states what was true when
it was written, so an upgraded engine, a new scheduler, or a rebuilt
container can leave one stale. The version stamps flag the memories
whose software changed, and `slab memory review` lists them beside the
unverified ones. Confirm what holds and forget what does not.
