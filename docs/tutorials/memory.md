# Machine memory

Mason records facts about the machine it runs on. A package behaves
unlike its documentation, a flag turns out to matter, a workaround takes
an afternoon to find. The agent writes the fact once, and every later
session on that machine reads it.

This page covers what belongs in memory, the file format, the two agent
tools, the sandbox, and the commands you use to read and prune the store.

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
slab-stack memory path
```

```text
/Users/you/.config/slab/memory
```

Each memory is one markdown file named for the memory. The frontmatter
carries the description and the provenance, and the body carries the
fact:

```bash
slab-stack memory show vllm-mamba-cache
```

```text
---
description: vLLM refuses to start a hybrid-Mamba model at the default max_num_seqs on one
  80 GB GPU. Read before serving one.
created: 2026-08-28
updated: 2026-08-28
agent: pi
model: qwen3-30b
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
- `created`, `updated`, `agent`, and `model` are provenance the store
  writes. You may edit any file by hand, and a file you write yourself
  needs only the description and the body.

A malformed file is an error that names the file and the rule. It is
never skipped, because a memory that vanished from the catalog would be
undebuggable.

## How the agent uses memory

The system prompt carries one line per memory: the name and the
description. The fact itself stays on disk until the agent asks for it:

```text
# Memory

Facts earlier sessions recorded about this machine and its software. Call the recall tool with a name before you rely on one: the line below is a summary, and the memory itself holds the detail. Each reflects the machine when it was written, so when one names a flag, a path, or a version, confirm it still holds before you build on it. When you find a quirk of this machine or its software worth keeping, record it with remember once you have confirmed it. Machine facts only: results belong in runs, project decisions in the notebook, and credentials nowhere.

- vllm-mamba-cache: vLLM refuses to start a hybrid-Mamba model at the default max_num_seqs on one 80 GB GPU. Read before serving one.
```

The `recall` tool returns the fact and who recorded it:

```text
A hybrid attention/Mamba model reserves one fixed Mamba cache block per concurrent decode sequence, allocated at startup. After the weights load, the memory left under gpu_memory_utilization fits far fewer blocks than the default max_num_seqs of 1024, and CUDA graph capture aborts naming the number that fits.

Mason needs a handful of concurrent sequences, not a thousand. Set [agent.serve] args = ["--max-num-seqs", "32"], which also leaves more memory for the KV cache and so for longer contexts.

[recorded by pi on 2026-08-28, model qwen3-30b]
```

The `remember` tool writes one. It takes the name, the description, and
the body, and it answers with where the fact landed:

```text
recorded as memory 'vllm-mamba-cache' in /Users/you/.config/slab/memory/vllm-mamba-cache.md; every later session on this machine reads it
```

`remember` is a mutating tool, so it passes the approval gate like every
other one. The preview shows the whole text and names the agent that
asks, because what it writes enters every later session's prompt.

Re-using a name replaces that memory. The creation date survives, the
update date moves, and the new writer is recorded. This is how an agent
consolidates two related facts into one. An agent cannot delete a memory.

Set `[agent] memory = false` to run a session that neither reads nor
writes memories. The block and both tools disappear, and the store keeps
what it holds.

## Memory in the sandbox

A sandbox job runs the container with `--containall --no-home`, so
`~/.config` is not visible inside it. `mason sandbox render` therefore
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

## Reading and pruning the store

`slab-stack memory list` prints the catalog: the name, the date, the
agent that recorded it, and the description.

```bash
slab-stack memory list
```

```text
mace-model-inside-the-fence  2026-08-28  md-expert         The sandbox cannot reach ~/.cache, so a MACE model file must live in the project directory.
vllm-mamba-cache             2026-08-28  pi                vLLM refuses to start a hybrid-Mamba model at the default max_num_seqs on one 80 GB GPU. Read before serving one.
2 memory(s) in /Users/you/.config/slab/memory
```

Add `--json` for the same catalog with the full provenance of each
memory.

`slab-stack memory forget <name>` deletes one memory. It prints what it
is about to delete and asks first, and `--yes` skips the question.

```bash
slab-stack memory forget mace-model-inside-the-fence
```

```text
mace-model-inside-the-fence: The sandbox cannot reach ~/.cache, so a MACE model file must live in the project directory.
permanently delete /Users/you/.config/slab/memory/mace-model-inside-the-fence.md? [y/N]:
```

This is the only way a memory leaves the machine. `slab-stack purge`
deletes project state that nobody promoted, and it does not touch
memories, which are durable machine state that outlives every project.

Read the store when a machine changes. A memory states what was true when
it was written, so an upgraded engine, a new scheduler, or a rebuilt
container can leave one stale. Prune what no longer holds.
