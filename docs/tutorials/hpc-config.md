# Configuring SLAB for your HPC

Everything machine-specific in SLAB — paths, the right `pw.x`, SLURM
partitions, the resident agent's model endpoint — lives in **layered TOML
configuration**, not in code. One file describes a cluster; the same
workflow scripts run unchanged everywhere.

## The three layers

Files merge key-by-key, lowest precedence first:

| layer | where | who writes it |
|---|---|---|
| **site** | the file `$SLAB_SITE_CONFIG` points at | the cluster maintainer, exported from a module file |
| **user** | `~/.config/slab/config.toml` (`$XDG_CONFIG_HOME` honored) | you, once per machine |
| **project** | `./slab.toml`, or the file `$SLAB_CONFIG` points at | the project, in version control |

A scalar in a higher layer replaces the lower one; deeper tables merge.
The explicit environment stays above all of it: `$SLAB_WORKSPACE`,
`$SLAB_PSEUDOS`, `$SLAB_ENGINES`, and explicit function arguments override
any file. Start from the fully commented template:

```bash
slab config init          # writes ./slab.toml
slab config init --user   # writes ~/.config/slab/config.toml
```

The same template is committed at
[`templates/slab.toml`](https://github.com/tarbaugh/SLAB/tree/main/templates),
next to an engine-registry counterpart (`templates/engines.json`) — copy,
fill in the blanks, keep local. A SLAB checkout gitignores every file named
`slab.toml` or `engines.json` (only the templates themselves are exempt), so
a filled-in machine file kept next to the code never leaves the cluster.

`slab config show` prints the merged result **with each value's origin**,
so "why is it using that partition?" is a one-command question:

<!-- no-verify -->
```text
site: /sw/slab/config.toml
project: /home/tom/cu-project/slab.toml
  engines.qe.command = 'srun pw.x'  [/sw/slab/config.toml (site)]
  hpc.default_partition = 'cpu'  [/home/tom/cu-project/slab.toml (project)]
  ...
unset keys use built-in defaults ('slab config init' shows them all)
```

Unknown keys are refused with the offending file named — a typo in a
cluster config surfaces at load, never silently configures nothing. Path
values expand `~` and `${VAR}`, and an **unset** variable is a loud error
(`os.path.expandvars` would quietly leave `$USRE` literal).

## What goes in it

```toml
schema_version = 1

[paths]
workspace = "/scratch/${USER}/slab-workspace"
pseudos = "/shared/sw/slab/pseudos"          # pseudopotential family root
engines = "/shared/sw/slab/engines.json"     # cluster engine registry

[engines.qe]
command = "srun pw.x"       # outranks ASE's own [espresso] config section;
                            # batch jobs only — on the login node use plain
                            # "pw.x" (srun outside an allocation queues or
                            # hangs, and slab refuses it there loudly)
pseudo_dir = "/shared/sw/pseudos"

[hpc]
cluster = "delta"
account = "abc-123"
default_partition = "cpu"
setup = ["module load quantum-espresso/7.4"]

[hpc.partitions.cpu]
time_limit = "24:00:00"
ntasks_per_node = 64
mem = "240G"
launcher = "srun"

[hpc.partitions.gpu]
time_limit = "12:00:00"
gres = "gpu:a100:4"
qos = "gpu"
setup = ["module load cuda/12.4"]     # runs after the [hpc] setup lines
sbatch_extra = ["--exclusive"]        # raw directives the schema does not model

[agent]
model = "meta-models/Muse-Glimmer-30B"
context_window = 131072

[agent.serve]                          # how to start that model on a GPU node
partition = "gpu"
time_limit = "08:00:00"
tool_call_parser = "llama4_pythonic"   # vLLM's, and model-specific
args = ["--tensor-parallel-size 4", "--max-model-len 131072"]
setup = [
  "source $SCRATCH/venvs/vllm/bin/activate",
  "export HF_HOME=$SCRATCH/hf-cache",  # pre-downloaded on the login node
  "export HF_HUB_OFFLINE=1",           # compute nodes are firewalled; serve from disk
]
```

The `[agent]` section configures the resident agent the same way — which model
it talks to, and (via `compute_profile`) how big a calculation it should reach
for on this machine. See
[Mason](mason.md#compute-budget-sizing-the-physics-to-the-machine).

Notice that `[agent]` has no `endpoint`. The agent's model server runs on
whichever GPU node the scheduler hands out, so its URL does not exist until the
job starts — `[agent.serve]` declares the launch and the job records the URL it
landed on. Everything is `[hpc.partitions]` reuse: a serve job is an ordinary
batch job that happens to run a server.

```bash
slab mason serve render          # the script, before you trust it
slab mason serve start --wait    # submit, then follow it to a live endpoint
slab mason serve status          # record + job state + a live probe
slab mason serve stop            # cancel and clear the record
```

`[agent.serve]` values are shell for the *compute node*, so no variable in them
is expanded at load — `setup = ["source $SCRATCH/venvs/vllm/bin/activate"]`
reaches the node verbatim. The full walkthrough is in
[Mason on the cluster](mason.md#on-the-cluster-end-to-end).

Configuration supplies *defaults that resolve into explicit values* — it
never reaches a cache key itself. The `qe` engine's cache identity records
the command and pseudo directory that actually ran, wherever they came
from; retuning a config file can never silently re-serve old physics under
a new meaning.

## SLURM without a workflow engine

SLAB's scheduler layer is deliberately thin: runs, caching, and
verification live in the workspace regardless of where the process
executes. The `[hpc]` section drives four verbs:

```bash
slab hpc partitions
slab hpc render "slab run relax.py" --name si-relax
slab hpc submit "slab run relax.py" --name si-relax --time 02:00:00
slab hpc status 4242314
```

`render` prints the exact sbatch script `submit` would use — read it
before trusting it. Only fields the config sets become `#SBATCH`
directives (Parsl's convention); SLAB adds no silent resource defaults:

<!-- no-verify -->
```text
#!/bin/bash -l
#SBATCH --job-name=si-relax
#SBATCH --partition=cpu
#SBATCH --output=si-relax-%j.out
#SBATCH --account=abc-123
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=64
#SBATCH --mem=240G

set -euo pipefail

module load quantum-espresso/7.4
# partition launcher omitted: 'slab' is a single-process driver;
# engines bring their own MPI (e.g. [engines.qe] command = "srun pw.x")
slab run relax.py
```

The usual payload is `slab run workflow.py`: the scheduler moves the
process, and the result is still a traced, check-gated run in the
workspace. Note what the launcher did *not* touch: a `slab` payload is
never prefixed with the partition's `launcher` — `srun` on the Python
driver would start one copy of the whole workflow per task, all writing
one run database, each deadlocking on its own nested engine `srun`. The
launcher belongs on the engine command inside the config, and slab
enforces that split in the rendered script. The submitted script is kept next to the job's outputs — a
job's exact script is provenance, not scratch.

Status polling asks `squeue` first and falls back to `sacct` (finished
jobs leave the queue), collapsing SLURM's ~22 states onto seven —
`pending`, `running`, `completed`, `failed`, `cancelled`, `timeout`,
`undetermined` — with the raw state string preserved as evidence. When
neither command answers, the state is reported as *undetermined* with the
reason; an unknown is never dressed up as a known.

## A cluster maintainer's checklist

1. Install SLAB into a shared environment.
2. Write `/sw/slab/config.toml` (paths, `[engines.qe]`, `[hpc]`
   partitions) and, optionally, `/sw/slab/engines.json` — starting from
   the files in `templates/` (see [Engines](engines.md) for the registry).
3. Pre-stage the two downloads compute nodes cannot make themselves
   (they are typically firewalled): install the pseudopotential families
   protocols will ask for (`slab pseudos install sssp` into a shared
   `paths.pseudos` root), and warm the MLIP checkpoint cache from a node
   with internet — `python -c "from mace.calculators import mace_mp;
   mace_mp(model='small')"` populates `~/.cache/mace` — or serve MLIPs
   through rootstock and skip local checkpoints entirely.
4. Export from the module file:
   `setenv SLAB_SITE_CONFIG /sw/slab/config.toml`.
5. Users check their view with `slab config show` and
   `slab engines list` — which now also reports the cluster's partitions.
6. If users will run the resident agent, add `[agent.serve]` (a GPU partition,
   the `tool_call_parser` your vLLM build registers, and `setup` lines that
   activate the vLLM venv and point `HF_HOME` at a model cache pre-downloaded
   on the login node, with `HF_HUB_OFFLINE=1`) so nobody has to rediscover the
   serving recipe. `slab mason doctor` is how they confirm the parser name was
   right.

Users then override per project in `slab.toml`, and nothing about a
cluster is baked into anyone's Python.
