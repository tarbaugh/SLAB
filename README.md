# SLAB — Simplest Layer for Atomistic Backends

[![tests](https://github.com/tarbaugh/SLAB/actions/workflows/tests.yml/badge.svg)](https://github.com/tarbaugh/SLAB/actions/workflows/tests.yml)

An agent-native state layer for atomistic materials modeling. Every run
starts as temporary and becomes permanent only when you promote it.
Workflows are plain Python, and SLAB traces the task graph as the script
runs, verifies the results with checks, and expires what nobody promoted.

**Documentation: [tarbaugh.github.io/SLAB](https://tarbaugh.github.io/SLAB/)**
has the overview, the tutorials (every code block executed against the real
API), and the architecture document.

## Install

```bash
pip install "slab-stack @ git+https://github.com/tarbaugh/SLAB"
```

Optional extras: `[rootstock]` adds the thin client for cluster-served
MLIPs (no torch), and `[mcp]` adds the MCP server for agents. From a
checkout, `pip install -e ".[dev,mcp]"`. Python ≥ 3.11. There is no
daemon, no database server, and no required configuration. A workspace is
a directory (`.slab/` by default) that holds a SQLite file and a
content-addressed store, and one command, `slab`, drives everything.

## What it can do

- **Engines.** EMT and Lennard-Jones built in, Quantum ESPRESSO and LAMMPS
  as built-ins that drive the real executables, rootstock-served MLIP
  checkpoint ids usable directly as engine names, and a cluster engine
  registry for everything else.
  [Engines](https://tarbaugh.github.io/SLAB/tutorials/engines/)
- **Protocols.** AiiDA's named Quantum ESPRESSO input protocols and SSSP
  pseudopotential families, applied by name and traced by value.
  [Protocols & pseudopotentials](https://tarbaugh.github.io/SLAB/tutorials/protocols-and-pseudos/)
- **Builders.** Structures from atomsk, structures and metadata from an
  offline Materials Project snapshot, and machine-learned potentials
  trained with gracemaker, each as a traced task.
  [Engines § Builders](https://tarbaugh.github.io/SLAB/tutorials/engines/#builders-atomsk)
- **Runs, checks, retention, caching.** Runs earn `verified` from
  machine-checkable hooks, retention is tiered by artifact role, and a
  rerun of an unchanged script is a cache hit.
  [Lifecycle](https://tarbaugh.github.io/SLAB/tutorials/lifecycle-and-retention/),
  [Verification](https://tarbaugh.github.io/SLAB/tutorials/verification/),
  [Caching & resume](https://tarbaugh.github.io/SLAB/tutorials/caching-and-resume/)
- **Failure is evidence.** A failed run carries a structured failure record,
  the engine's own error lines, and the scratch files that explain it.
  [Debugging failures](https://tarbaugh.github.io/SLAB/tutorials/debugging-failures/)
- **HPC.** One layered TOML file per cluster declares paths, engines, and
  SLURM partitions, and `slab hpc` renders, submits, and polls jobs.
  [HPC configuration & SLURM](https://tarbaugh.github.io/SLAB/tutorials/hpc-config/)
- **Agents.** `slab mcp` serves a workspace to any MCP client as tools.
  [Agents over MCP](https://tarbaugh.github.io/SLAB/tutorials/agents-mcp/)
- **Mason.** The resident research agent for long campaigns, with a roster
  of specialists, Agent Skills, machine memory, and its model served as a
  batch job.
  [Mason](https://tarbaugh.github.io/SLAB/tutorials/mason/),
  [The roster & skills](https://tarbaugh.github.io/SLAB/tutorials/roster-and-skills/),
  [Machine memory](https://tarbaugh.github.io/SLAB/tutorials/memory/)

The design argument, and why existing workflow engines do not fit agentic
use, is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Demo

The demo relaxes five perturbed variants of a Cu supercell under EMT, each
in its own run with a stated intent. It needs no extras and runs in
seconds:

```bash
python examples/demo.py
```

```text
workspace: .slab   engine: emt   system: Cu x 32 atoms
  run 01m1ff4dzf  E = -0.213146 eV  fmax = 0.0407  steps =  6  -> verified
  run 01m1ff4e0h  E = -0.213732 eV  fmax = 0.0439  steps = 10  -> verified
  run 01m1ff4e1m  E = -0.214059 eV  fmax = 0.0315  steps = 13  -> verified
  run 01m1ff4e2z  E = -0.213770 eV  fmax = 0.0419  steps = 13  -> verified
  run 01m1ff4e47  E = -0.214043 eV  fmax = 0.0301  steps = 14  -> verified

lowest energy: run 01m1ff4e1m  (E = -0.214059 eV)
decide what deserves permanence, then clean up:
  slab promote 01m1ff4e1m --reason 'lowest energy of 5 variants'
  slab expire --older-than 0d
  slab gc
```

All five runs earned `verified` from their checks, and none is permanent.
That decision happens now, after the results exist:

```text
$ slab promote 01m1ff4e1m --reason 'lowest energy of 5 variants'
promoted 01m1ff4e1mhs7c7318nthym00x  cu-relax-2

$ slab expire --older-than 0d
expired 01m1ff4e47ymqgm0qyqh25sgwm  cu-relax-4
expired 01m1ff4e2zs3pd4ncc47kwsq9v  cu-relax-3
expired 01m1ff4e0hw5zx8hhkw2ewb5ar  cu-relax-1
expired 01m1ff4dzfa01fg4ad5bgmcx64  cu-relax-0
4 run(s) expired

$ slab gc
dropped 31 blob(s), freeing 186450 bytes; 8 kept
```

The expired runs stay queryable as hash-and-recipe skeletons. The promoted
run keeps full bytes for its terminal artifacts, while its intermediate
trajectory is hash-only, so the archive holds exactly what someone decided
to keep:

```text
$ slab show 01m1ff4e1m
run 01m1ff4e1mhs7c7318nthym00x  cu-relax-2
  state:   promoted    status: completed
  created: 2026-09-01T21:48:47.540198+00:00
  intent:  variant 2: rattle stdev=0.06 A - hunting the lowest-energy relaxation of the batch
  checks:  3/3 passed
    [+] forces_converged: fmax=0.0315097 < 0.05
    [+] energy_is_finite: energy=-0.214059 is finite
    [+] energy_unit_declared: unit 'eV' == expected 'eV'
  tasks:
    1. relax  completed  0.039s
  artifacts:
    variant-2.traj  intermediate  33088B  hash-only  f6e2501985cc
    relaxed.xyz  terminal  3493B  bytes  8a19a3b1f3e5
    result  terminal  101B  bytes  a9535ebc422a
  history:
    quarantined -> verified  by checks: 3/3 assertions passed
    verified -> promoted  by user: lowest energy of 5 variants
```

The [quickstart](https://tarbaugh.github.io/SLAB/tutorials/quickstart/)
walks the same loop from Python and from the CLI.

## A workflow is a script

<!-- no-verify -->
```python
from foundation import Workspace, check, converged
from foundation.tasks import relax

ws = Workspace(".slab")
with ws.start_run(name="si-relax", intent="baseline lattice constant") as run:
    relaxed, info = relax(atoms, engine="emt", fmax=0.05)    # traced, cached, recorded

    @check
    def forces_converged():
        return converged(info["fmax"], below=0.05, label="fmax")

    run.keep("relaxed.xyz", relaxed)                          # declared terminal artifact
# exit: checks evaluated -> quarantined becomes verified; promote it when you decide it matters
```

Swap `engine="emt"` for `"qe"`, `"lammps"`, or a served MLIP checkpoint id
such as `"mace-mp-0-medium"`, and the engine's version and options enter
the cache key and the recipe with it.

## Benchmark

Five copper questions with known answers, run as autonomous campaigns per
model and scored on one criterion: did the agent compute a correct
answer, backed by verified runs?

<!-- benchmark:summary:start -->
| Model | Machine | Passed |
| --- | --- | --- |
| llama3.1:8b | laptop | 0/5 |
| llama3.1:8b-32k | laptop | 0/5 |

Five copper questions with known answers; [the benchmark](https://tarbaugh.github.io/SLAB/benchmark/) has the rule.
<!-- benchmark:summary:end -->

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev,mcp]"
.venv/bin/pytest          # tests + doctests + coverage
.venv/bin/mypy            # strict on src/
.venv/bin/ruff check .
```

License: MIT.
