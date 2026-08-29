# SLAB

**Simplest Layer for Atomistic Backends** is an agent-native state layer
for atomistic materials modeling. It keeps the record of what an agent
computed, verifies the results, and retains only what you promote.

Every run starts as temporary, and it becomes permanent only when you
promote it. Nothing is stored permanently by default and deleted later.
Workflows are plain imperative Python, and SLAB traces the task graph as the
script runs. Machine-checkable verification hooks decide when a run counts
as verified, and an explicit promotion command is the only action that makes
data permanent. Everything else expires automatically.

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

## The problem SLAB owns

An agent does not run one calculation. It runs hundreds while it explores,
and most of them are drafts. Skill packs and agent harnesses make those
calculations easy to start, but they leave the results as loose files in a
repository. After a week of agentic work, nobody can say which numbers are
verified, which files are safe to delete, or how the one result that
matters was made.

SLAB sits under that work. Every calculation lands as a run with a recorded
recipe, a cache identity, and a lifecycle state. Verification is a property
a run earns from its checks, not a claim in a notebook. Unpromoted data
expires on its own, so the workspace stays small without manual cleanup.
The archive holds only what someone decided to keep, and every kept run
carries the complete recipe that reproduces it.

## Use it from any harness

SLAB does not require its own agent.

- **Python.** Workflows are ordinary scripts. Any harness that can run
  Python can hold a workspace.
- **MCP.** `foundation mcp` serves a workspace as a set of MCP tools, so
  Claude Code, Cursor, or any MCP client can start, inspect, and promote
  runs. See [Agents over MCP](tutorials/agents-mcp.md).
- **Mason.** The distribution includes a resident research agent for long
  campaigns on a cluster, with its model server started as a batch job. It
  is optional. See [Mason, the resident agent](tutorials/mason.md).

Mason's skills follow the Agent Skills specification exactly, and skills
written for other consumers load unmodified from a project or user skills
directory. So you can bring an external skill pack and keep the SLAB
lifecycle under it. See [The roster & skills](tutorials/roster-and-skills.md).

## Three packages

SLAB is three packages in one distribution, `slab-stack`. Each installs a
command of the same name.

- **`slab`** gives access to computational software: engines and
  calculators, the cluster engine registry, Quantum ESPRESSO protocols,
  pseudopotential families, and the SLURM layer.
- **`foundation`** keeps state and runs workflows: runs, artifacts, caching,
  verification, retention, and the MCP server.
- **`mason`** is the resident research agent.

`mason` depends on `foundation` and `slab`, `foundation` depends on `slab`,
and `slab` depends on neither. So you can drive an engine without a
workspace, and keep a workspace without an agent.

## Why not an existing workflow engine?

Existing materials workflow engines (AiiDA, atomate2/jobflow, pyiron,
FireWorks) were designed for human experts. Three of their costs are fatal
for agentic use:

1. **Provenance totality.** Every intermediate is stored forever as an
   immutable graph node, including retrieved wavefunctions, trajectories,
   and charge densities that nobody ever reads. Deletion feels like surgery
   instead of housekeeping.
2. **Declaration-time epistemics.** You must decide whether a run is
   "production" at submission time, but that information only exists at
   completion time, after you see convergence and output sanity. So debug
   work runs in the production profile, and the archive fills with failed
   attempts.
3. **Ceremony-heavy APIs.** WorkChain-style declarative process classes are
   hard for trained humans to read, and they are token-expensive for LLM
   agents to generate and re-read while debugging.

SLAB inverts all three. The full argument is in
[Architecture](architecture.md).

## The lifecycle

```
quarantined ──checks pass──▶ verified ──promote──▶ promoted ──archive──▶ archived
    │  │                        │
    │  └────────force-promote───┘ (recorded)
    │ ttl                       │ ttl
    ▼                           ▼
 expired ◀──────────────────────┘
```

Two rules are structural, not policy. Promoted data cannot expire, because
the transition does not exist and a retention policy with a TTL on
`promoted` fails validation. Unpromoted data expires automatically.

Retention is tiered by artifact role, not by data type. Promoted runs keep
full bytes for **terminal** artifacts and **input** roots, while
**intermediate** bytes are hash-and-discarded. Their content hash and the
complete recipe stay on the run forever.

## Failure is evidence, not a status

SLAB's user is an LLM agent, and such an agent can devise a niche correction
if it can see what actually happened. It might shrink the perturbation,
switch the engine, or loosen a threshold. So failed runs and tasks carry
structured failure records that hold the exception, a trimmed traceback, and
diagnostic notes. The scratch data that explains a failure survives it, so a
crash in the middle of an optimization keeps its partial trajectory as an
artifact. Checks record the `observed` and `expected` values they compared.
SLAB delivers evidence rather than running an error protocol. See the
[debugging tutorial](tutorials/debugging-failures.md).

## Install

```bash
pip install -e .              # core: pydantic + typer + ase
pip install -e ".[rootstock]" # + cluster-served MLIPs (thin client, no torch)
pip install -e ".[mcp]"       # + MCP server for agents
```

One install brings all three packages and all three commands.

Python ≥ 3.11. There is no daemon, no database server, and no required
configuration. A workspace is a directory (`.slab/` by default) that holds a
SQLite file and a content-addressed store, and a cluster describes itself in
one optional layered TOML file. See
[HPC configuration](tutorials/hpc-config.md).

## Where to go next

- **[Quickstart](tutorials/quickstart.md)** covers the full loop (run,
  verify, promote, expire, gc) in five minutes, with no heavy dependencies.
- **[Lifecycle & retention](tutorials/lifecycle-and-retention.md)** explains
  the states, the TTLs, and retention policy as data.
- **[Verification checks](tutorials/verification.md)** shows how a run earns
  `verified`.
- **[Caching & resume](tutorials/caching-and-resume.md)** shows why
  rerunning a script is the resume mechanism.
- **[Engines](tutorials/engines.md)** covers the built-in engines, cluster
  registries, and rootstock checkpoints served by name.
- **[HPC configuration & SLURM](tutorials/hpc-config.md)** describes the one
  layered TOML file per cluster, with paths, engines, partitions, and batch
  submission.
- **[Debugging failures](tutorials/debugging-failures.md)** walks the
  failure evidence surfaces, which are tuned for LLM consumers.
- **[Agents over MCP](tutorials/agents-mcp.md)** serves a workspace to an
  agent as a set of MCP tools.
- **[Mason, the resident agent](tutorials/mason.md)** introduces the
  built-in Claude-Code-class harness for open models, tuned for long
  research projects, with its model server started as a batch job.
- **[The roster & skills](tutorials/roster-and-skills.md)** covers the agent
  cards and the Agent Skills format, including skills from external packs.
- **[Machine memory](tutorials/memory.md)** keeps what one session learns
  for the next.

## Status

MVP vertical slice, working end to end. It includes:

- the lifecycle state machine;
- a content-addressed artifact store with tiered retention;
- define-by-run tracing with content-hash caching;
- verification hooks;
- relaxation and single-point tasks for ASE, Quantum ESPRESSO, LAMMPS, and
  MLIPs served through rootstock;
- AiiDA-style input protocols and SSSP pseudopotential families;
- layered HPC configuration with a SLURM submission layer;
- the Mason agent harness, for open models self-served on a GPU node or for
  Claude, with its model server as a batch job;
- three commands (`slab`, `foundation`, `mason`) and an MCP server.

Quality gates: 1000+ tests (every docstring example runs as a doctest), ~95%
coverage, mypy `--strict`, and adversarial multi-agent review passes whose
confirmed findings became regression tests. A test reads the AST of every
module and fails on an import that crosses the package layering the wrong
way.

Key paths are verified against real software, not only mocks:

- the QE engine, against a real `pw.x` 7.4.1;
- the two-fidelity chain (a rootstock-served MLIP relax, then a QE single
  point on the relaxed geometry), against a real `pw.x` 7.5;
- the LAMMPS engine, against a real `lmp` (22 Jul 2025);
- the balanced protocol, against a real SSSP install;
- Mason, against a real Llama 3.1 served by Ollama, in an autonomous relax
  whose reported energy matches an independent calculation exactly.

MIT licensed.
