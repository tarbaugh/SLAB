# SLAB

**Simplest Layer for Atomistic Backends** — agent-native workflow
orchestration for atomistic materials modeling.

Runs are born ephemeral and promoted to permanent — never born permanent and
deleted. Workflows are plain imperative Python; the graph is traced, never
declared. Machine-checkable verification hooks gate what "verified" means, and
an explicit one-command promotion is the *only* thing that makes data
permanent. Everything else silently expires.

```python
from slab import Workspace, check, converged
from slab.tasks import relax

ws = Workspace(".slab")
with ws.start_run(name="si-relax", intent="baseline lattice constant") as run:
    relaxed, info = relax(atoms, engine="mace", fmax=0.05)   # traced, cached, recorded

    @check
    def forces_converged():
        return converged(info["fmax"], below=0.05, label="fmax")

    run.keep("relaxed.xyz", relaxed)                          # declared terminal artifact
# exit: checks evaluated -> quarantined becomes verified; promote it when you decide it matters
```

## Why another workflow engine?

Existing materials workflow engines (AiiDA, atomate2/jobflow, pyiron,
FireWorks) were designed for human experts and impose costs that are fatal for
agentic use:

1. **Provenance totality.** Every intermediate is stored forever as an
   immutable graph node — retrieved wavefunctions, trajectories, and charge
   densities nobody ever reads. Deletion feels like surgery instead of
   housekeeping.
2. **Declaration-time epistemics.** You must decide whether a run is
   "production" at *submission* time, but that information only exists at
   *completion* time — after seeing convergence and output sanity. So debug
   work runs in the production profile and the archive fills with the
   archaeology of failures.
3. **Ceremony-heavy APIs.** WorkChain-style declarative process classes are
   hard for trained humans to read and token-expensive for LLM agents to
   generate and re-read while debugging.

SLAB inverts all three. The full argument lives in
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

Two rules are structural, not policy: nothing makes promoted data expire (the
transition does not exist, and a retention policy carrying a TTL for
`promoted` fails validation), and expiry of unpromoted data is automatic and
silent. Retention is tiered by *artifact role*, not data type: promoted runs
keep full bytes for **terminal** artifacts and **input** roots, while
**intermediate** bytes are hash-and-discarded — the content hash and the
complete recipe survive on the run forever.

## Failure is evidence, not a status

SLAB's user is an LLM agent that can devise a *niche* correction — shrink the
perturbation, switch the engine, loosen a threshold — **if** it can see what
actually happened. Failed runs and tasks carry structured failure records
(exception, trimmed traceback, diagnostic notes); the scratch data that
explains a failure survives it (a mid-optimization crash keeps its partial
trajectory as an artifact); checks report the `observed`/`expected` values
they compared. Evidence delivery, not error protocol — see the
[debugging tutorial](tutorials/debugging-failures.md).

## Install

```bash
pip install -e .              # core: pydantic + typer + ase
pip install -e ".[mace]"      # + MACE foundation model in-process (torch)
pip install -e ".[rootstock]" # + cluster-served MLIPs (thin client, no torch)
pip install -e ".[mcp]"       # + MCP server for agents
```

Python ≥ 3.11. No daemon, no database server, no configuration: a workspace
is a directory (`.slab/` by default) holding a SQLite file and a
content-addressed store.

## Where to go next

- **[Quickstart](tutorials/quickstart.md)** — the full loop (run → verify →
  promote → expire → gc) in five minutes, no heavy dependencies.
- **[Lifecycle & retention](tutorials/lifecycle-and-retention.md)** — states,
  TTLs, and retention policy as data.
- **[Verification checks](tutorials/verification.md)** — how a run earns
  `verified`.
- **[Caching & resume](tutorials/caching-and-resume.md)** — rerunning a script
  *is* the resume mechanism.
- **[Engines](tutorials/engines.md)** — MACE in-process, cluster registries,
  and rootstock checkpoints served silently.
- **[Debugging failures](tutorials/debugging-failures.md)** — the failure
  evidence surfaces, tuned for LLM consumers.
- **[Agents over MCP](tutorials/agents-mcp.md)** — serve a workspace to an
  agent as a set of MCP tools.

## Status

MVP vertical slice, working end to end: lifecycle state machine,
content-addressed artifact store with tiered retention, define-by-run tracing
with content-hash caching, verification hooks, MACE/ASE/Quantum ESPRESSO
relaxation task, CLI, MCP server. 580 tests (every docstring example runs as
a doctest), ~100% coverage on the load-bearing core, mypy `--strict`, plus
adversarial multi-agent review passes whose confirmed findings are regression
tests — and the QE engine is verified against a real `pw.x` 7.4.1.

MIT licensed.
