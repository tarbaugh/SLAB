# SLAB — Simplest Layer for Atomistic Backends

Agent-native workflow orchestration for atomistic materials modeling.

**Runs are born ephemeral and promoted to permanent — never born permanent and
deleted.** Workflows are plain imperative Python; the graph is traced, never
declared. Machine-checkable verification hooks gate what "verified" means, and
an explicit one-command promotion is the *only* thing that makes data
permanent. Everything else silently expires.

```python
from slab import Workspace, task, check, converged
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

## Why

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
[ARCHITECTURE.md](ARCHITECTURE.md).

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
keep full bytes for **terminal** artifacts and **input** roots (so recompute-
on-demand is a real promise), while **intermediate** bytes are
hash-and-discarded — the content hash and the complete recipe (inputs, code
version, engine versions, parameters) survive on the run forever.

## Install

```bash
pip install -e .              # core: pydantic + typer + ase
pip install -e ".[mace]"      # + MACE foundation model (torch)
pip install -e ".[mcp]"       # + MCP server for agents
pip install -e ".[dev]"       # tests, lint, types
```

Python ≥ 3.11. No daemon, no database server, no configuration: a workspace
is a directory (`.slab/` by default) holding a SQLite file and a
content-addressed store.

## The demo

Relax 5 perturbed variants of a Si supercell with MACE, promote the best,
expire the rest, and garbage-collect:

```bash
python examples/demo.py                # MACE + Si (downloads the model on first use)
python examples/demo.py --engine emt   # EMT + Cu: no extras, runs in seconds
```

```text
workspace: .slab   engine: mace   system: Si x 64 atoms
  run 01kzs0en5y  E = -85.905208 eV  fmax = 0.0450  steps =  4  -> verified
  run 01kzs0er2e  E = -85.906516 eV  fmax = 0.0497  steps = 11  -> verified
  run 01kzs0erjh  E = -85.905218 eV  fmax = 0.0497  steps = 14  -> verified
  run 01kzs0es4w  E = -85.906204 eV  fmax = 0.0458  steps = 15  -> verified
  run 01kzs0esre  E = -85.904782 eV  fmax = 0.0461  steps = 16  -> verified

lowest energy: run 01kzs0er2e  (E = -85.906516 eV)
```

Each variant is its own run with a stated intent; the convergence checks
passed, so all five landed `verified`. Nothing is permanent yet — that
decision happens now, *after* the results exist:

```bash
$ slab promote 01kzs0er2e --reason "lowest energy of 5 variants"
promoted 01kzs0er2ebvvwxhrfmqybq2fp  si-relax-1

$ slab expire --older-than 0d
4 run(s) expired

$ slab gc
dropped 31 blob(s), freeing 121331 bytes; 8 kept
```

The archive now contains exactly one run's terminal artifacts (plus its
recompute roots); everything else is a hash-and-recipe skeleton that remains
fully queryable:

```text
$ slab show 01kzs0er2e                      $ slab show 01kzs0en5y   # expired
  state:   promoted                           state:   expired
  checks:  3/3 passed                         checks:  3/3 passed
  artifacts:                                  artifacts:
    variant-1.traj  ...  hash-only              variant-0.traj  ...  hash-only
    relaxed.xyz     ...  bytes                  relaxed.xyz     ...  hash-only
    result          ...  bytes                  result          ...  hash-only
```

## CLI

| Verb | What it does |
| --- | --- |
| `slab run script.py` | Execute a zero-ceremony workflow script inside a traced run (lands in quarantine). Scripts that manage their own runs are executed with plain `python`. |
| `slab list [--state S] [--status S] [-q]` | List runs, newest first. |
| `slab show <id> [--json]` | One run: state, intent, checks, tasks, artifacts, history. Ids accept unique prefixes, git-style. |
| `slab promote <id> [--reason ...] [--force]` | Make a run permanent. `--force` promotes an unverified run and is recorded as forced. |
| `slab expire [--older-than 30d]` | Expire unpromoted runs past their TTL (state change only). `0d` = everything unpromoted, now. |
| `slab gc [--dry-run]` | Drop artifact bytes no retention rule demands. |
| `slab mcp` | Serve the workspace to agents over MCP (stdio). |

Workspace resolution: `-w/--workspace` flag > `$SLAB_WORKSPACE` > `./.slab`.
Retention policy: `--policy file.json` > `<workspace>/policy.json` > defaults.

## Retention policy as data

```json
{
  "quarantined": {"ttl_days": 30},
  "verified":    {"ttl_days": 90},
  "promoted":    {"keep": ["terminal", "input"]}
}
```

TTLs attach to lifecycle states, anchored to when the run *entered* its
current state. Roles not listed in `keep` are hash-only. A TTL on `promoted`
or `archived` is rejected at validation — the asymmetry is enforced, not
conventional.

## Agent surface (MCP)

```json
{"mcpServers": {"slab": {"command": "slab", "args": ["mcp"]}}}
```

Tools: `launch_workflow`, `list_runs`, `show_run`, `promote_run`,
`expire_runs`, `gc` — the same code paths as the CLI, returning structured
JSON. Script output is captured into the result so prints can't corrupt the
protocol channel.

## Non-goals

- **No physics engines** and no new file-format parsers beyond what ASE
  provides. Backends are reached through the ASE `Calculator` contract;
  LAMMPS/VASP/GROMACS/QE/MLIPs are BLAS — SLAB is NumPy one layer up.
- **No HPC scheduler integration** in the MVP. Execution is local; the
  runtime is designed so a SLURM executor can be added behind the same
  tracing surface.
- **No web UI.**
- **No distributed daemon.** A workspace is a directory; concurrency is
  handled at the SQLite transaction level.

## Status

MVP vertical slice, working end to end: lifecycle state machine,
content-addressed artifact store with tiered retention, define-by-run tracing
with content-hash caching, verification hooks, MACE/ASE relaxation task, CLI,
MCP server. 430+ tests, ~100% coverage on the load-bearing core, mypy
`--strict`. The `RunStore` protocol is the seam for Postgres; the backend
factory is the seam for more engines.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev,mcp]"
.venv/bin/pytest          # tests + doctests + coverage
.venv/bin/mypy            # strict on src/
.venv/bin/ruff check src tests
```

Docstring examples are executed as doctests on every test run — the API docs
cannot silently rot.

License: MIT.
