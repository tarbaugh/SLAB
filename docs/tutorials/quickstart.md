# Quickstart

This page shows the whole SLAB loop in five minutes. You relax a structure
inside a traced run, verify it out of quarantine, promote it to permanent,
and clean up, all in plain Python with no daemon and no configuration.

## Install

SLAB needs Python ≥ 3.11. The core install brings `pydantic`, `typer`, and
`ase`, which is enough for everything on this page because the EMT engine
ships with ASE:

<!-- no-verify -->
```bash
pip install -e .
```

You configure nothing. A workspace is a directory that holds a SQLite file
and a content-addressed store, and SLAB creates it on first use.

## Run, check, keep

Build a rattled Cu supercell, open a workspace, and relax it inside a run.
Three things happen in the `with` block:

- The `relax` call is traced, so SLAB hashes the inputs, caches the result,
  and records the trajectory.
- The `@check` registers a verification hook.
- `run.keep` declares which value is a terminal artifact, the thing that
  promotion will later preserve.

```python
from ase.build import bulk

from foundation import Workspace, check, converged
from foundation.tasks import relax

atoms = bulk("Cu", "fcc", a=3.58, cubic=True) * (2, 2, 2)   # 32 atoms
atoms.rattle(stdev=0.05, seed=42)

ws = Workspace("qs-workspace")
with ws.start_run(name="cu-relax", intent="quickstart: relax a rattled Cu supercell") as run:
    relaxed, info = relax(atoms, engine="emt", fmax=0.05)

    @check
    def forces_converged():
        return converged(info["fmax"], below=0.05, label="fmax")

    run.keep("relaxed", relaxed)

print(f"E = {info['energy']:.6f} eV  fmax = {info['fmax']:.4f}  steps = {info['steps']}")

final = ws.runs.get(run.id)
print(f"state = {final.state.value}  status = {final.status.value}")
```

```text
E = -0.213682 eV  fmax = 0.0435  steps = 11
state = verified  status = completed
```

The numbers reproduce exactly, because the rattle seed is fixed and EMT is
deterministic. The `intent` is narrative provenance that states why this run
exists, and SLAB stores it on the run and shows it in every listing.

## Verification is earned

The run was born `quarantined`, like every run. When the `with` block exits,
the runtime evaluates each registered check and stores its result, including
the observed and expected values (`fmax=0.0434817 < 0.05`). The run moves
`quarantined -> verified` only because every assertion passed.

A run with zero checks stays quarantined forever, because verification is
earned, never defaulted. If the block raises, SLAB marks the run `failed`
with a structured failure record, and the run ages out.

!!! note
    Rerunning this script is also the resume mechanism. `relax` is content-hash
    cached on its inputs (structure, engine, parameters), so a second execution
    replays the cached result in milliseconds and continues from there.

## Promote what matters

Nothing so far is permanent. `verified` means the machine-checkable claims
held, but whether the result matters is your decision, and you make it now,
after you see the result. Promotion is the only action that makes data
permanent, and it records a reason:

```python
promoted = ws.runs.transition(run.id, "promoted", reason="quickstart result worth keeping")
print(f"state = {promoted.state.value}")
```

```text
state = promoted
```

Promoted runs cannot expire. The transition does not exist in the lifecycle,
and a retention policy that puts a TTL on `promoted` fails validation.

## Housekeeping

Everything you never promote expires automatically. Cleanup has two phases.
First, `expire_due` moves overdue unpromoted runs to `expired`, which is a
state change only. Then `gc` drops every artifact byte that no retention
rule demands:

```python
expired = ws.expire_due()
report = ws.gc()
print(f"expired {len(expired)} run(s); dropped {len(report.dropped)} blob(s), kept {len(report.kept)}")
ws.close()
```

```text
expired 0 run(s); dropped 2 blob(s), kept 6
```

Nothing expired, because the only run is promoted and the default TTLs (30
days quarantined, 90 days verified) are not due. But `gc` still reclaimed
bytes. Retention is tiered by artifact role, and even a promoted run keeps
full bytes only for **terminal** artifacts and **input** roots. The
intermediate BFGS trajectory (`relax.traj`) was hash-and-discarded, but its
content hash and complete recipe stay on the run forever, so it remains
queryable and recomputable. The full model is in
[Lifecycle and retention](lifecycle-and-retention.md).

## The same loop from the CLI

Every verb goes through the same operations layer as the Python API. A
zero-ceremony script, one with no `start_run` of its own, launches with
`slab run` and lands in quarantine. The rest of the loop is one
command each:

<!-- no-verify -->
```bash
slab run relax_cu.py --intent "baseline"    # traced run, lands quarantined
slab list                                   # runs, newest first
slab show 01kzsm                            # ids accept unique prefixes, git-style
slab promote 01kzsm --reason "worth keeping"
slab expire --older-than 0d                 # everything unpromoted, now
slab gc
```

`slab show` prints the state, the intent, the checks with their compared
values, the tasks, the artifacts (with `bytes` or `hash-only` presence), and
the full transition history. Agents get the identical surface as MCP tools,
as described in [Agents over MCP](agents-mcp.md). To swap EMT for a
cluster-served code or a rootstock MLIP, see [Engines](engines.md).
