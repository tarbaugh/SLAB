# Quickstart

The whole SLAB loop in five minutes: relax a structure inside a traced run, watch
verification gate it out of quarantine, promote it to permanent, and clean up.
Plain Python, no daemon, no configuration.

## Install

SLAB needs Python ≥ 3.11. The core install brings `pydantic`, `typer`, and `ase` —
enough for everything on this page, because the EMT engine ships with ASE:

<!-- no-verify -->
```bash
pip install -e .
```

There is nothing to configure. A workspace is a directory holding a SQLite file
and a content-addressed store, created on first use.

## Run, check, keep

Build a rattled Cu supercell, open a workspace, and do a relaxation inside a run.
Three things happen in the `with` block: the `relax` call is traced (inputs hashed,
results cached, the trajectory recorded), the `@check` registers a verification
hook, and `run.keep` declares which value is a *terminal* artifact — the thing
promotion will later preserve.

```python
from ase.build import bulk

from slab import Workspace, check, converged
from slab.tasks import relax

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

The numbers are reproducible — the rattle seed is fixed and EMT is deterministic.
The `intent` is narrative provenance: *why* this run exists, stored on the run and
shown in every listing.

## Verification is earned

The run was born `quarantined` — every run is. On exit from the `with` block the
runtime evaluated the registered check, stored its result (including the observed
and expected values: `fmax=0.0434817 < 0.05`), and only because every assertion
passed did the run move `quarantined -> verified`. A run with zero checks stays
quarantined forever: verification is earned, never defaulted. If the block had
raised, the run would be marked `failed` with a structured failure record and
simply age out.

!!! note
    Rerunning this script is also the resume mechanism. `relax` is content-hash
    cached on its inputs (structure, engine, parameters), so a second execution
    replays the cached result in milliseconds and gets on with whatever comes next.

## Promote what matters

Nothing so far is permanent. `verified` means "the machine-checkable claims held" —
whether the result *matters* is your decision, made now, after seeing it. Promotion
is the only thing that makes data permanent, and it wants a reason:

```python
promoted = ws.runs.transition(run.id, "promoted", reason="quickstart result worth keeping")
print(f"state = {promoted.state.value}")
```

```text
state = promoted
```

Promoted runs cannot expire — the transition does not exist in the lifecycle, and
a retention policy that tries to attach a TTL to `promoted` fails validation.

## Housekeeping

Everything you never promote silently expires. Cleanup is two-phase: `expire_due`
flips overdue unpromoted runs to `expired` (a state change only), then `gc` drops
every artifact byte no retention rule demands:

```python
expired = ws.expire_due()
report = ws.gc()
print(f"expired {len(expired)} run(s); dropped {len(report.dropped)} blob(s), kept {len(report.kept)}")
ws.close()
```

```text
expired 0 run(s); dropped 2 blob(s), kept 6
```

Nothing expired — our only run is promoted, and the default TTLs (30 days
quarantined, 90 verified) are nowhere near due. But `gc` still reclaimed bytes:
retention is tiered by artifact *role*, and even a promoted run keeps full bytes
only for **terminal** artifacts and **input** roots. The intermediate BFGS
trajectory (`relax.traj`) was hash-and-discarded — its content hash and complete
recipe stay on the run forever, so it remains queryable and recomputable. The
full model is in [Lifecycle and retention](lifecycle-and-retention.md).

## The same loop from the CLI

Every verb goes through the same operations layer as the Python API. Zero-ceremony
scripts (no `start_run` of their own) launch with `slab run` and land in
quarantine; the rest of the loop is one command each:

<!-- no-verify -->
```bash
slab run relax_cu.py --intent "baseline"    # traced run, lands quarantined
slab list                                   # runs, newest first
slab show 01kzsm                            # ids accept unique prefixes, git-style
slab promote 01kzsm --reason "worth keeping"
slab expire --older-than 0d                 # everything unpromoted, now
slab gc
```

`slab show` prints state, intent, checks with their compared values, tasks,
artifacts (with `bytes` vs `hash-only` presence), and the full transition history.
Agents get the identical surface as MCP tools — see [Agents and MCP](agents-mcp.md),
and [Engines](engines.md) for swapping EMT for MACE or cluster-served codes.
