# Agents over MCP

LLM agents are SLAB's primary user, so the workspace speaks their native protocol: `slab mcp` serves the same operations as the CLI — one shared code path in `slab._ops`, two skins — as MCP tools over stdio, returning structured JSON instead of formatted text.

## Setup

The server ships as an extra and any MCP client launches it as a subprocess:

<!-- no-verify -->
```bash
pip install 'slab[mcp]'
```

<!-- no-verify -->
```json
{"mcpServers": {"slab": {"command": "slab", "args": ["mcp"]}}}
```

The workspace is resolved exactly as for the CLI: `-w/--workspace` flag > `$SLAB_WORKSPACE` > `./.slab` (so `{"args": ["mcp", "-w", "/scratch/proj/.slab"]}` pins a specific one). No daemon, no database server — the workspace is a directory, and concurrent CLI and MCP access coexist at the SQLite transaction level.

!!! note
    Under stdio MCP, stdout *is* the protocol channel. `launch_workflow` therefore redirects everything a workflow script prints (checks included, which evaluate at run exit) into the result's `output` field. Scripts can `print()` freely; nothing corrupts the wire.

## The toolbox

Seven tools, each a thin wrapper over the operations layer:

| Tool | What it does |
| --- | --- |
| `launch_workflow` | Execute a plain-Python workflow script in a fresh traced run. |
| `list_runs` | Runs newest first, filterable by lifecycle `state` and execution `status`. |
| `show_run` | Everything about one run: checks, tasks, artifacts, history, failure evidence. |
| `promote_run` | Make a run permanent (`verified -> promoted`), with a recorded reason. |
| `expire_runs` | Expire unpromoted runs past their TTL; `older_than="0d"` means everything, now. |
| `gc` | Drop artifact bytes no retention rule demands (`dry_run=True` only reports). |
| `list_engines` | Built-in engines, the cluster registry's declarations, rootstock checkpoint ids, QE protocols, installed pseudo families. |

**`launch_workflow(script_path, name=None, intent=None)`** runs a zero-ceremony script — bare `@task` calls and `@check` declarations, no `Workspace` or `start_run` of its own — inside a fresh run that lands in quarantine. The result carries the `run_id`, final `state` (`verified` if all checks passed), check counts, and the captured `output`. On failure it includes the structured `failure` record; if even recording the failure failed (storage died mid-crash), a raw `traceback` string appears instead. Always pass `intent` — why this run exists.

**`show_run(run_id)`** is the evidence surface. Beyond the run's fields it returns check results *with the observed/expected values their assertions compared*, traced tasks with recipes and cache-hit flags, artifacts annotated with `bytes_available` (still stored, or hash-and-discarded), and the full lifecycle history. Failed runs and tasks carry a `failure` record — exception type, message, trimmed traceback, and diagnostic notes — the input for deciding a specific correction instead of retrying blind. Ids accept unique prefixes, git-style, here and in `promote_run`.

**`list_engines()`** answers "what can I compute with, here": SLAB's built-ins (`emt`/`lj`/`mace`/`qe`/`rootstock`), everything the cluster's engine registry declares (with the maintainer's declared versions and whether a probe verifies each entry), and — under `rootstock` — the canonical MLIP checkpoint ids the local rootstock install serves, each usable directly as the `engine=` argument. It also lists the named QE input protocols (`qe_protocols`) and the installed pseudopotential families (`pseudo_families`) — see [Engines](engines.md) and [Protocols & pseudopotentials](protocols-and-pseudos.md).

## A session: fail, inspect, correct, promote

A representative exchange, payloads abbreviated but structurally truthful.

<!-- no-verify -->
```json
launch_workflow({"script_path": "probe.py",
                 "intent": "rattle Cu hard (stdev=0.5) to probe basin escape"})
```

<!-- no-verify -->
```json
{"run_id": "01k4q8...", "name": "probe", "state": "quarantined", "status": "failed",
 "intent": "rattle Cu hard (stdev=0.5) to probe basin escape",
 "error": "LinAlgError: Eigenvalues did not converge",
 "checks_passed": 0, "checks_total": 0, "tasks_recorded": 1,
 "failure": {"type": "LinAlgError", "message": "Eigenvalues did not converge",
             "traceback": "Traceback (most recent call last):\n  ...",
             "notes": ["relax failed after 3 completed step(s); trajectory has 4 frame(s), last frame: E=41.283624 eV, max|F|=63.1042 eV/Å; partial trajectory kept as artifact 'relax-failed.traj'"]},
 "output": ""}
```

The note already contains the diagnosis: after three steps the structure sits 41 eV high with 63 eV/Å residual forces — the rattle destroyed the crystal rather than perturbing it. `show_run("01k4q8")` would add the per-task failure record, the recipe that produced it, and the kept `relax-failed.traj` for actual inspection. The agent shrinks the perturbation and relaunches:

<!-- no-verify -->
```json
launch_workflow({"script_path": "probe.py",
                 "intent": "retry with stdev=0.05: 0.5 destroyed the lattice (E=+41 eV after 3 steps)"})
```

<!-- no-verify -->
```json
{"run_id": "01k4q9...", "state": "verified", "status": "completed",
 "checks_passed": 1, "checks_total": 1, "tasks_recorded": 1,
 "output": "E = -0.026784 eV  fmax = 0.0199\n"}
```

Checks passed, so the run left quarantine on its own. Permanence is still a separate, explicit decision:

<!-- no-verify -->
```json
promote_run({"run_id": "01k4q9", "reason": "converged baseline after correcting stdev"})
```

<!-- no-verify -->
```json
{"id": "01k4q9...", "state": "promoted", "status": "completed", ...}
```

The failed probe stays in quarantine, partial trajectory and all, until its TTL — diagnostics self-clean instead of accumulating. See [Debugging failures](debugging-failures.md) for the full evidence contract.

## Intent, and lifecycle hygiene for agents

`intent` is narrative provenance: the *why* that a recipe cannot capture. It is stored on the run, shown by `list_runs` and `show_run`, and — as the retry above shows — the natural place to record what the previous attempt taught. Weeks later, an agent (the same one or another) querying the workspace reads intents as the lab notebook: which runs were baselines, which were corrections, which were speculative. The lifecycle guidance follows from SLAB's one asymmetry ([Lifecycle & retention](lifecycle-and-retention.md)): promote only what deserves keeping, always with a reason; let everything else expire, and run `expire_runs` + `gc` periodically to reclaim it. Promotion is the only path to permanence, so an agent that never promotes leaves nothing behind — and one that promotes indiscriminately recreates the archive-of-failures problem SLAB exists to avoid.

## Under the hood

Every tool calls `slab._ops` — the functions below are exactly what `launch_workflow` and `show_run` run, so you can reproduce the agent's view without an MCP client. First, the workflow script an agent would launch (zero ceremony: no `Workspace`, no `start_run` — the runner supplies both):

```python
from pathlib import Path

from slab import Workspace
from slab._ops import launch_script, run_details

Path("relax_cu.py").write_text('''\
from ase.build import bulk
from slab import check, converged, current_run
from slab.tasks import relax

atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
atoms.rattle(stdev=0.05, seed=42)
relaxed, info = relax(atoms, engine="emt", fmax=0.05)
print(f"E = {info['energy']:.6f} eV  fmax = {info['fmax']:.4f}")

@check
def forces_converged():
    return converged(info["fmax"], below=0.05, label="fmax")

current_run().keep("relaxed.xyz", relaxed)
''')

result = launch_script(
    Path("agent-ws"), "relax_cu.py",
    intent="baseline Cu relax with EMT", capture_output=True,
)
print(result["state"], f'{result["checks_passed"]}/{result["checks_total"]} checks passed')
print("captured:", result["output"].strip())

with Workspace("agent-ws") as ws:
    details = run_details(ws, result["run_id"])

gate = details["checks"][0]
print(gate["name"], "observed:", gate["observed"], "expected:", gate["expected"])
print([(a["name"], a["role"]) for a in details["artifacts"]])
```

```text
verified 1/1 checks passed
captured: E = -0.026784 eV  fmax = 0.0199
forces_converged observed: 0.01990506040266342 expected: {'below': 0.05}
[('relax.traj', 'intermediate'), ('relaxed.xyz', 'terminal')]
```

The numbers are deterministic (fixed rattle seed, EMT); only run ids and timestamps vary. Note the shapes: the check stored the observed residual and the threshold it was compared against, and the artifacts carry roles — the trajectory is an *intermediate* (hash-and-discarded once retention tiers kick in), the declared result a *terminal* (kept in full if this run is ever promoted). That role distinction is the whole retention story, told in [Lifecycle & retention](lifecycle-and-retention.md); how these runs get built in the first place is the [Quickstart](quickstart.md), and the design argument is the [Architecture](../architecture.md) page.
