# Agents over MCP

LLM agents are SLAB's primary user, so the workspace speaks their native protocol. `slab mcp` serves the same operations as the CLI as MCP tools over stdio, with one shared code path in `foundation._ops` and two skins, and the tools return structured JSON instead of formatted text.

This page is for *external* agents such as Claude. The *resident* agent is Mason, with its own [roster of specialists and skills](roster-and-skills.md); both drive the same workspace. The tool set is the resident agent's, so a harness that drives Foundation over MCP can be measured on the same [benchmark](../benchmark.md).

## Setup

The server ships as an extra, and any MCP client launches it as a subprocess:

<!-- no-verify -->
```bash
pip install 'slab-stack[mcp]'
```

<!-- no-verify -->
```json
{"mcpServers": {"slab": {"command": "slab", "args": ["mcp"]}}}
```

The workspace is resolved exactly as for the CLI: `-w/--workspace` flag > `$SLAB_WORKSPACE` > `./.slab`. So `{"args": ["mcp", "-w", "/scratch/proj/.slab"]}` pins a specific one. There is no daemon and no database server, because the workspace is a directory, and concurrent CLI and MCP access coexist at the SQLite transaction level.

!!! note
    Under stdio MCP, stdout is the protocol channel. `launch_workflow` therefore redirects everything a workflow script prints into the result's `output` field, including the checks, which evaluate at run exit. Scripts can `print()` freely, and nothing corrupts the wire.

## The toolbox

Twenty-four tools, each a thin wrapper over the operations layer, and three more on a cluster:

| Tool | What it does |
| --- | --- |
| `launch_workflow` | Execute a plain-Python workflow script in a fresh traced run that carries this server's session id. `ntasks`, `threads`, and `gpus` size the run. The server reserves the slice, the run takes it as an affinity mask plus `CUDA_VISIBLE_DEVICES`, and a slice that does not fit what is free is refused with the free amounts. |
| `wait_for_run` | Block until a run finishes or the timeout passes. Takes an id, a prefix, or a run name; without one, waits for every running run of this session. A run whose recorded process on this host is gone is marked failed and answered at once with outcome `process_gone`. |
| `list_runs` | Runs newest first, filterable by lifecycle `state`, execution `status`, and the `session` that created them. Marks failed every running run whose recorded process on this host is gone before it lists. |
| `show_run` | Everything about one run: checks, tasks, artifacts, history, failure evidence. |
| `promote_run` | Make a run permanent (`verified -> promoted`), with a recorded reason. |
| `list_sessions` | The client sessions that created runs, with run counts and state breakdowns. |
| `promote_session` | Promote every run one session created, reporting each outcome. |
| `expire_runs` | Expire unpromoted runs past their TTL. `older_than="0d"` means everything, now. |
| `gc` | Drop artifact bytes no retention rule demands. `dry_run=True` only reports. |
| `list_engines` | Built-in engines, the cluster registry's declarations, rootstock checkpoint ids, QE protocols, installed pseudo families, the configured builders, each partition's declared node, and this host's `budget` with what is `free` right now. |
| `list_tasks` | The traced tasks a workflow script may call: name, signature, and a one-line summary each. |
| `describe_task` | One task's full signature and docstring. |
| `search_materials` | Filtered search over the offline Materials Project snapshot (`[builders.mp]`): elements, ranges, ordering, a row cap. |
| `get_material` | One snapshot record by material id, with its elements and the resolved CIF path. Absence is reported as absence; there is no online fallback. |
| `query_materials` | One read-only `SELECT` over the snapshot's metadata database, for what the filters cannot express. |
| `submit_job`, `job_status`, `cancel_job` | SLURM batch jobs, with the scripts kept under the workspace's `jobs/` directory. Present only when `slab.toml` configures `[hpc]` partitions. `submit_job` takes `nodes`, `ntasks_per_node`, `cpus_per_task`, `gpus_per_node`, and `mem`; the size replaces the partition's directives and must fit the node the partition declares. |
| `notebook` | Append a dated entry to the project's `NOTEBOOK.md`, or read its latest entries. |
| `plan` | Rewrite the project's `PLAN.md`, or read it. |
| `list_memories`, `recall`, `remember` | The machine's memory: what earlier sessions on this machine recorded about its software. See [Machine memory](memory.md). |
| `list_skills`, `skill` | The skill catalog, and one skill's instructions with its bundled files. The catalog is the one the resident agent loads. |
| `report_results` | Record the session's answer: results with units, and the run ids that produced them. |
| `retire_session` | Call it last. Promote the verified runs the answer rests on, anchors from earlier sessions included, and expire this session's other runs. A cited run that is not verified is reported, never forced. `uncited` is `keep`, `expire`, or `purge`; `dry_run=True` only reports. |

The server holds one session id for its lifetime, `mcp-<stamp>-<pid>`, and states it in its instructions. Every run it launches carries that id, so `list_runs(session=...)` and `promote_session` see the session whole. The project directory is the one the server was started in: its `slab.toml`, its notebook and plan, and its `skills/` directory apply.

## What stays in Mason

Three of the resident agent's tools are not offered over MCP, because they are mechanisms of one agent loop and not surfaces of the workspace:

- `delegate` hands a brief to a specialist's own loop. A harness has its own way to spawn agents.
- `review` hands the plan to the critic and gates compute on the verdict. A harness reviews its own plans.
- `finish` ends the loop with a report. Over MCP the session stays open, and `report_results` records the answer without ending anything.

Files and a shell are not offered either. A harness brings its own. The science review that flags a campaign reads Mason transcripts, so a harness session is scored but not reviewed.

**`launch_workflow(script_path, name=None, intent=None, ntasks=None, threads=None, gpus=None)`** runs a zero-ceremony script inside a fresh run that lands in quarantine. A zero-ceremony script has bare `@task` calls and `@check` declarations, with no `Workspace` or `start_run` of its own. The result carries the `run_id`, the final `state` (`verified` if all checks passed), the check counts, the `resources` the run held, and the captured `output`. On failure, it includes the structured `failure` record, and if even recording the failure failed (storage died mid-crash), a raw `traceback` string appears instead. Always pass `intent`, which says why this run exists.

The three size arguments make the launch sized. The server reserves `ntasks x threads` cpu ids and `gpus` gpu ids of its host before the run starts, runs the script as a child process that claims the reservation, and the child takes the slice as its affinity mask and `CUDA_VISIBLE_DEVICES`. A slice that does not fit what is free is refused with the free amounts, and `list_engines` reports `budget` and `free` so a harness sizes within them. An unsized launch runs in the server's process and reserves every free cpu. The Mason tool takes the same three arguments, and the reservation is released when the run ends or the holder dies; see [Mason](mason.md#compute-budget-sizing-the-physics-to-the-machine) for the guarantee.

**`show_run(run_id)`** is the evidence surface. Beyond the run's fields, it returns check results with the observed/expected values their assertions compared, traced tasks with recipes and cache-hit flags, artifacts annotated with `bytes_available` (still stored, or hash-and-discarded), and the full lifecycle history. Failed runs and tasks carry a `failure` record with the exception type, message, trimmed traceback, and diagnostic notes, which is the input for a specific correction instead of a blind retry. Ids accept unique prefixes, git-style, here and in `promote_run`.

**`list_engines()`** answers "what can I compute with, here". It lists SLAB's built-ins (`emt`/`lammps`/`lj`/`qe`/`rootstock`), and everything the cluster's engine registry declares, with the maintainer's declared versions and whether a probe verifies each entry. Under `rootstock`, it lists the canonical MLIP checkpoint ids the local rootstock install serves, each usable directly as the `engine=` argument. It also lists the named QE input protocols (`qe_protocols`) and the installed pseudopotential families (`pseudo_families`). See [Engines](engines.md) and [Protocols & pseudopotentials](protocols-and-pseudos.md).

## A session: fail, inspect, correct, promote

A representative exchange, with payloads abbreviated but structurally truthful.

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

The note already contains the diagnosis. After three steps, the structure sits 41 eV high with 63 eV/Å residual forces, so the rattle destroyed the crystal instead of perturbing it. `show_run("01k4q8")` would add the per-task failure record, the recipe that produced it, and the kept `relax-failed.traj` for inspection. The agent shrinks the perturbation and relaunches:

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

The failed probe stays in quarantine, partial trajectory and all, until its TTL, so diagnostics self-clean instead of accumulating. See [Debugging failures](debugging-failures.md) for the full evidence contract.

## A benchmark question, scripted

The MCP tools are the whole surface a campaign needs. The script below drives question 1 of the benchmark through the server in one process, the way a harness would drive it over stdio, and then scores the session it left. Question 1 asks for the lattice constant of fcc copper. The client loads a skill, launches a workflow, writes the notebook, reports the result against the run, and the scorer reads the record:

```python
import asyncio
from pathlib import Path

from foundation.mcp_server import build_server
from slab_stack import benchmark

project = Path("project")
project.mkdir(exist_ok=True)
server = build_server(Path("agent-ws"), project=project, session="mcp-demo")


def call(tool, args=None):
    result = asyncio.run(server.call_tool(tool, args or {}))
    structured = result[1] if isinstance(result, tuple) else result.structured_content
    return structured["result"] if set(structured) == {"result"} else structured


tools = sorted(t.name for t in asyncio.run(server.list_tools()))
print(len(tools), "tools:", ", ".join(tools))

(project / "a0.py").write_text('''\
from ase.build import bulk
from foundation import check, converged
from foundation.tasks import relax_cell

atoms = bulk("Cu", "fcc", a=3.6)
relaxed, info = relax_cell(atoms, engine="emt", fmax=0.01)
a0 = (4 * relaxed.get_volume()) ** (1 / 3)
print(f"a0 = {a0:.4f} Å")

@check
def forces_converged():
    return converged(info["fmax"], below=0.01)
''')
loaded = call("skill", {"name": "equation-of-state"})
print("skill:", loaded["name"], "files:", loaded["files"])
launched = call("launch_workflow", {"script_path": "project/a0.py", "intent": "Q1: a0 of fcc Cu under emt"})
print(launched["state"], f'{launched["checks_passed"]}/{launched["checks_total"]} checks passed;', launched["output"].strip())
a0 = float(launched["output"].split("a0 = ")[1].split()[0])
call("notebook", {"entry": f"a0 = {a0} Å (run {launched['run_id'][:10]})", "heading": "Q1"})
reported = call("report_results", {"results": {"a0": {"value": a0, "unit": "Å"}}, "run_ids": [launched["run_id"]]})
print("reported for session", reported["session"], "->", Path(reported["recorded"]).name)
retired = call("retire_session", {"run_ids": [launched["run_id"]]})
print("retired:", retired["runs_promoted"], "promoted,", retired["runs_expired"], "expired of", retired["runs_total"], "run(s)")

record = benchmark.score_session(Path("agent-ws"), "mcp-demo", question=benchmark.find_question("1"))
print("scored:", record["passed"], record["engine_class"], record["engines"], record["reviewed_by"])
```

```text
24 tools: describe_task, expire_runs, gc, get_material, launch_workflow, list_engines, list_memories, list_runs, list_sessions, list_skills, list_tasks, notebook, plan, promote_run, promote_session, query_materials, recall, remember, report_results, retire_session, search_materials, show_run, skill, wait_for_run
skill: equation-of-state files: ['SKILL.md', 'assets/eos_scan.py', 'scripts/fit_eos.py']
verified 1/1 checks passed; a0 = 3.5907 Å
reported for session mcp-demo -> mcp-demo.jsonl
retired: 1 promoted, 0 expired of 1 run(s)
scored: True mlip ['emt'] []
```

The record is one JSON lines file under `<workspace>/sessions/`, with the skills the session loaded, the results it reported, and the retire outcome. `slab benchmark score --session mcp-demo --question 1` scores it from the command line. The question must be named, because a harness record holds no opening instruction, and the last field printed is the empty list of reviewers: the science review reads Mason transcripts only.

`retire_session` is the last call of a session. It promotes the verified runs the answer cites and expires the session's other runs, which cannot be undone, so report first and retire once. The default for the uncited runs comes from the workspace's retention policy. See [Lifecycle & retention](lifecycle-and-retention.md).

## Intent, and lifecycle hygiene for agents

`intent` is narrative provenance, the why that a recipe cannot capture. SLAB stores it on the run and shows it in `list_runs` and `show_run`, and as the retry above shows, it is the natural place to record what the previous attempt taught. Weeks later, an agent (the same one or another) that queries the workspace reads the intents as the lab notebook, which shows which runs were baselines, which were corrections, and which were speculative.

The lifecycle guidance follows from SLAB's one asymmetry ([Lifecycle & retention](lifecycle-and-retention.md)):

- Promote only what deserves keeping, always with a reason.
- Let everything else expire, and run `expire_runs` + `gc` periodically to reclaim it.
- End a session with `retire_session`: cite the runs the answer rests on, and let the session's other runs expire.
- Read what ran. The session record holds a `command` event for every job `submit_job` submitted and, once a run finishes under `launch_workflow` or `wait_for_run`, one for each distinct engine command its tasks resolved, with the setup lines and the KOKKOS switches a LAMMPS command asks for. A GPU result whose command has no `-k on` was computed on the host. `list_engines` lists every LAMMPS route with its command and switches, and `engine=` picks one per call.

Promotion is the only path to permanence, so an agent that never promotes leaves nothing behind, and an agent that promotes indiscriminately recreates the archive-of-failures problem that SLAB exists to avoid.

## Under the hood

Every tool calls `foundation._ops`, and the functions below are exactly what `launch_workflow` and `show_run` run, so you can reproduce the agent's view without an MCP client. First, the workflow script an agent would launch. It has zero ceremony, with no `Workspace` and no `start_run`, because the runner supplies both:

```python
from pathlib import Path

from foundation import Workspace
from foundation._ops import launch_script, run_details

Path("relax_cu.py").write_text('''\
from ase.build import bulk
from foundation import check, converged, current_run
from foundation.tasks import relax

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

The numbers are deterministic (fixed rattle seed, EMT), and only run ids and timestamps vary. Note the shapes. The check stored the observed residual and the threshold it was compared against, and the artifacts carry roles: the trajectory is an intermediate, hash-and-discarded once retention tiers apply, while the declared result is a terminal, kept in full if this run is ever promoted. That role distinction is the whole retention model, described in [Lifecycle & retention](lifecycle-and-retention.md). For how these runs are built in the first place, see the [Quickstart](quickstart.md), and for the design argument, see [Architecture](../architecture.md).
