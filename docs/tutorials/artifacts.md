# Artifacts

A run keeps its files as artifacts. Each artifact is a name on the run and
the SHA-256 of its bytes, and the artifact store holds the bytes once under
that hash. This page shows the three ways to reach an artifact:

| Form | Where | What it does |
| --- | --- | --- |
| `read_artifact(run_id, name)` | Mason tool, MCP tool, `foundation._ops.read_artifact` | Read a run's artifact by name. The read follows the run's cache hits to the run that executed the task. |
| `read_artifact(hash=...)` | the same three | Read any stored bytes by a SHA-256 prefix of 6 or more characters, and name the runs and tasks that reference them. |
| `files=["run:<id>/<name> as <basename>"]` | `run_lammps` | Stage a run's artifact beside the script, with no copy through the shell. |

## A cache hit keeps no files

A cache hit executes nothing, so the task keeps none of its files on the
run that asked for it. The files are artifacts of the run where the task
executed. SLAB calls that run the producing run.

The example runs a short argon MD leg twice. The second run is a relaunch,
and its `run_lammps` call is a cache hit:

```python
from ase.build import bulk

from foundation import Workspace
from foundation._ops import read_artifact, run_details
from foundation.tasks import run_lammps

MD = """\
units metal
atom_style atomic
read_data structure.data
pair_style lj/cut 8.5
pair_coeff 1 1 0.0104 3.40
velocity all create 60 42
fix nvt all nvt temp 60 60 0.1
thermo_style custom step temp pe press
thermo_modify line yaml
thermo 50
run 200
write_data final.data nocoeff
"""
argon = bulk("Ar", "fcc", a=5.26, cubic=True) * (2, 2, 2)
ws = Workspace("workspace")

with ws.start_run(name="md") as first:
    run_lammps(MD, atoms=argon, label="ar")
with ws.start_run(name="md-relaunch") as again:
    run_lammps(MD, atoms=argon, label="ar")

print(ws.runs.list_artifacts(again.id))
task = run_details(ws, again.id)["tasks"][0]
print(task["cache_hit"], task["artifacts_on"] == first.id)
```

```text
[]
True True
```

The relaunch holds no artifacts. Its task record carries `artifacts_on`,
which names the producing run. `show_run` shows the same field on every
cache-hit task, over MCP and in Mason.

## Read by name

Name the relaunch and the artifact. The read finds no `ar-final.data` on
the relaunch, so it follows the cache hit to the producing run. The first
line of the answer says so:

```python
read = read_artifact(ws, run_id=again.id, name="ar-final.data")
print("\n".join(read.head))
print(read.text.splitlines()[2])
```

```text
ar-final.data is read from run 01m2gevcnm: task 2 run_lammps on run 01m2gevcsj was a cache hit of that run
ar-final.data (4312 bytes, sha256 0f9896dc68d5)
32 atoms
```

When neither run holds the name, the error lists the names each run holds.

## Stage a run's artifact for the next task

An entry in `run_lammps(files=...)` is a path or a run artifact reference.
The reference has two forms:

- `run:<id>/<name>` stages the artifact under its own name.
- `run:<id>/<name> as <basename>` stages it under *basename*.

The id is a full run id or a unique prefix. The reference follows the run's
cache hits, the same way a read by name does. Name the staged file by its
bare basename in the script.

The quench below starts from the MD leg's final configuration. It names the
relaunch, which holds no files of its own:

```python
QUENCH = """\
units metal
atom_style atomic
read_data start.data
pair_style lj/cut 8.5
pair_coeff 1 1 0.0104 3.40
thermo_style custom step pe press
thermo_modify line yaml
minimize 1e-10 1e-10 1000 10000
"""
start = f"run:{again.id}/ar-final.data as start.data"
with ws.start_run(name="quench") as quench:
    result, info = run_lammps(QUENCH, files=[start], label="q")
print(round(result["thermo"]["PotEng"], 4))
record = ws.runs.list_tasks(quench.id)[0]
print(record.recipe["extra"]["provenance"])
```

```text
-2.6953
{'files': {'start.data': 'run:01m2gevcnma3exe6e82zc6cgkh/ar-final.data'}}
```

The MD script writes the data file with `nocoeff`. A plain `write_data`
adds a `Pair Coeffs` section, and `read_data` of that file before
`pair_style` stops LAMMPS with `Must define pair_style before Pair Coeffs`.

The recipe records the run that holds the bytes, and the cache key does
not. The reference enters the cache identity through the artifact's hash
and the staged basename, so a reference to other bytes recomputes and a
reference to the same bytes on another run is a cache hit:

```python
with ws.start_run(name="quench-again") as repeat:
    run_lammps(QUENCH, files=[f"run:{first.id}/ar-final.data as start.data"], label="q")
print(ws.runs.list_tasks(repeat.id)[0].cache_hit)
```

```text
True
```

A reference is refused before LAMMPS starts when the run holds no such
artifact, when retention has discarded its bytes, or when the script never
names the basename. A dry run reads the reference from the real workspace,
because its own throwaway store holds no earlier run.

## Read by hash

`show_run` lists each task's inputs and outputs as hashes. Pass a hash
prefix alone to read those bytes. The answer names every run artifact and
every task slot that holds the hash:

```python
digest = ws.runs.get_artifact(first.id, "ar-final.data").hash
print("\n".join(read_artifact(ws, digest=digest[:10]).head))
```

```text
sha256 0f9896dc68d5 (4312 bytes)
referenced by: run 01m2gevcnm artifact ar-final.data (intermediate)
```

A task's input or output is a serialized value. A read by hash decodes it,
JSON as indented text and anything else as its `repr`:

```python
value = read_artifact(ws, digest=record.outputs["return[0]"][:10])
print(value.head[1])
print("\n".join(value.text.splitlines()[:4]))
```

```text
referenced by: run 01m2gevcsm task 3 run_lammps output return[0]; run 01m2gevcvk task 4 run_lammps output return[0]
{
 "artifacts": {
  "averages": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
  "thermo": "84d538aeb0f9168e70a8c27ffd63484759e1bac31f34d5464a38412a07c0f60b"
```

When retention has discarded the bytes and a run still records the hash,
the error says so and names the run.

## The tools

The Mason tool and the MCP tool take the same arguments: `run_id` and
`name`, or `hash` alone. The Mason tool returns the text line-numbered and
digests a recognised engine output, as before. The MCP tool returns a
`head`, the `text` of the window that `offset` and `limit` select, and the
total number of `lines`.

A follow reads the producing run's record, and that run has its own
lifecycle. Retention can expire and purge the producing run while the
relaunch stays promoted. Promote the producing run too when a result rests
on its files.
