# Caching & resume

SLAB has no `--resume` flag, no checkpoint files, and no restart handlers.
Rerunning the script is the resume mechanism. Every `@task` call is
content-hash cached across runs in a workspace, so finished work is skipped
and the script continues exactly where it died.

## The model

Inside a run, each `@task` call serializes its bound arguments (defaults
applied), content-hashes them into the workspace's artifact store, and
computes a cache key. The key fingerprints everything that determines the
result:

- the function's module and qualname;
- a hash of its source, and a hash of its bytecode (so REPL- and
  `exec`-defined functions are covered too);
- the fingerprint of every closure cell value;
- the resolved versions of any declared engines;
- the task's `cache_extra` contribution;
- the input hashes.

A completed task with the same key may exist from any earlier run in the
workspace. If it does, and its output bytes are still present, the stored
outputs are returned without executing.

Two honesty rules bound the cache. First, failed tasks never populate it. A
retry after an exception computes for real and never serves a poisoned
entry. Second, if retention has discarded the output bytes (see
[Lifecycle & retention](lifecycle-and-retention.md)), the same key is a cache
miss and the task recomputes. A hash-only skeleton is a recipe, not a
result.

## Seeing the cache work

The tracer is inert outside a run. A decorated function called at module
scope is just the function. Nothing is hashed, and nothing is recorded:

```python
from slab import Workspace, task

@task
def scale(x):
    print("computing...")
    return 3 * x

print(scale(2.0))  # no active run: a plain call
```

```text
computing...
6.0
```

Inside runs, the first call with a given input computes. The second call is
served from cache without any message, even in a different run:

```python
ws = Workspace("workspace")

with ws.start_run(name="first") as first:
    print(scale(2.0))

with ws.start_run(name="second") as second:
    print(scale(2.0))            # silent: no "computing..."

print(ws.runs.list_tasks(first.id)[0].cache_hit)
print(ws.runs.list_tasks(second.id)[0].cache_hit)
```

```text
computing...
6.0
6.0
False
True
```

The cache hit is not invisible bookkeeping. The second run gets a full
`TaskRecord`, with the same recipe and the same input and output hashes, and
with `cache_hit=True`. Provenance is complete either way. A changed input is
a different key, and it computes honestly:

```python
with ws.start_run(name="third") as third:
    print(scale(4.0))
print(ws.runs.list_tasks(third.id)[0].cache_hit)
```

```text
computing...
12.0
False
```

## Crash, rerun, resume

Here is the mechanism doing its real job. A two-task pipeline dies in task
two on the first execution. The example simulates a node failure with a flag
file. The fix is not a restart API. The fix is to run the same script again:

```python
from pathlib import Path

@task
def build_supercell(n):
    print("build_supercell: computing")
    return n * 2

@task
def analyze(y):
    print("analyze: computing")
    if not Path("survived-once.flag").exists():
        Path("survived-once.flag").touch()
        raise RuntimeError("node died mid-analysis")
    return y * 10

def pipeline():
    with ws.start_run(name="pipeline", intent="crash-resume demo") as run:
        a = build_supercell(3)
        print("result:", analyze(a))
    return run

try:
    pipeline()                    # first execution: crashes in analyze
except RuntimeError as e:
    print("crashed:", e)

resumed = pipeline()              # the rerun IS the resume
for t in ws.runs.list_tasks(resumed.id):
    print(t.name, t.status.value, "cache_hit =", t.cache_hit)
```

```text
build_supercell: computing
analyze: computing
crashed: node died mid-analysis
analyze: computing
result: 60
build_supercell completed cache_hit = True
analyze completed cache_hit = False
```

`build_supercell` succeeded before the crash, so its result was cached even
though its run failed. Caching is per-task, not per-run. On the rerun, the
cache serves it without a message, and only `analyze` executes. The crashed
run stays behind as a quarantined failed run. It carries the structured
failure evidence (exception type, trimmed traceback, diagnostic notes) and
expires on its TTL like any other unpromoted run.

!!! note
    This is why SLAB commits a provisional `running` row before each task
    executes. A hard-killed process still leaves its input references visible,
    so a concurrent retention sweep cannot remove the roots the resume will
    need.

## What invalidates a cached result

Anything that would change the answer changes the key. Edit the function
body, even to a version with no retrievable source, and the source and
bytecode hashes miss. Rebind a closure cell (for example, two products of
the same task factory with different bound parameters) and the closure
fingerprints miss.

Environment identity is opt-in and explicit. `@task(engines=("ase",))` pins
the installed `ase` version into the recipe and the key, so a package upgrade
invalidates results computed under the old version. For identity that lives
outside pip, `cache_extra` is a callable that receives the bound arguments
and returns a dict folded into the key.

`relax` wires `cache_extra=describe_engine`. At call time, it resolves what
the `engine` argument actually names (a built-in, a cluster-registry entry
with its full spec, or a rootstock checkpoint) and folds that into the key.
When a maintainer bumps a `qe-delta` site alias from 7.3 to 7.4 in the
cluster's registry file, every cached relax that used it is honestly
invalidated. Any other spec edit (options, env, calculator path) invalidates
the same way. See [Engines](engines.md) for the registry itself.

```python
from ase.build import bulk
from slab.tasks import relax

atoms = bulk("Cu", "fcc", a=3.58) * (2, 2, 2)
atoms.rattle(stdev=0.05, seed=42)

with ws.start_run(name="cu-relax", intent="EMT baseline") as cu:
    relaxed, info = relax(atoms, engine="emt", fmax=0.05)
    print(info["converged"], round(info["energy"], 6))

recipe = ws.runs.list_tasks(cu.id)[0].recipe
print(recipe["engines"]["ase"])   # pinned engine version (yours may differ)
print(recipe["extra"])            # describe_engine's contribution
```

```text
True -0.053417
3.29.0
{'engine': 'emt', 'source': 'builtin', 'version': None}
```

## What the key cannot see

A global that your function reads and mutates between calls is invisible to
the fingerprint. The bytecode references the name, not the value. A task
whose behavior depends on mutated globals can be served a stale hit. Treat
globals as constants, or pass the value as an argument so it is hashed like
any other input.

Closure cells are the opposite case. They are fingerprinted, but a cell that
holds an unserializable value (a lambda, an open handle) cannot be. Serving
one computation's result for another is the one failure the cache must never
produce. Rather than risk that collision, SLAB marks such a call
uncacheable. It records normally but computes every time.

## The DAG is derived, not declared

There is no graph API. Task B consuming task A's output is visible because
B's input hash equals A's output hash. That is provenance by equality:

```python
with ws.start_run(name="dag") as dag:
    analyze(scale(5.0))
first_task, second_task = ws.runs.list_tasks(dag.id)
print(first_task.outputs["return"] == second_task.inputs["y"])
```

```text
computing...
analyze: computing
True
```

Tuple returns are stored element-wise (`return[0]`, `return[1]`, ...). The
`atoms, info = relax(...)` pattern therefore leaves a per-value hash for each
element, and downstream consumers of either one are linked the same way.
Start from the [Quickstart](quickstart.md) if you have not seen runs and
checks yet. For the lifecycle side of what happens to all these runs, see
[Architecture](../architecture.md).
