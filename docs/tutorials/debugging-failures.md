# Debugging failures

When a run fails, SLAB does not retry, restart, or route the error anywhere.
It captures structured evidence — exception, trimmed traceback, diagnostic
notes, the scratch data that explains what happened — and stops.

## Evidence, not protocol

AiiDA-style engines treat failure as a dispatch problem: a calculation exits
with a code, and the code routes to a predefined restart handler written by
whoever anticipated that failure mode. That design assumes the set of
corrections is enumerable in advance.

SLAB's user is an LLM agent (or a human at a terminal) that can improvise a
*niche* correction — shrink the perturbation, switch the engine, loosen a
threshold — **if** it can see what actually happened. So the contract is
evidence delivery: failed runs and tasks carry a structured `failure` record,
and delivery is tiered so listings stay cheap. `slab list` shows a one-line
`error` per run; `slab show <id>` fetches the full record for the one run you
are actually debugging.

## A failing task

The evidence channel is plain [PEP 678](https://peps.python.org/pep-0678/):
`Exception.add_note`. No SLAB import, no special error type — annotate the
exception at the point of failure, where the diagnostic values are still in
scope, and the tracer does the rest. Here is a task standing in for an engine
call that diverges:

```python
from slab import Workspace, task

@task
def scf_energy(kpts: int) -> float:
    exc = RuntimeError("SCF failed to converge in 200 iterations")
    exc.add_note("last SCF residual: 3.2e-2 (target 1e-6)")
    exc.add_note(f"kpts={kpts}; residual stopped decreasing after iteration 120")
    raise exc

ws = Workspace("evidence-demo")
try:
    with ws.start_run(name="mgo-scf", intent="k-point convergence probe") as run:
        scf_energy(kpts=4)
except RuntimeError as exc:
    print("caught:", exc)
```

```text
caught: SCF failed to converge in 200 iterations
```

The exception propagated normally — SLAB never swallows your errors — but on
the way out, two things were recorded: the task's record was marked failed
with the evidence attached, and the run itself was marked failed.

## Reading the record

The failed task record carries a one-line `error` for listings and a
structured `failure` dict for debugging — `type`, `message`, `traceback`, and
`notes`:

```python
record = ws.runs.list_tasks(run.id)[0]
print(record.status.value, "-", record.error)
print(record.failure["type"], "/", record.failure["message"])
for note in record.failure["notes"]:
    print("note:", note)
print("\n".join(record.failure["traceback"].splitlines()[-3:]))
```

```text
failed - RuntimeError: SCF failed to converge in 200 iterations
RuntimeError / SCF failed to converge in 200 iterations
note: last SCF residual: 3.2e-2 (target 1e-6)
note: kpts=4; residual stopped decreasing after iteration 120
RuntimeError: SCF failed to converge in 200 iterations
last SCF residual: 3.2e-2 (target 1e-6)
kpts=4; residual stopped decreasing after iteration 120
```

The record is information-rich but token-bounded, because its consumer reads
it verbatim. The traceback keeps the entry point and the failure site of each
contiguous frame group (first 3 and last 5 frames) and elides deep middles
with a `[... N frame(s) elided ...]` marker; every message piece is clipped at
2000 characters so a giant exception message can never crowd out the frames;
the whole text caps at 10000 characters, keeping the *end* of a long exception
chain. Notes are capped at 10. Notes also appear at the end of the formatted
traceback (that is standard PEP 678 rendering), but listing them separately in
`notes` means a reader can act on them without parsing the traceback at all.

The run carries the same evidence. When an exception escapes the
`start_run` block, the run is marked `failed`, checks are skipped, and the
`failure` record lands on the run itself:

```python
failed = ws.runs.get(run.id)
print(failed.state.value, "/", failed.status.value)
print(failed.error)
print(failed.failure["type"] == record.failure["type"])
```

```text
quarantined / failed
RuntimeError: SCF failed to converge in 200 iterations
True
```

Note the lifecycle state: still `quarantined`. Failure is an execution
status, not a lifecycle state — the failed run sits in quarantine like any
other unverified run, fully inspectable until it expires.

## What `relax` does for you

The built-in [relax task](engines.md) applies the same pattern with domain
knowledge added. On a mid-optimization crash — an engine exception, a worker
dying — it captures the evidence before the scratch directory vanishes:

- The exception gets a note with the completed step count and the last
  trajectory frame's energy and residual force.
- Inside a run, the partial trajectory is kept as an artifact named
  `{label or 'relax'}-failed.traj` — so you can see whether the structure flew
  apart or the energy oscillated, frame by frame.
- Untraced calls (outside `start_run`) still get the note; there is just no
  artifact store to keep the trajectory in.

<!-- no-verify -->
```python
try:
    with ws.start_run(name="cu-relax", intent="probe rattled Cu") as run:
        relax(atoms, engine="some-flaky-engine", label="cu")
except RuntimeError as e:
    print(e.__notes__)
```

```text
["relax failed after 3 completed step(s); trajectory has 3 frame(s), last
frame: E=-0.018471 eV, max|F|=0.3032 eV/Å; partial trajectory kept as
artifact 'cu-failed.traj'"]
```

That note alone often decides the correction: a residual force of 0.3 eV/Å
after 3 steps on a gently rattled crystal says "engine hiccup, retry or switch
engine"; a max|F| in the hundreds says "structure is unphysical, fix the
input". Diagnostics capture is best-effort and never masks the original
error — if keeping the trajectory itself fails (disk full), the note says so
and the original exception still propagates.

## When the engine writes files

In-process engines fail as Python exceptions with meaningful messages.
File-IO engines ([Quantum ESPRESSO](engines.md#quantum-espresso)) fail as a
bare `CalledProcessError: ... returned non-zero exit status 2` — `pw.x`
wrote the actual story to files in a scratch directory that is about to
vanish. So the failure path reads the story back out: QE's fenced
`Error in routine ...` block (or, when there is no block, flagged stop lines
and the output tail, plus stderr) becomes notes, and the engine's
input/output/`CRASH` files are kept as artifacts. A missing pseudopotential
file, captured from a real run (paths shortened):

<!-- no-verify -->
```text
notes:
- relax failed after 0 completed step(s)
- engine error (espresso.pwo): Error in routine readpp (1):
  file /opt/pseudos/sssp/DoesNotExist.UPF not found
- engine stderr tail (espresso.err): STOP 1
- engine files kept as artifacts: 'si-failed.pwi', 'si-failed.pwo', 'si-failed.err'
```

And the classic — an SCF that cannot converge (here capped at one iteration),
which QE reports *without* a fenced error block:

<!-- no-verify -->
```text
- engine output flagged (espresso.pwo): convergence NOT achieved after   1 iterations: stopping
```

"Exit status 2" invites a blind retry; "smearing is needed" or "convergence
NOT achieved" is a correction an agent can actually compute. The kept
`si-failed.pwi` is the exact input that crashed — reproducible outside SLAB
with nothing but `pw.x`.

## Checks as correction inputs

A run that *completes* but fails [verification](verification.md) is the other
debugging case. Check results store the `observed`/`expected` values their
assertions compared — the numbers a correction is computed from:

```python
from slab import check, converged

with ws.start_run(name="cu-verify", intent="threshold probe") as verify_run:
    fmax = 0.062

    @check
    def forces_converged():
        return converged(fmax, below=0.05, label="fmax")

result = ws.runs.list_check_results(verify_run.id)[0]
print(result.passed, "-", result.message)
print("observed:", result.observed, " expected:", result.expected)
print(ws.runs.get(verify_run.id).state.value)
ws.close()
```

```text
False - fmax=0.062 !< 0.05
observed: 0.062  expected: {'below': 0.05}
quarantined
```

"fmax was 0.062 against 0.05" is enough to compute the fix — rerun with more
steps, or a tighter optimizer — without re-reading the workflow.

!!! note
    Failure diagnostics self-clean. Failed runs sit in quarantine under the
    default 30-day TTL, so kept evidence (partial trajectories included)
    survives long enough to be debugged and then
    [expires with the run](lifecycle-and-retention.md). Nothing accumulates
    forever unless you promote it.

## The surfaces

`slab show` renders each traceback under its owner: the run's `failure`
prints at run level *unless* a failed task carries the same exception (the
usual case — it propagated), in which case it renders once, under that task.
For the failed relax run above:

<!-- no-verify -->
```text
$ slab show 01k...
run 01k...  cu-relax
  state:   quarantined    status: failed
  error:   RuntimeError: SCF diverged
  created: 2026-08-12T02:06:21.820454+00:00
  intent:  probe rattled Cu
  tasks:
    1. relax  failed  0.011s  error: RuntimeError: SCF diverged
       Traceback (most recent call last):
         File ".../slab/tracing.py", line 223, in _traced_call
           result = f(*args, **kwargs)
         File ".../slab/tasks.py", line 119, in relax
           converged = bool(optimizer.run(fmax=fmax, steps=steps))
         File ".../ase/optimize/optimize.py", line 506, in run
           return Dynamics.run(self, steps=steps)
         [... 2 frame(s) elided ...]
         File ".../ase/calculators/calculator.py", line 517, in get_property
           self.calculate(atoms, [name], system_changes)
       RuntimeError: SCF diverged
       relax failed after 3 completed step(s); trajectory has 3 frame(s),
       last frame: E=-0.018471 eV, max|F|=0.3032 eV/Å; partial trajectory
       kept as artifact 'cu-failed.traj'
  artifacts:
    cu-failed.traj  intermediate  3581B  bytes  8c700c74ed78
```

`slab show <id> --json` emits the same details machine-readably — `failure`
keys on the run and on each task, `observed`/`expected` on each check. Agents
get identical structures without the CLI: the MCP `show_run` tool returns this
JSON, and `launch_workflow` returns the `failure` record directly in its
result when a launched script fails — see [Agents over MCP](agents-mcp.md).
The philosophy behind the whole arrangement is argued in
[Architecture](../architecture.md).
