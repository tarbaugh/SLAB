# Debugging failures

When a run fails, SLAB does not retry, restart, or route the error anywhere.
It captures structured evidence and stops. The evidence is the exception, a
trimmed traceback, diagnostic notes, and the scratch data that explains what
happened.

## Evidence, not protocol

AiiDA-style engines treat failure as a dispatch problem. A calculation exits
with a code, and the code routes to a predefined restart handler written by
whoever anticipated that failure mode. That design assumes the set of
corrections is enumerable in advance.

SLAB's user is an LLM agent, or a human at a terminal, and either one can
improvise a niche correction, such as a smaller perturbation, a different
engine, or a looser threshold, if it can see what actually happened. So the
contract is evidence delivery. Failed runs and tasks carry a structured
`failure` record, and delivery is tiered so that listings stay cheap:
`foundation list` shows a one-line `error` per run, and `foundation show <id>`
fetches the full record for the one run you are debugging.

## A failing task

The evidence channel is plain [PEP 678](https://peps.python.org/pep-0678/),
`Exception.add_note`, with no SLAB import and no special error type.
Annotate the exception at the point of failure, where the diagnostic values
are still in scope, and the tracer does the rest. Here is a task that stands
in for an engine call that diverges:

```python
from foundation import Workspace, task

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

The exception propagated normally, because SLAB never swallows your errors.
On the way out, SLAB recorded two things: it marked the task's record failed
with the evidence attached, and it marked the run itself failed.

## Reading the record

The failed task record carries a one-line `error` for listings and a
structured `failure` dict for debugging, which holds `type`, `message`,
`traceback`, and `notes`:

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
it verbatim. The bounds:

- The traceback keeps the entry point and the failure site of each
  contiguous frame group (the first 3 and last 5 frames) and elides deep
  middles with a `[... N frame(s) elided ...]` marker.
- Every message piece is clipped at 2000 characters, so a giant exception
  message can never crowd out the frames.
- The whole text is capped at 10000 characters, and the cap keeps the end
  of a long exception chain.
- Notes are capped at 10.

Notes also appear at the end of the formatted traceback, which is standard
PEP 678 rendering, but the separate `notes` list means a reader can act on
them without parsing the traceback at all.

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

Note the lifecycle state, which is still `quarantined`. Failure is an
execution status, not a lifecycle state, so the failed run sits in
quarantine like any other unverified run, fully inspectable until it
expires.

## What `relax` does for you

The built-in [relax task](engines.md) applies the same pattern with domain
knowledge added. On a crash in the middle of an optimization, such as an
engine exception or a dying worker, it captures the evidence before the
scratch directory vanishes:

- The exception gets a note with the completed step count and the last
  trajectory frame's energy and residual force.
- Inside a run, the partial trajectory is kept as an artifact named
  `{label or 'relax'}-failed.traj`, so you can see whether the structure
  flew apart or the energy oscillated, frame by frame.
- Untraced calls (outside `start_run`) still get the note, but there is no
  artifact store to keep the trajectory in.
- `single_point` shares the same contract minus the trajectory, because
  nothing is optimized. It puts engine error notes on the exception and
  keeps the engine's own files as `{label or 'single-point'}-failed.*`.

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
frame: E=-0.006493 eV, max|F|=0.4410 eV/Å; partial trajectory kept as
artifact 'cu-failed.traj'"]
```

That note alone often decides the correction. A residual force of 0.3 eV/Å
after 3 steps on a gently rattled crystal says "engine hiccup: retry, or
switch engine", while a max|F| in the hundreds says "the structure is
unphysical: fix the input". Diagnostics capture is best-effort and never
masks the original error, so if keeping the trajectory itself fails (disk
full), the note says so, and the original exception still propagates.

## When the engine writes files

In-process engines fail as Python exceptions with meaningful messages, but
file-IO engines fail with the explanation somewhere else entirely.
[Quantum ESPRESSO](engines.md#quantum-espresso) fails as a bare
`CalledProcessError: ... returned non-zero exit status 2`, while `pw.x`
wrote the actual explanation to files in a scratch directory that is about
to vanish. [LAMMPS](engines.md#lammps) is one step worse, because
`lammpsrun` raises the real `ERROR: ...` inside a reader thread, where no
caller can catch it, and Python sees only
`RuntimeError: Failed to retrieve any thermo_style-output`.

So the failure path reads the explanation back out of the retained files:

- For QE, the fenced `Error in routine ...` block becomes notes. When there
  is no block, the flagged stop lines, the output tail, and stderr become
  notes instead. The engine's input, output, and `CRASH` files are kept as
  artifacts.
- For LAMMPS, the `ERROR` line(s) become notes, with one line of preceding
  context, which is the echoed command that died or the last thermo row
  before a blow-up. The input, log, and data files are kept.

A LAMMPS potential file that cannot be opened, captured from a real run:

<!-- no-verify -->
```text
notes:
- relax failed after 0 completed step(s)
- engine error (log_lammps00000113gc6gof): ERROR on proc 0: cannot open
  eam potential file Cu_u3.eam: No such file or directory
  (src/src/potential_file_reader.cpp:58)
- engine log context (log_lammps00000113gc6gof): pair_coeff 1 1 Cu_u3.eam
```

And a missing pseudopotential file, captured from a real `pw.x` run, with
paths shortened:

<!-- no-verify -->
```text
notes:
- relax failed after 0 completed step(s)
- engine error (espresso.pwo): Error in routine readpp (1):
  file /opt/pseudos/sssp/DoesNotExist.UPF not found
- engine stderr tail (espresso.err): STOP 1
- engine files kept as artifacts: 'si-failed.pwi', 'si-failed.pwo', 'si-failed.err'
```

And the classic case, an SCF that cannot converge (here capped at one
iteration), which QE reports without a fenced error block:

<!-- no-verify -->
```text
- engine output flagged (espresso.pwo): convergence NOT achieved after   1 iterations: stopping
```

"Exit status 2" invites a blind retry, while "smearing is needed" or
"convergence NOT achieved" is a correction an agent can actually compute.
The kept `si-failed.pwi` is the exact input that crashed, and you can
reproduce it outside SLAB with nothing but `pw.x`.

## Checks as correction inputs

A run that completes but fails [verification](verification.md) is the other
debugging case. Check results store the `observed` and `expected` values
their assertions compared, which are the numbers a correction is computed
from:

```python
from foundation import check, converged

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

"fmax was 0.062 against 0.05" is enough to compute the fix, such as a rerun
with more steps or a tighter optimizer, without re-reading the workflow.

!!! note
    Failure diagnostics self-clean. Failed runs sit in quarantine under the
    default 30-day TTL, so kept evidence (partial trajectories included)
    survives long enough to be debugged and then
    [expires with the run](lifecycle-and-retention.md). Nothing accumulates
    forever unless you promote it.

## The surfaces

`foundation show` renders each traceback under its owner. The run's `failure`
prints at run level unless a failed task carries the same exception, which
is the usual case because the exception propagated, and then it renders
once, under that task. For the failed relax run above:

<!-- no-verify -->
```text
$ foundation show 01m0m5gz
run 01m0m5gza60x903wjk1dpkg1g4  cu-relax
  state:   quarantined    status: failed
  error:   RuntimeError: SCF diverged
  created: 2026-08-22T07:21:23.014396+00:00
  intent:  probe rattled Cu
  tasks:
    1. relax  failed  0.009s  error: RuntimeError: SCF diverged
       Traceback (most recent call last):
         File ".../foundation/tracing.py", line 223, in _traced_call
           result = f(*args, **kwargs)
         File ".../foundation/tasks.py", line 137, in relax
           converged = bool(optimizer.run(fmax=fmax, steps=steps))
         File ".../ase/optimize/optimize.py", line 506, in run
           return Dynamics.run(self, steps=steps)
         [... 2 frame(s) elided ...]
         File ".../ase/calculators/calculator.py", line 517, in get_property
           self.calculate(atoms, [name], system_changes)
       RuntimeError: SCF diverged
       relax failed after 3 completed step(s); trajectory has 3 frame(s),
       last frame: E=-0.006493 eV, max|F|=0.4410 eV/Å; partial trajectory
       kept as artifact 'cu-failed.traj'
  artifacts:
    cu-failed.traj  intermediate  2851B  bytes  2234b81f42cf
```

`foundation show <id> --json` emits the same details in machine-readable form,
with `failure` keys on the run and on each task, and `observed` and
`expected` on each check. Agents get identical structures without the CLI,
because the MCP `show_run` tool returns this JSON, and `launch_workflow`
returns the `failure` record directly in its result when a launched script
fails. See [Agents over MCP](agents-mcp.md). The philosophy behind the whole
arrangement is argued in [Architecture](../architecture.md).
