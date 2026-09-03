# Lifecycle & retention

Every SLAB run starts as temporary and becomes permanent only by an explicit action. Nothing is stored permanently by default and deleted later. This page covers the state machine, the retention policy that ages unpromoted runs out, and the two-phase `expire` + `gc` housekeeping built on both.

## The state machine

```
quarantined ──checks pass──▶ verified ──promote──▶ promoted ──archive──▶ archived
    │  │                        │
    │  └────────force-promote───┘ (recorded)
    │ ttl                       │ ttl
    ▼                           ▼
 expired ◀──────────────────────┘
```

Two rules are structural, not policy. Nothing makes promoted or archived data expire, because the transition does not exist, with or without `force`. And `expired` and `archived` are terminal states. The relation itself is queryable:

```python
from foundation import LifecycleState, can_transition

print(can_transition(LifecycleState.VERIFIED, LifecycleState.PROMOTED))
print(can_transition(LifecycleState.QUARANTINED, LifecycleState.PROMOTED))
print(can_transition(LifecycleState.QUARANTINED, LifecycleState.PROMOTED, force=True))
print(can_transition(LifecycleState.PROMOTED, LifecycleState.EXPIRED, force=True))
```

```text
True
False
True
False
```

`force=True` unlocks exactly one edge, quarantined → promoted, and records it as forced. It never unlocks expiry of promoted data.

Each run also carries an execution status that is orthogonal to the lifecycle: `pending → running → completed | failed`. A run that fails execution needs no special lifecycle handling. It stays quarantined and ages out, which makes failure diagnostics self-cleaning. See [Debugging failures](debugging-failures.md).

## Three runs, three states

One workspace, three endings. A checked run lands `verified`, an unchecked probe stays `quarantined`, and a third run is force-promoted by hand.

```python
from ase.build import bulk
from foundation import Workspace, check, converged
from foundation.tasks import relax

atoms = bulk("Cu", "fcc", a=3.58, cubic=True)
atoms.rattle(stdev=0.05, seed=42)

ws = Workspace("lifecycle-demo")

with ws.start_run(name="baseline", intent="relax the rattled cell") as a:
    relaxed, info = relax(atoms, engine="emt", fmax=0.05)

    @check
    def forces_converged():
        return converged(info["fmax"], below=0.05, label="fmax")

    a.keep("relaxed.xyz", relaxed)

with ws.start_run(name="probe", intent="eyeball the unrelaxed cell") as b:
    scratch = b.keep("notes", {"hint": "unrelaxed"}, role="intermediate")

with ws.start_run(name="rescue", intent="loose relax, promoted by hand") as c:
    relaxed_c, info_c = relax(atoms, engine="emt", fmax=0.2)
    c.keep("relaxed-loose.xyz", relaxed_c)

for run_id in (a.id, b.id, c.id):
    r = ws.runs.get(run_id)
    print(f"{r.name:9s} status={r.status.value:9s} state={r.state.value}")
```

```text
baseline  status=completed state=verified
probe     status=completed state=quarantined
rescue    status=completed state=quarantined
```

All three runs completed, but only the run whose checks passed became verified. Completion is a fact about execution, and verification is a claim about the result. The two axes never collapse into one.

Force-promotion is the recorded escape hatch for keeping a run without verification, or before it. Every transition lands in the run's history with an actor, a reason, and a `forced` flag:

```python
promoted = ws.runs.transition(
    c.id, "promoted", force=True,
    actor="agent", reason="loose relax is good enough for the survey",
)
print(promoted.state.value)
for t in ws.runs.history(c.id):
    print(f"{t.from_state.value} -> {t.to_state.value}  actor={t.actor}  "
          f"forced={t.forced}  reason={t.reason!r}")
```

```text
promoted
quarantined -> promoted  actor=agent  forced=True  reason='loose relax is good enough for the survey'
```

`forced` is true only when the transition needed force. Passing `force=True` on a normally-legal promotion records `forced=False`, so the audit trail cannot be made to look routine.

## The asymmetry is enforced, not conventional

A retention policy that puts a TTL on `promoted` is not a misconfiguration that SLAB warns about. It is unrepresentable:

```python
from foundation import RetentionPolicy

try:
    RetentionPolicy.model_validate({"promoted": {"ttl_days": 365}})
except ValueError as e:
    print(e)
```

```text
1 validation error for RetentionPolicy
  Value error, retention policy cannot put ttl_days on 'promoted': promotion is the keep decision; promoted data never expires [type=value_error, input_value={'promoted': {'ttl_days': 365}}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.13/v/value_error
```

The same rejection covers `archived` and `expired`. Together with the missing transition edge, "promoted data never expires" holds at two independent layers.

## Policy as data

A `RetentionPolicy` maps lifecycle states to rules. `ttl_days` says how long a run may sit in the state, and `keep` says which artifact roles (`terminal`, `input`, `intermediate`) must retain bytes while a run is there. The defaults:

- Quarantined runs expire after 30 days, and verified runs after 90.
- Alive runs keep all bytes, so you can inspect them.
- Promoted runs keep `terminal` outputs and `input` recompute roots, and let `intermediate` bytes go hash-only.

```python
from foundation import DEFAULT_POLICY

print(DEFAULT_POLICY.quarantined.ttl_days, DEFAULT_POLICY.verified.ttl_days)
print(sorted(role.value for role in DEFAULT_POLICY.promoted.keep))

strict = RetentionPolicy.model_validate({
    "quarantined": {"ttl_days": 7},
    "promoted": {"keep": ["terminal"]},
})
print(strict.quarantined.ttl_days, strict.verified.ttl_days)
```

```text
30.0 90.0
['input', 'terminal']
7.0 90.0
```

Unmentioned states keep their defaults, and unknown keys are rejected. Because the policy is plain data, it can also live in a file. Drop this at `<workspace>/policy.json`, or point the CLI at any path with `--policy file.json`:

```json
{
  "quarantined": {"ttl_days": 7},
  "verified":    {"ttl_days": 30},
  "promoted":    {"keep": ["terminal", "input"]}
}
```

TTLs anchor to `state_entered_at`, so the clock restarts when a run changes state. A run promoted on day 29 was never in danger, and a freshly verified run gets the full verified window no matter how long it sat in quarantine.

## Expire, then gc

Housekeeping has two deliberate phases. `expire_due` is the TTL sweep, and it is a state change only. `gc` reclaims artifact bytes that no run's retention rule demands. Both `expire_due` and its CLI counterpart `slab expire` take the sweep time as an argument (`now=` / `--older-than`), so the example can run a 40-day-later sweep without waiting:

```python
from datetime import timedelta
from foundation import utcnow

print("probe bytes on disk:", ws.artifacts.has(scratch.hash))

expired = ws.expire_due(DEFAULT_POLICY, now=utcnow() + timedelta(days=40))
for r in expired:
    print(f"expired: {r.name} ({r.state.value})")
print("baseline is still:", ws.runs.get(a.id).state.value)
print("rescue is still:  ", ws.runs.get(c.id).state.value)

report = ws.gc()
print(f"dropped={len(report.dropped)} kept={len(report.kept)} "
      f"freed_bytes={report.freed_bytes} orphans={len(report.orphans)}")
print("probe bytes on disk:", ws.artifacts.has(scratch.hash))
print("probe reference:    ", ws.runs.list_artifacts(b.id)[0].name)
ws.close()
```

```text
probe bytes on disk: True
expired: probe (expired)
baseline is still: verified
rescue is still:   promoted
dropped=3 kept=10 freed_bytes=3084 orphans=0
probe bytes on disk: False
probe reference:     notes
```

At day 40, only `probe` is past its 30-day quarantine TTL. `baseline` has 50 days of verified window left, and `rescue` is structurally out of reach. `gc` then drops three blobs: the expired run's artifact, plus the promoted run's traced intermediates. Promotion keeps terminal bytes and recompute roots, not the scratch in between. Note the last two lines. The bytes are gone (`ws.artifacts.has` flipped to `False`), but the reference (name, role, hash, recipe) survives in the run database forever. Whether bytes are currently available is a property of the store, not of the reference.

!!! note
    The exact byte count is stable here, because seeds are fixed and EMT is deterministic. Run ids and timestamps differ every run.

The two phases exist because they carry different risk. Expiry is cheap and reviewable, since it is a state change that you can list (`slab list --state expired`) and inspect before any byte is touched, and `gc --dry-run` reports what would drop without dropping it. Byte deletion is the irreversible step, so it gets its own explicit command.

Two safety valves follow the same logic:

- Blobs that no run references are orphans: the residue of a task that failed before its row was written, or of a process killed mid-write. `gc` keeps an orphan younger than the policy's `orphan_ttl_days` (default 1 day), because it may belong to a run that is about to record it, and reports it under `orphans`. Older than that, `gc` drops it and reports it under `orphans_dropped`. Set `orphan_ttl_days` to `null` in the policy file to keep orphans forever.
- Runs at status `running` are never swept by default. A hard-killed process leaves its run at `running` forever, so `expire --include-running` (or `include_running=True`) exists for when you know those processes are dead. Such runs are marked failed first, then expired.

## Promote a whole session

One conversation with the agent produces several runs. A convergence study is a smoke test and three ladders. Every run records the session that created it, so you can promote the whole conversation without collecting run ids.

Mason stamps each run it launches with the chat's id, which is the name of that chat's transcript. Type `/status` in `slab mason chat` to read it. Foundation treats the value as an opaque string, so any client can stamp its own runs. Pass `--session` to `slab run`, or export `$SLAB_SESSION` before it.

The example creates four runs in one session. Three pass their checks, and the fourth has none:

```python
import time

from foundation import Workspace, check, converged

chat_ws = Workspace("session-demo")
chat = "20260828-013504-48123"
ladders = [
    ("nb-smoke", "one balanced-protocol single point", 0.0004),
    ("nb-kmesh-ladder", "k-point ladder at 40 Ry", 0.0002),
    ("nb-cutoff-ladder", "cutoff ladder at a dense mesh", 0.0005),
]

for name, intent, residual in ladders:
    with chat_ws.start_run(name=name, intent=intent, session=chat):
        @check
        def converged_to_1_meV(residual=residual):
            return converged(residual, below=0.001, label="meV/atom")
    time.sleep(1.1)

with chat_ws.start_run(name="nb-probe", intent="eyeball one rung; no checks", session=chat):
    pass

for run in chat_ws.runs.list_runs(session=chat):
    print(f"{run.name:17s} {run.state.value:12s} session={run.session}")
chat_ws.close()
```

```text
nb-probe          quarantined  session=20260828-013504-48123
nb-cutoff-ladder  verified     session=20260828-013504-48123
nb-kmesh-ladder   verified     session=20260828-013504-48123
nb-smoke          verified     session=20260828-013504-48123
```

List the sessions in the workspace to find the id:

```bash
slab sessions
```

```text
SESSION                    RUNS   AGE  STATES
20260828-013504-48123         4    4s  3 verified, 1 quarantined
```

Then promote the session. A full id works, and so does a unique prefix:

```bash
slab promote --session 20260828 --reason "the Nb convergence study"
```

```text
  [+] 01m14tsx0f  nb-smoke             promoted checks passed
  [+] 01m14tsy31  nb-kmesh-ladder      promoted checks passed
  [+] 01m14tsz5m  nb-cutoff-ladder     promoted checks passed
  [-] 01m14tt088  nb-probe             skipped  not verified: pass --force to promote it anyway
session 20260828-013504-48123: 3 promoted, 0 already permanent, 1 skipped
```

The command reports every run it considered, and the reason lands in each promoted run's history. The exit code is 1 while any run stays behind, so a script can tell a whole promotion from a partial one.

Add `--force` to take the runs that were never verified:

```bash
slab promote --session 20260828 --force --reason "probe kept for the record"
```

```text
  [=] 01m14tsx0f  nb-smoke             already  already permanent
  [=] 01m14tsy31  nb-kmesh-ladder      already  already permanent
  [=] 01m14tsz5m  nb-cutoff-ladder     already  already permanent
  [+] 01m14tt088  nb-probe             promoted forced: never verified
session 20260828-013504-48123: 1 promoted, 3 already permanent, 0 skipped
```

Every outcome is idempotent, so rerunning the command is safe. Each run commits on its own, and a run that is already permanent is reported rather than touched.

The table states what a session promote does with each run:

| Run is | Without `--force` | With `--force` |
|---|---|---|
| verified | promoted | promoted |
| promoted or archived | reported as already permanent | reported as already permanent |
| never verified, not failed | skipped | promoted |
| failed | skipped | skipped |
| expired | skipped | skipped |

A failed run never becomes permanent this way. A bulk command must not sweep failures into permanence, so promote such a run by its own id, where you can read what failed first.

Two limits are worth knowing. Runs created before session stamping carry no session, and `slab sessions` counts them in a trailing line. Promote those by id, because `slab promote` takes several ids at once:

```bash
slab promote 01m13035ys 01m12zww7j 01m12ztp0f
```

`slab mason chat --resume` opens a new session with a new id, so the runs of a resumed chat promote as their own session.

## Fast-forward, then purge

The two phases above respect the retention policy. Two verbs override it: `slab fast-forward` and `slab purge`. Use them when a line of work is finished and you have promoted everything you intend to keep.

`slab fast-forward` moves every unpromoted run to `expired`, now. It is a state change only, like `expire`. Promoted and archived runs are not touched. Runs stuck at status `running` are skipped unless you pass `--include-running`.

**Warning: `slab purge` deletes data permanently. Run it with `--dry-run` first, and promote every run you want to keep before you run it.**

`slab purge` removes all expired data, metadata included:

- The database rows of every expired run: the run, its transitions, its artifact references, its tasks, and its checks. `slab show` can no longer answer for a purged run.
- The artifact bytes those runs referenced, unless a surviving run references the same hash. Blobs that no run references at all stay, exactly as in `gc`.
- Mason session transcripts, together with their delegation transcripts, their compaction summaries (`<stem>.compactions.md`), and their review records (`mason/reviews/`). The newest conversation and its files stay, so `slab mason chat --resume` keeps working. Pass `--all-sessions` to remove them too. The notebook and the plan live in the project directory and are never touched.
- The `.sbatch` scripts and SLURM `.out` files of finished jobs, from `<workspace>/jobs/` and from the serve directory. Jobs still in the queue keep their files, and the serve endpoint record is never touched.

Only runs in the `expired` state can be deleted. The store refuses any other state, so promoted and archived runs cannot be purged. Promote a run to keep it; the two commands together remove everything you did not.

For where these states come from in the first place, see the [Quickstart](quickstart.md). For the argument behind the design, see [Architecture](../architecture.md).
