# Lifecycle & retention

Every SLAB run is born ephemeral and moves toward permanence only by explicit action — nothing is born permanent and later deleted. This page walks the state machine, the retention policy that ages unpromoted runs out, and the two-phase `expire` + `gc` housekeeping built on both.

## The state machine

```
quarantined ──checks pass──▶ verified ──promote──▶ promoted ──archive──▶ archived
    │  │                        │
    │  └────────force-promote───┘ (recorded)
    │ ttl                       │ ttl
    ▼                           ▼
 expired ◀──────────────────────┘
```

Two rules are structural, not policy: nothing makes promoted or archived data expire (the transition does not exist, with or without `force`), and `expired`/`archived` are terminal. The relation itself is queryable:

```python
from slab import LifecycleState, can_transition

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

`force=True` unlocks exactly one edge — quarantined → promoted, recorded as forced. It never unlocks expiry of promoted data.

Orthogonal to the lifecycle, each run carries an *execution status*: `pending → running → completed | failed`. A run that fails execution needs no special lifecycle handling — it stays quarantined and ages out, which is what makes failure diagnostics self-cleaning (see [Debugging failures](debugging-failures.md)).

## Three runs, three states

One workspace, three endings: a checked run that lands `verified`, an unchecked probe that stays `quarantined`, and a run we will force-promote by hand.

```python
from ase.build import bulk
from slab import Workspace, check, converged
from slab.tasks import relax

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

All three completed; only the run whose checks passed became verified. Completion is a fact about execution, verification is a claim about the result — the two axes never collapse into one.

Force-promotion is the recorded escape hatch for when you want a run kept without (or before) verification. Every transition lands in the run's history with actor, reason, and a `forced` flag:

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

`forced` is true only when the transition *needed* force — passing `force=True` on a normally-legal promotion records `forced=False`. The audit trail cannot be gamed into looking routine.

## The asymmetry is enforced, not conventional

A retention policy that puts a TTL on `promoted` is not misconfiguration to be warned about — it is unrepresentable:

```python
from slab import RetentionPolicy

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

The same rejection covers `archived` and `expired`. Between this and the missing transition edge, "promoted data never expires" holds at two independent layers.

## Policy as data

A `RetentionPolicy` maps lifecycle states to rules: `ttl_days` (how long a run may sit in the state) and `keep` (which artifact *roles* — `terminal`, `input`, `intermediate` — must retain bytes while a run is there). The defaults: quarantined expires after 30 days, verified after 90; alive runs keep all bytes so they can be inspected; promoted runs keep `terminal` outputs and `input` recompute roots but let `intermediate` bytes go hash-only.

```python
from slab import DEFAULT_POLICY

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

Unmentioned states keep their defaults, unknown keys are rejected. Because the policy is plain data, it also lives in a file — drop this at `<workspace>/policy.json`, or point the CLI at any path with `--policy file.json`:

```json
{
  "quarantined": {"ttl_days": 7},
  "verified":    {"ttl_days": 30},
  "promoted":    {"keep": ["terminal", "input"]}
}
```

TTLs anchor to `state_entered_at` — the clock restarts when a run changes state, so a run promoted on day 29 was never in danger, and a freshly verified run gets the full verified window regardless of how long it sat in quarantine.

## Expire, then gc

Housekeeping is two deliberate phases. `expire_due` is the TTL sweep — a state change only. `gc` reclaims artifact bytes that no run's retention rule demands. Both `expire_due` and its `slab expire` CLI counterpart take the sweep time as an argument (`now=` / `--older-than`), so we can test a 40-day-later sweep without waiting:

```python
from datetime import timedelta
from slab import utcnow

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

At day 40, only `probe` is past its 30-day quarantine TTL; `baseline` has 50 days of verified window left, and `rescue` is structurally out of reach. Then `gc` drops three blobs: the expired run's artifact, plus the *promoted* run's traced intermediates — promotion keeps terminal bytes and recompute roots, not the scratch in between. Note the last two lines: the bytes are gone (`ws.artifacts.has` flipped to `False`) but the reference — name, role, hash, recipe — survives in the run database forever. Whether bytes are currently available is a property of the store, not of the reference.

!!! note
    The exact byte count is stable here — seeds are fixed and EMT is deterministic — but run ids and timestamps differ every run.

Why two phases instead of one `cleanup`? Because they carry different risk. Expiry is cheap and reviewable — a state change you can list (`slab list --state expired`) and inspect before any byte is touched, and `gc --dry-run` reports what would drop without dropping it. Byte deletion is the irreversible step, so it gets its own explicit command. Two safety valves follow the same logic: blobs referenced by no run are reported as `orphans` but never deleted (they may belong to an in-flight run that has not recorded its references yet), and runs at status `running` are never swept by default — a hard-killed process leaves its run at `running` forever, so `expire --include-running` (or `include_running=True`) exists for when you know those processes are dead: such runs are marked failed first, then expired.

For where these states come from in the first place, see the [Quickstart](quickstart.md); for the argument behind the design, see [Architecture](../architecture.md).
