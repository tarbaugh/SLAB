# Verification checks

A run is not trustworthy by default. It earns `verified` by passing
machine-checkable assertions that you register with `@check`. This page
covers the verification contract, the assertion vocabulary, and the recorded
escape hatch for when you overrule your own checks.

## The contract

Checks evaluate once, when the `with ws.start_run(...)` block exits cleanly.
Three rules govern the outcome, and the runtime enforces all three:

1. Every check passes **and at least one check exists**. The run moves
   `quarantined -> verified`.
2. Zero checks. The run stays `quarantined`. Verification is earned, never
   defaulted. A run that asserted nothing has proven nothing.
3. The block raised. The run is marked `failed`, and checks are skipped
   entirely. There is nothing to verify. The run carries a structured
   failure record instead. See [Debugging failures](debugging-failures.md).

Here is a relaxation that earns it. Each `@check` is a zero-argument function
that closes over whatever it needs. The three below use the assertion
vocabulary (`converged`, `within_bounds`, `finite`), imported from the
package root.

```python
from ase.build import bulk
from slab import Workspace, check, converged, finite, within_bounds
from slab.tasks import relax

atoms = bulk("Cu", "fcc", a=3.58) * (2, 2, 2)
atoms.rattle(stdev=0.05, seed=42)

ws = Workspace("checks-demo")
with ws.start_run(name="cu-relax", intent="verification tutorial") as run:
    relaxed, info = relax(atoms, engine="emt", fmax=0.05)

    @check
    def forces_converged():
        return converged(info["fmax"], below=0.05, label="fmax")

    @check
    def energy_sane():
        # honest, wide bounds: catches sign errors and unit mix-ups, not noise
        return within_bounds(info["energy"], lo=-1.0, hi=1.0, label="energy")

    @check
    def positions_finite():
        return finite(relaxed.get_positions(), label="positions")

print(f"E = {info['energy']:.6f} eV  fmax = {info['fmax']:.4f}  steps = {info['steps']}")
print("state:", ws.runs.get(run.id).state.value)
```

```text
E = -0.053417 eV  fmax = 0.0388  steps = 9
state: verified
```

EMT is deterministic and the rattle seed is fixed, so these numbers reproduce
exactly.

## Reading the results

SLAB stores check results on the run as data. Each result has a name, a
kind, a pass/fail flag, a human-readable message, and the `observed` and
`expected` values the assertion compared:

```python
for r in ws.runs.list_check_results(run.id):
    print(f"{r.name}: kind={r.kind} passed={r.passed}")
    print(f"  {r.message}")
    print(f"  observed={r.observed} expected={r.expected}")
```

```text
forces_converged: kind=converged passed=True
  fmax=0.0388034 < 0.05
  observed=0.038803414059372716 expected={'below': 0.05}
energy_sane: kind=within_bounds passed=True
  energy=-0.0534174 within [-1, 1]
  observed=-0.05341742483454048 expected={'hi': 1.0, 'lo': -1.0}
positions_finite: kind=finite passed=True
  all 24 positions values finite
  observed=None expected=finite
```

`finite` elides `observed` for sequences longer than 8 elements. The message
carries the count. A whole force array stored as check metadata would help
no one.

## Plain asserts work too

A check does not have to return an `Assertion`. A raised `AssertionError`
records a failure whose message is the assert's message. A returned `None`
records a pass. A bare `bool` is also accepted. One failing check is enough
to keep the run quarantined. The passing check is still recorded, but the
gate needs all of them:

```python
with ws.start_run(name="cu-relax-strict", intent="tighter bar") as run2:
    relaxed2, info2 = relax(atoms, engine="emt", fmax=0.05)   # cache hit: same inputs

    @check
    def took_few_steps():
        assert info2["steps"] <= 2, f"needed {info2['steps']} optimizer steps, wanted <= 2"

    @check
    def forces_converged():
        return converged(info2["fmax"], below=0.05, label="fmax")

print("state:", ws.runs.get(run2.id).state.value)
for r in ws.runs.list_check_results(run2.id):
    print(f"{r.name}: passed={r.passed}  {r.message}")
```

```text
state: quarantined
took_few_steps: passed=False  needed 9 optimizer steps, wanted <= 2
forces_converged: passed=True  fmax=0.0388034 < 0.05
```

Nothing bad happens to an unverified run. It sits in quarantine under its
TTL and expires if nobody acts. That asymmetry is the point. A failed
verification costs nothing to abandon.

## Garbage in, failed assertion out

The vocabulary never raises on a malformed observed value. A check that
crashed on the NaN it was built to catch would defeat its purpose. So garbage
input produces a failed `Assertion`, which is data you can record and read,
not an exception:

```python
a = converged(float("nan"), below=0.05)
print(type(a).__name__, a.passed)
print(a.message)
print("observed:", a.observed, " expected:", a.expected)
```

```text
Assertion False
residual is not a finite number: nan
observed: nan  expected: {'below': 0.05}
```

The same holds for `None`, strings, bools, and infinities. Malformed
expectations are the opposite case. `converged(0.01, below="0.05")` is a
programming error, and it raises `TypeError` immediately.

One vocabulary function needs a disclaimer. `units(observed, expected)`
compares producer-declared unit strings. The match is exact after whitespace
stripping, and case matters. `units(info["energy_unit"], "eV")` therefore
catches the "engine returned kcal/mol, workflow assumed eV" class of error,
as long as producers record their units. It is annotation capture, not
dimensional analysis. It does not convert, derive, or reason about units.

## observed/expected is agent-facing data

The stored `observed` and `expected` values are the numbers an agent computes
a correction from. "fmax was 0.062 against 0.05" tells the agent to rerun
with more steps or to loosen the threshold. A bare "check failed" tells it
nothing. `slab show <id> --json` and the MCP `show_run` tool return both
fields, next to the structured failure records described in
[Debugging failures](debugging-failures.md):

```json
{"name": "forces_converged", "kind": "converged", "passed": false,
 "message": "fmax=0.062 !< 0.05", "observed": 0.062, "expected": {"below": 0.05}}
```

## Force-promotion, the recorded escape hatch

Sometimes you overrule your own checks. Here, the steps bar was aspirational,
and the physics is fine. `quarantined -> promoted` is legal only with
`force=True`, and the transition history records it as forced forever. It is
an escape hatch, not a bypass:

```python
promoted = ws.runs.transition(run2.id, "promoted",
                              reason="steps bar was aspirational; forces converged",
                              force=True)
print(promoted.state.value, "forced:", ws.runs.history(run2.id)[-1].forced)
ws.close()
```

```text
promoted forced: True
```

The CLI spelling is `slab promote <id> --force --reason "..."`. For where
promotion fits in the larger lifecycle (TTLs, expiry, and what promoted runs
retain), see [Architecture](../architecture.md). For the happy path from
script to promoted result, start at the [Quickstart](quickstart.md).
