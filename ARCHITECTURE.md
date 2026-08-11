# SLAB architecture

This document explains the three load-bearing ideas — the lifecycle model,
tiered provenance, and the promotion-over-deletion asymmetry — and then walks
the layers that implement them. It is written to be read by both humans and
LLM agents; agents are SLAB's primary user, and that assumption shapes almost
every interface decision recorded here.

## 1. The problem being solved

Materials workflow engines accumulated two structural debts.

**Provenance totality.** The classic design records every intermediate of
every run as an immutable node in a provenance graph. The intention is noble —
perfect reproducibility — but the consequence is that storage fills with
retrieved wavefunctions, optimization trajectories, and charge densities that
no one will ever read, attached to runs that failed or were superseded within
the hour. Because the graph is immutable and referentially entangled, deleting
anything feels like surgery, so nobody does. The archive's signal-to-noise
ratio decays monotonically.

**Declaration-time epistemics.** The same engines ask the user to declare, at
submission time, whether a run is "production" — worth recording — or scratch.
But whether a run was worth recording is precisely what you learn at
*completion* time: did it converge, are the forces sane, is the energy
plausible? Asking the question before the information exists guarantees it is
answered wrong. In practice everyone runs everything in the production
profile, and the archive becomes the archaeology of failures.

**Ceremony.** Declarative process classes (`spec.input`/`spec.outline`
WorkChains) turn a five-line calculation into a page of framework-shaped code.
For humans this is a readability tax. For LLM agents — which generate the
workflow, read it back while debugging, and pay per token — it is a direct
operating cost and a rich source of error.

SLAB's position: record cheaply, decide late, keep deliberately.

## 2. The lifecycle model

Every run moves through a small state machine:

```
quarantined ──checks pass──▶ verified ──promote──▶ promoted ──archive──▶ archived
    │  │                        │
    │  └────────force-promote───┘ (recorded as forced)
    │ ttl                       │ ttl
    ▼                           ▼
 expired ◀──────────────────────┘
```

- **quarantined** — every run is born here. Ephemeral: a TTL is ticking.
- **verified** — the run's declared checks all passed. Still ephemeral;
  verification confers *eligibility*, not permanence.
- **promoted** — somebody (human or agent) decided this run matters. Permanent.
- **archived** — promoted data moved to cold status. Terminal.
- **expired** — the ephemeral window lapsed without promotion. Terminal.

Three properties are enforced structurally rather than by convention:

1. **Promoted data cannot expire.** There is no `promoted → expired`
   transition in the relation, `force=True` does not create one, and a
   retention policy that attaches a TTL to `promoted` or `archived` fails
   validation. You cannot configure your way into deleting promoted data by
   accident.
2. **`force` unlocks exactly one edge** — `quarantined → promoted`, the human
   override for wrong or absent checks — and its use is recorded on the
   transition (`forced=True`). The tests assert that force changes the
   legality of no other pair.
3. **Terminal states have no exits.**

Orthogonal to the lifecycle, each run has an **execution status**
(`pending → running → completed | failed`). The separation matters: a run that
crashes needs no special lifecycle handling. It stays quarantined, fails its
way to the TTL, and disappears. Failure hygiene is the default outcome, not a
cleanup chore. (A `pending → completed` edge exists for cache-served runs that
finish without ever starting.)

Every transition records *who* (`actor`), *why* (`reason`), and *when* — the
lifecycle is also the beginning of narrative provenance. Runs additionally
carry a free-text `intent` because capturing the goal of a run costs almost
nothing and is the piece of provenance existing tools capture worst.

Mechanics worth knowing:

- `state_entered_at` is stamped on every transition; TTLs anchor to it, so a
  run's retention clock restarts when its state changes (30 days in
  quarantine, then a fresh 90 in verified).
- Transitions execute inside SQLite `BEGIN IMMEDIATE` transactions and are
  revalidated against the state read *inside* the transaction; racing writers
  serialize, and the loser gets an error naming the actual current state.
- `transition(..., expected="quarantined")` provides compare-and-swap. The TTL
  sweep uses it so a run promoted mid-sweep can never be expired from stale
  information. The sweep also skips runs whose status is `running` — but only
  a run's own process ever advances its status, so a hard kill (SIGKILL, OOM,
  power loss) would leave a run protected forever. `expire --include-running`
  is the explicit recovery: when the operator knows those processes are dead,
  overdue running runs are marked `failed` (with an explanatory error) and
  then expired, so the documented age-out holds even for crashed hosts.

## 3. Promotion over deletion: the asymmetry argument

Both SLAB and the totality model face the same two possible mistakes:

- keep something worthless (cost: storage, attention, archive noise);
- lose something valuable (cost: a recomputation — or, in the worst designs,
  irreplaceable data).

The totality model makes *keeping* the default and requires an act of deletion
to correct it. SLAB makes *expiring* the default and requires an act of
promotion to correct it. The asymmetry is justified by three observations.

**The decision happens where the information is.** Promotion is a
completion-time act, taken while looking at converged forces and sane
energies. Deletion in a keep-everything system is a *later* act, taken months
after context has evaporated, by someone (often nobody) who must reconstruct
whether anything references this WAVECAR. Decisions made with information beat
decisions made without it.

**The failure modes are not symmetric.** Keeping junk compounds silently — the
marginal wavefunction costs nothing today and the archive is unusable in two
years. Losing value in SLAB is bounded and noisy: quarantine defaults give a
30-day window to notice; verification flags what is *eligible*; and because
every expired run keeps its hashes and recipes, the loss is never
irreplaceable data, only the compute to regenerate it (see §4).

**Curation must be the path of least resistance.** In a keep-everything
system, hygiene requires ongoing effort, so entropy wins. In SLAB, hygiene is
the automatic outcome and *keeping* is the deliberate one-command act
(`slab promote <id> --reason ...`), performed exactly when the user knows why.
The archive converges toward being a set of things someone chose, each
carrying the reason it was chosen.

Deletion of promoted data is intentionally hard (there is no CLI verb for it);
expiry of unpromoted data is intentionally silent. Both halves are the same
design decision.

## 4. Tiered provenance

The provenance guarantee is **total for everything promoted** — SLAB
selectively *keeps bytes*, it never selectively *records*.

Artifact bytes live in a content-addressed store (CAS): SHA-256, stored once
at `cas/ab/cd/<hash>`, written via temp-file + atomic rename, read-only on
disk. References live in the run database as `ArtifactRef` rows — run, name,
role, hash, size, recipe. The two lifetimes are independent, which is the
entire trick: **`discard(hash)` deletes bytes; every reference, hash, and
recipe survives.** An expired run remains a complete, queryable skeleton —
what ran, with what inputs, what it produced, and how to remake it.

Retention tiers on **artifact role**:

| role | meaning | promoted-run default |
| --- | --- | --- |
| `terminal` | declared final output (`run.keep(...)`) | bytes kept |
| `input` | recompute root: data that entered from outside | bytes kept |
| `intermediate` | reproducible byproduct (trajectories, scratch) | hash + recipe only |

Everything a traced task touches is classified automatically: outputs are
`intermediate`; inputs are `intermediate` if some task in the same run
produced them, else they are `input` roots. The roots rule exists because
"recompute on demand" is otherwise a dangling promise — a recipe whose inputs
were themselves discarded reconstructs nothing. With roots retained, any
dropped intermediate is reachable by replaying recipes forward from stored
bytes.

Garbage collection is a set computation, not a policy guess: for every
reference in every run, ask that run's state rule whether the referenced
bytes are demanded; drop the blobs demanded by no one. Consequences that fall
out for free:

- A hash shared between a promoted run's terminal and an expired run's
  intermediate survives — any demand keeps the blob, and content addressing
  already deduplicated it.
- Unreferenced blobs (a `put` that never became a reference) are *reported as
  orphans, never deleted*.
- Blobs a rule demands but that are absent are reported as `missing`:
  out-of-band deletion becomes visible instead of silent.
- In-flight work is protected by construction, not by luck: the tracer
  commits a provisional task record (status `running`, inputs included)
  *before* executing, so a long-running task's input references are visible
  to a concurrent gc even when the same bytes are also referenced by an
  expired run (content addressing deduplicates across runs — orphan
  protection alone would not see the collision). The remaining unprotected
  window is the milliseconds between storing a finished task's outputs and
  committing them, not the task's wall time.
- Input classification is order-aware: a value counts as `intermediate` only
  if an *earlier* task in the run produced it. Without that ordering, a
  fixed-point task (an idempotent relax or canonicalize whose output bytes
  equal its input bytes) would launder the run's external input into a
  droppable intermediate and gc would destroy the recompute root, leaving a
  circular recipe.

Housekeeping is deliberately two-phase — `expire` changes states (cheap,
reviewable), `gc` deletes bytes (not reversible). Policies are plain data:

```json
{"quarantined": {"ttl_days": 30}, "promoted": {"keep": ["terminal", "input"]}}
```

## 5. Define-by-run tracing

The workflow surface is ordinary imperative Python. `@task` wraps a plain
function; outside a run context it *is* the plain function. Inside
`Workspace.start_run(...)`:

1. Bound arguments (defaults applied) are serialized, hashed, and **stored**
   (inputs are part of the recipe — §4).
2. A cache key is fingerprinted from the function's identity and environment:
   module + qualname, the source hash, a *bytecode* hash (constants and
   referenced names included — REPL- and `exec`-defined functions with no
   retrievable source still get distinct identities), the fingerprint of
   every **closure cell** value (two functions from the same factory with
   different bound parameters are different computations), resolved engine
   versions, and the input hashes. A completed task with the same key whose
   output bytes still exist short-circuits execution — rerun a crashed script
   and finished work is skipped, visibly (`cache_hit=True` rows are
   recorded). Bytes discarded by retention mean an honest cache miss and
   recomputation. Failed tasks never populate the cache, and a closure cell
   that cannot be serialized makes the call *uncacheable* — computed honestly
   every time — rather than a collision risk. The key's honest boundary:
   globals are not fingerprinted. A task whose behavior depends on module
   state mutated between calls can be served a stale cached result — treat
   globals as constants, or pass them as arguments.
3. Otherwise the function runs — after a provisional `TaskRecord` (status
   `running`) is committed, so retention sees the work in flight (§4). The
   return value is stored (exact tuples element-wise, so
   `atoms, info = relax(...)` leaves per-value hashes; tuple subclasses like
   NamedTuple are stored whole so cache restores preserve their type), and
   the record is finalized with the full recipe: inputs, code version, engine
   versions (`@task(engines=("mace-torch",))` pins installed versions into
   both recipe and cache key), python/slab versions, and human-readable
   parameters.

**The DAG is derived, not declared.** Task B consuming task A's output is
visible because B's input hash equals A's output hash. There is no graph API
to learn and no way to declare the graph wrong. The honest limitation: hash
equality proves *equal content*, not *causal identity* — two tasks that
independently produce identical bytes are indistinguishable from a data-flow
edge. For workflow-scale data (structures, parameter dicts) this is harmless
in practice and the trade — zero ceremony — is the point. (This is the
PyTorch-vs-TF1 lesson applied to provenance: legibility to humans is a proxy
for token-efficiency for agents.)

Serialization is tiered like everything else: JSON-faithful values get
canonical JSON (sorted keys — equal dicts hash equally), `bytes` pass through
raw, everything else is pickled; a 2-byte tag makes each blob self-describing.
Encoding instability can only cause a spurious cache *miss*, never a wrong
cache hit — within the key's stated boundary above, the failure direction is
chosen, not accidental.

## 6. Verification hooks as the contract

Agents need verification more than they need audit trails: an agent must know
*whether to trust a result* before deciding what to do next.

Checks are zero-argument closures declared in the workflow itself:

```python
@check
def forces_converged():
    return converged(info["fmax"], below=0.05, label="fmax")
```

The vocabulary (`converged`, `within_bounds`, `finite`, `units`) returns
`Assertion` *data* — never exceptions — and degrades to failed assertions on
NaN/None/garbage, because a check that crashes on the pathology it exists to
catch is useless. Plain `assert` statements also work (AssertionError → failed
result with its message; clean return → pass); a check that raises anything
else becomes a failed result of kind `error`. `units` is deliberately just
annotation comparison — it catches "engine returned kcal/mol, workflow assumed
eV" exactly when producers declare units, without importing a unit system
(physics remains a non-goal).

At completion the runtime evaluates every registered check, stores each result
on the run, and gates `quarantined → verified` on *all passed and at least one
exists*. **A run with no checks stays quarantined**: verification is earned,
never defaulted. It can still be promoted — via `force=True`, which is
recorded — so the escape hatch exists but leaves a trace. If the script
raises, the run is marked `failed`, checks are skipped, and the exception
propagates unchanged.

### 6a. Failure as evidence

The verification story covers "can I trust this result?"; its complement is
"why is there no result?". Legacy engines answer with predefined error
*protocols* — AiiDA exit codes routed to restart handlers that apply fixed
corrections. SLAB's primary user can improvise a correction specific to the
material system and the actual failure — but only from evidence. So the
failure surface delivers evidence and stops there:

- **Structured failure records.** A failed run or task stores, next to its
  one-line `error`, a `failure` record built by `slab.errors.failure_record`:
  exception type, message, and traceback with chained causes. The record is
  deliberately token-bounded — messages clip at 2 kchars, deep frame runs keep
  the entry point and the failure site with the middle elided, the whole text
  caps at 10 kchars — because a 50-frame torch traceback delivered verbatim
  buys nothing an agent can act on.
- **Diagnostics ride the exception.** Tasks annotate failures with plain
  `Exception.add_note` (PEP 678) — no SLAB API to learn. `relax` notes the
  completed step count and the last trajectory frame's energy and residual
  force; the notes land both inside the traceback and as a separate `notes`
  list, so an agent can read the curated summary without parsing anything.
- **Scratch data survives the crash.** A mid-optimization failure keeps the
  partial trajectory as an intermediate artifact (`relax-failed.traj`) —
  the object that distinguishes "atoms flew apart" from "slow oscillation
  near convergence". Retention makes this free: failed runs sit in
  quarantine with a TTL, so failure diagnostics self-clean, where AiiDA's
  retrieved-files-of-failed-calcs accumulate forever.
- **Checks report their numbers.** `CheckResult` always stored
  `observed`/`expected`; `run_details` now surfaces them over the CLI's
  `--json` and MCP, so "fmax 0.062 vs 0.05" is data, not prose.
- **Tiered delivery.** `slab list` shows one line per run; the full evidence
  is fetched per-run by `show`. Best-effort capture never masks the original
  exception: if keeping diagnostics itself fails, the failure record says so
  and the real error still propagates.

## 7. The layers

```
CLI (typer)          MCP server (stdio)      ← two skins, one behavior
         \                /
          slab._ops                          ← shared operations
              |
   Workspace / ActiveRun / @task / @check     ← runtime (contextvar-scoped)
              |
   RunStore protocol ──── SQLiteRunStore      ← metadata: runs, transitions,
              |                                 tasks, checks, artifact refs
   ArtifactStore (CAS)                        ← bytes, content-addressed
              |
   backends.get_calculator("mace"|"emt")      ← ASE Calculator seam
```

- **Storage.** One SQLite file (WAL, `BEGIN IMMEDIATE` writes, in-transaction
  revalidation; tested against racing threads and stale cross-process
  handles) plus one CAS directory. A workspace is a directory; there is no
  daemon and nothing to configure. `RunStore` is a protocol — the Postgres
  seam — and schema versions are tracked via `PRAGMA user_version`.
- **Engines.** `slab.backends` maps names to ASE calculators from two
  sources: built-ins (`mace` in-process, `rootstock` cluster-served, ASE's
  toys) and the cluster engine registry (§7a). SLAB implements no physics;
  adding LAMMPS or VASP means adding a registry entry, and nothing in
  tracing, lifecycle, or retention changes. Heavy imports (ASE, torch) are
  quarantined behind `slab.tasks`/`slab.backends` so the core package and CLI
  stay import-light.
- **Local-first execution.** The MVP runs tasks in-process. Checkpointing and
  restart come from the cache (§5), not from an executor: rerunning a script
  *is* the resume mechanism. A SLURM executor can slot behind `@task` later
  without changing user code.

### 7a. Engines on clusters: the rootstock pattern, generalized

`Garden-AI/rootstock <https://github.com/Garden-AI/rootstock>`_ solved
"many MLIPs, conflicting environments, shared HPC clusters" with a shape SLAB
adopts twice — once by delegation, once by generalization.

**Delegation: rootstock serves checkpoint ids silently.** On a cluster with
a rootstock install, any canonical checkpoint id is directly an engine name:
`engine="mace-mp-0-medium"` resolves through rootstock (SLAB asks the install
whether some environment declares the id — the same AST-based, no-import
resolution rootstock itself uses) and forwards to
`rootstock.RootstockCalculator`: the MLIP runs in a maintainer-prebuilt,
verified environment in a worker subprocess (i-PI protocol over a Unix
socket), and the user's environment carries only a thin client. SLAB does not
reimplement any of that machinery — checkpoint→environment resolution,
worker lifecycle, and cache redirection are rootstock's job; SLAB's job is
that the engine name and options are traced inputs, so checkpoint identity is
part of the cache key and the recipe for free, and `relax` closes the worker
(its `close()`) in a `finally`. Resolution order across all sources is
built-ins → registry → checkpoint ids: a curated registry alias beats bare
resolution, and a name that is none of the three fails with an error that
says which sources were consulted. The explicit `engine="rootstock"` form
remains for what silent mode cannot express (`:custom` checkpoints with user
weights).

**Generalization: the engine registry** (`slab.engines`). Four rootstock
ideas, applied to *all* engines:

1. *The client is only a bootstrap.* Rootstock's baked-in table maps cluster
   name → install root and nothing else; everything about an install is
   declared by the install (`{root}/layout.json`), because tables baked into
   pinned clients go stale. SLAB goes one step further: the client bakes in
   nothing — it discovers a registry *file* (explicit path >
   `$SLAB_ENGINES` > `~/.config/slab/engines.json`) and the file declares
   everything: how each engine builds (a dotted path to an ASE calculator),
   its default options, the environment variables the code needs, the
   maintainer's declared version, and a verification probe.
2. *Canonical names, capability resolution.* Workflow code addresses engines
   by name (`qe`, `lammps`, `mace-mp`); the registry maps names to concrete
   builds on this cluster. The same script runs on any cluster declaring the
   names it uses. Collisions with built-in names are refused at validation —
   ambiguity is never resolved by precedence order.
3. *Maintainer-verified installs.* `slab engines verify` runs each entry's
   declared probe (import-checks entries without one); users trust names that
   verify. Rootstock's richer per-(checkpoint, cluster) verification state —
   `verified_at > built_at` freshness, nightly smoke jobs — is the natural
   next step and deliberately not rebuilt here yet.
4. *An explicit layout contract.* The registry carries `layout_version`; a
   client that meets a newer version refuses loudly instead of misreading —
   the same rule the run database applies via `PRAGMA user_version`.

Declared versions flow into provenance *and* into cache identity: `relax`
records `engine_source`/`engine_version` in its info dict, and its
`cache_extra` hook (a `@task` option: a callable over the bound arguments
whose result is folded into both the cache key and the recipe) resolves the
engine's registry identity at call time — a version bump in the registry is
a cache miss, never a stale hit. This closes the gap `engines=` pinning
cannot see: registry engines are not pip distributions.

The trust model is inherited from rootstock and stated rather than implied:
registry entries execute maintainer-declared code and environment as the
calling user. SLAB isolates configuration, not privilege; trusting a
cluster's `engines.json` is trusting its module farm.

## 8. Agent-native decisions, collected

- **Run ids are ULIDs** (26 chars, time-ordered) and every surface accepts
  unique prefixes, git-style — short strings for agents to echo around.
- **Error messages are interface**, written to be actionable verbatim:
  `illegal lifecycle transition: quarantined -> promoted (promoting an
  unverified run requires force=True)`, with a `force_would_allow` attribute
  so an agent can distinguish "retry with force" from "impossible".
- **The MCP server is thin by construction** — seven tools over the same
  `_ops` functions the CLI calls, so agent and human observations cannot
  drift. Workflow script output is captured into results because stdout is
  the protocol channel.
- **Intent is captured everywhere** it can be: run intent, transition
  reasons, promotion reasons. Text is cheap; archaeology is not.
- **Docstring examples are doctests**, executed on every test run — the
  documentation an agent reads cannot silently diverge from behavior.
