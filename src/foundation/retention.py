"""Retention policies as data, and the housekeeping operations built on them.

A :class:`RetentionPolicy` attaches rules to *lifecycle states*, not to data
types — that is the design's central move. Two operations consume it:

* :func:`expire_due` — the TTL sweep. Quarantined/verified runs whose time in
  their current state exceeds the state's ``ttl_days`` transition to
  ``expired``. Automatic and silent, exactly as the lifecycle intends.
* :func:`gc` — byte reclamation. For every artifact reference, the owning
  run's state rule says whether that reference *demands bytes* (role in
  ``keep``) or is satisfied by hash+recipe alone. Bytes with no demanding
  reference anywhere are dropped from the artifact store; every reference,
  hash, and recipe survives.

A third operation sits outside the policy because the state machine has
already decided for it:

* :func:`purge_expired` — true deletion. Where gc keeps every reference,
  hash, and recipe, purge removes ``expired`` runs outright — rows and any
  bytes no surviving run references. Nothing else is reachable: the store
  refuses to delete a run in any other state.

Expiry, gc, and purge are deliberately separate phases: state changes are
cheap and reversible in review, byte deletion is not, and row deletion is
the end of traceability for what it removes.

The asymmetry is enforced structurally: a policy that puts a TTL on
``promoted`` or ``archived`` fails validation — promoted data cannot be aged
out, only explicitly archived.
"""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timedelta
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from foundation.artifacts import ArtifactStore
from foundation.errors import IllegalStatusChangeError, IllegalTransitionError
from foundation.lifecycle import ExecutionStatus, LifecycleState
from foundation.models import ArtifactRole, Run, utcnow
from foundation.store import RunStore

_ALL_ROLES = frozenset(ArtifactRole)
_NEVER_EXPIRE = (LifecycleState.PROMOTED, LifecycleState.ARCHIVED, LifecycleState.EXPIRED)


class StateRule(BaseModel):
    """Retention rule for one lifecycle state.

    Fields:
        ttl_days: Runs expire after this long in the state (``None`` = never).
            Anchored to ``run.state_entered_at``, so the clock restarts when a
            run changes state.
        keep: Artifact roles whose *bytes* must be retained while a run is in
            this state. Roles not listed are hash-only: gc may drop their bytes,
            keeping hash + recipe. (One field, not keep/hash_only pairs, so no
            contradictory configuration is expressible.)

    Examples:
        >>> rule = StateRule.model_validate({"ttl_days": 30, "keep": ["terminal"]})
        >>> sorted(role.value for role in rule.keep)
        ['terminal']
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ttl_days: float | None = Field(default=None, gt=0)
    keep: frozenset[ArtifactRole] = _ALL_ROLES


class RetentionPolicy(BaseModel):
    """Per-state retention rules. This is policy-as-data: build it from a dict.

    Defaults: quarantined runs expire after 30 days and verified runs after 90
    (only promotion confers permanence); while a run is alive (quarantined or
    verified) all its bytes are kept so it can be inspected; promoted and
    archived runs keep bytes for ``terminal`` artifacts and ``input`` roots
    (the recompute anchors) but drop ``intermediate`` bytes; expired runs keep
    no bytes. Promoted/archived rules cannot carry a TTL — validation enforces
    the promotion-is-permanent asymmetry.

    Examples:
        >>> policy = RetentionPolicy.model_validate({"quarantined": {"ttl_days": 7}})
        >>> policy.quarantined.ttl_days  # overridden
        7.0
        >>> policy.verified.ttl_days  # other states keep their defaults
        90.0
        >>> sorted(role.value for role in policy.rule_for("promoted").keep)
        ['input', 'terminal']
        >>> try:
        ...     RetentionPolicy.model_validate({"promoted": {"ttl_days": 365}})
        ... except ValueError:
        ...     print("rejected: promoted runs cannot have a TTL")
        rejected: promoted runs cannot have a TTL
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    quarantined: StateRule = StateRule(ttl_days=30)
    verified: StateRule = StateRule(ttl_days=90)
    promoted: StateRule = StateRule(keep=frozenset({ArtifactRole.TERMINAL, ArtifactRole.INPUT}))
    archived: StateRule = StateRule(keep=frozenset({ArtifactRole.TERMINAL, ArtifactRole.INPUT}))
    expired: StateRule = StateRule(keep=frozenset())
    # A blob no run references is an orphan: a task that failed before its
    # provisional row was written, or a process killed mid-put. For the first
    # day it may still belong to a run that is about to record it, so gc
    # keeps it; older than this, gc drops it. ``null`` keeps orphans forever.
    orphan_ttl_days: float | None = Field(default=1.0, ge=0)

    @model_validator(mode="after")
    def _no_ttl_on_permanent_states(self) -> RetentionPolicy:
        for state in _NEVER_EXPIRE:
            if self.rule_for(state).ttl_days is not None:
                raise ValueError(
                    f"retention policy cannot put ttl_days on {state.value!r}: "
                    f"promotion is the keep decision; promoted data never expires"
                )
        return self

    def rule_for(self, state: LifecycleState | str) -> StateRule:
        """Return the rule for a lifecycle state.

        Examples:
            >>> RetentionPolicy().rule_for("quarantined").ttl_days
            30.0
        """
        return cast(StateRule, getattr(self, LifecycleState(state).value))


DEFAULT_POLICY = RetentionPolicy()
"""The default policy described in :class:`RetentionPolicy`."""


class GcReport(BaseModel):
    """What :func:`gc` did (or, with ``dry_run``, would do).

    Fields:
        dropped: Hashes whose bytes were removed (hash+recipe survive on runs).
        kept: Hashes whose bytes are retained because some reference demands them.
        orphans: Stored hashes referenced by no run and kept, because they
            are younger than the policy's ``orphan_ttl_days`` — they may
            belong to an in-flight run that has not recorded its references
            yet.
        orphans_dropped: Unreferenced hashes older than ``orphan_ttl_days``,
            whose bytes were removed (nothing references them, so nothing
            survives to name them).
        missing: Hashes some reference *demands* but whose bytes are absent —
            normally empty; nonempty means bytes were discarded outside policy.
        freed_bytes: Total size of dropped blobs, orphans included.
    """

    model_config = ConfigDict(frozen=True)

    dropped: list[str]
    kept: list[str]
    orphans: list[str]
    orphans_dropped: list[str] = []
    missing: list[str]
    freed_bytes: int
    dry_run: bool


def expire_due(
    runs: RunStore,
    policy: RetentionPolicy = DEFAULT_POLICY,
    *,
    now: datetime | None = None,
    actor: str = "system",
    include_running: bool = False,
) -> list[Run]:
    """Expire every run that has exceeded its state's TTL; return them.

    Only quarantined/verified runs can carry TTLs (validated on the policy), so
    promoted data is structurally out of reach. Each expiry is compare-and-swap
    guarded: a run that changes state mid-sweep (e.g. gets promoted) is skipped
    silently rather than expired from stale information.

    Runs whose status is ``running`` are skipped by default — a live process
    owns them and expiry must never yank state from under it. But a hard-killed
    process (SIGKILL, OOM, power loss) leaves its run at ``running`` forever
    with no one to advance it; pass ``include_running=True`` when you know
    those processes are dead: overdue running runs are first marked ``failed``
    (with an explanatory error) and then expired.

    Examples:
        >>> from datetime import timedelta
        >>> from foundation.store import SQLiteRunStore
        >>> runs = SQLiteRunStore(":memory:")
        >>> stale = runs.create(Run(created_at=utcnow() - timedelta(days=45)))
        >>> fresh = runs.create(Run())
        >>> [r.id == stale.id for r in expire_due(runs, DEFAULT_POLICY)]
        [True]
        >>> runs.get(fresh.id).state.value
        'quarantined'
        >>> runs.close()
    """
    now = now if now is not None else utcnow()
    expired: list[Run] = []
    for state in (LifecycleState.QUARANTINED, LifecycleState.VERIFIED):
        ttl_days = policy.rule_for(state).ttl_days
        if ttl_days is None:
            continue
        cutoff = now - timedelta(days=ttl_days)
        for run in runs.list_runs(state=state):
            if run.state_entered_at > cutoff:
                continue
            if run.status is ExecutionStatus.RUNNING:
                if not include_running:
                    continue  # a live process owns this run; never yank state under it
                with suppress(IllegalStatusChangeError):  # it may finish mid-sweep; fine
                    runs.set_status(
                        run.id,
                        ExecutionStatus.FAILED,
                        error="presumed dead: expired by sweep while status was 'running'",
                    )
            try:
                expired.append(
                    runs.transition(
                        run.id,
                        LifecycleState.EXPIRED,
                        actor=actor,
                        reason=f"ttl: exceeded {ttl_days:g}d in {state.value}",
                        expected=state,
                    )
                )
            except IllegalTransitionError:
                continue  # state changed under us; never expire on stale information
    return expired


def gc(
    runs: RunStore,
    artifacts: ArtifactStore,
    policy: RetentionPolicy = DEFAULT_POLICY,
    *,
    dry_run: bool = False,
    now: datetime | None = None,
) -> GcReport:
    """Reclaim artifact bytes that no run's retention rule demands.

    Two kinds of reference are scanned: *declared artifacts* (with their
    explicit roles) and *traced task data*. Task blobs are classified
    automatically — outputs are ``intermediate``; inputs are ``intermediate``
    when produced by another task in the same run, else ``input`` (a recompute
    root). Everything a task touches is intermediate unless declared terminal.

    A blob's bytes are kept if *any* reference demands them — the same hash may
    be a promoted run's terminal output and an expired run's intermediate, and
    the promoted reference wins. Dropping is hash-and-discard: references,
    hashes, and recipes all survive in the run database.

    Examples:
        >>> import tempfile
        >>> from foundation.store import SQLiteRunStore
        >>> run_store = SQLiteRunStore(":memory:")
        >>> cas = ArtifactStore(tempfile.mkdtemp())
        >>> r = run_store.create(Run(name="si-relax"))
        >>> final = cas.put_bytes(b"final structure")
        >>> scratch = cas.put_bytes(b"wavecar")
        >>> _ = run_store.add_artifact(r.id, name="relaxed", role="terminal",
        ...                            hash=final, size_bytes=15)
        >>> _ = run_store.add_artifact(r.id, name="wave", role="intermediate",
        ...                            hash=scratch, size_bytes=7)
        >>> _ = run_store.transition(r.id, "promoted", force=True)
        >>> report = gc(run_store, cas, DEFAULT_POLICY)
        >>> report.dropped == [scratch], report.freed_bytes
        (True, 7)
        >>> cas.has(final), cas.has(scratch)
        (True, False)
        >>> len(run_store.list_artifacts(r.id))  # references always survive
        2
        >>> run_store.close()
    """
    demanded: set[str] = set()
    referenced: set[str] = set()

    def note(digest: str, role: ArtifactRole, keep: frozenset[ArtifactRole]) -> None:
        referenced.add(digest)
        if role in keep:
            demanded.add(digest)

    for run in runs.list_runs():
        keep = policy.rule_for(run.state).keep
        for ref in runs.list_artifacts(run.id):
            note(ref.hash, ref.role, keep)
        # Inputs are classified against what EARLIER tasks produced, in seq
        # order, before the consuming task's own outputs are added. Otherwise a
        # fixed-point task (output bytes == input bytes: an idempotent relax or
        # canonicalize) would launder the run's external input into an
        # "intermediate" and gc would destroy the recompute root.
        produced: set[str] = set()
        for task in runs.list_tasks(run.id):
            for digest in task.inputs.values():
                role = ArtifactRole.INTERMEDIATE if digest in produced else ArtifactRole.INPUT
                note(digest, role, keep)
            for digest in task.outputs.values():
                note(digest, ArtifactRole.INTERMEDIATE, keep)
                produced.add(digest)

    present = set(artifacts.hashes())
    to_drop = sorted((referenced - demanded) & present)
    # Orphans age from the moment their bytes landed. A run stages its inputs
    # seconds before it records them, so a day-old unreferenced blob is not
    # in flight; it is the residue of a task that failed before its row was
    # written, or of a process killed mid-put, and nothing will ever name it.
    moment = now if now is not None else utcnow()
    young: list[str] = []
    stale: list[str] = []
    for digest in sorted(present - referenced):
        age_days = (moment - artifacts.stored_at(digest)).total_seconds() / 86_400
        if policy.orphan_ttl_days is not None and age_days >= policy.orphan_ttl_days:
            stale.append(digest)
        else:
            young.append(digest)
    freed = sum(artifacts.size(digest) for digest in [*to_drop, *stale])
    if not dry_run:
        for digest in [*to_drop, *stale]:
            artifacts.discard(digest)
    return GcReport(
        dropped=to_drop,
        kept=sorted(demanded & present),
        orphans=young,
        orphans_dropped=stale,
        missing=sorted(demanded - present),
        freed_bytes=freed,
        dry_run=dry_run,
    )


class PurgeReport(BaseModel):
    """What :func:`purge_expired` did (or, with ``dry_run``, would do).

    Fields:
        deleted: Ids of the expired runs whose rows were removed.
        dropped: Hashes whose bytes went with them (no surviving reference).
        kept: Hashes the deleted runs referenced but a surviving run still does.
        freed_bytes: Total size of dropped blobs.
    """

    model_config = ConfigDict(frozen=True)

    deleted: list[str]
    dropped: list[str]
    kept: list[str]
    freed_bytes: int
    dry_run: bool


def _reachable_hashes(runs: RunStore, run: Run) -> set[str]:
    """Every blob hash a run references: declared artifacts and task data."""
    digests = {ref.hash for ref in runs.list_artifacts(run.id)}
    for task in runs.list_tasks(run.id):
        digests.update(task.inputs.values())
        digests.update(task.outputs.values())
    return digests


def purge_expired(
    runs: RunStore,
    artifacts: ArtifactStore,
    *,
    dry_run: bool = False,
) -> PurgeReport:
    """Delete expired runs outright: rows and bytes, irreversibly.

    The destructive third phase, after :func:`expire_due` (state) and
    :func:`gc` (bytes). Each expired run loses its row and, through the
    schema's cascade, its transitions, artifact references, tasks, and
    checks; then its blobs are dropped unless a surviving run references
    them. Only runs already ``expired`` are touched — the store refuses
    any other state — so promoted and archived data is structurally out
    of reach. Blobs referenced by no run at all are left alone, exactly
    as in gc: they may belong to an in-flight run that has not recorded
    its references yet.

    Examples:
        >>> import tempfile
        >>> from foundation.store import SQLiteRunStore
        >>> store = SQLiteRunStore(":memory:")
        >>> cas = ArtifactStore(tempfile.mkdtemp())
        >>> keep = store.create(Run(name="keep"))
        >>> gone = store.create(Run(name="gone"))
        >>> shared = cas.put_bytes(b"shared structure")
        >>> scratch = cas.put_bytes(b"wavecar")
        >>> _ = store.add_artifact(keep.id, name="s", role="terminal",
        ...                        hash=shared, size_bytes=16)
        >>> _ = store.add_artifact(gone.id, name="s", role="terminal",
        ...                        hash=shared, size_bytes=16)
        >>> _ = store.add_artifact(gone.id, name="w", role="intermediate",
        ...                        hash=scratch, size_bytes=7)
        >>> _ = store.transition(keep.id, "promoted", force=True)
        >>> _ = store.transition(gone.id, "expired", actor="ttl")
        >>> report = purge_expired(store, cas)
        >>> report.deleted == [gone.id], report.freed_bytes
        (True, 7)
        >>> cas.has(shared), cas.has(scratch)
        (True, False)
        >>> [r.name for r in store.list_runs()]
        ['keep']
        >>> store.close()
    """
    every = runs.list_runs()
    expired = [run for run in every if run.state is LifecycleState.EXPIRED]
    candidates: set[str] = set()
    for run in expired:
        candidates |= _reachable_hashes(runs, run)
    surviving: set[str] = set()
    for run in every:
        if run.state is not LifecycleState.EXPIRED:
            surviving |= _reachable_hashes(runs, run)
    to_drop = sorted(
        digest for digest in candidates - surviving if artifacts.has(digest)
    )
    freed = sum(artifacts.size(digest) for digest in to_drop)
    if not dry_run:
        for run in expired:
            runs.delete_run(run.id)
        for digest in to_drop:
            artifacts.discard(digest)
    return PurgeReport(
        deleted=[run.id for run in expired],
        dropped=to_drop,
        kept=sorted(candidates & surviving),
        freed_bytes=freed,
        dry_run=dry_run,
    )
