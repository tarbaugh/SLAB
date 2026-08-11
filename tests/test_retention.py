"""Tests for retention policies, the TTL sweep, and gc.

The gc tests tell the demo's story in miniature: many runs land in quarantine,
one gets promoted, and after expiry + gc the archive holds exactly the promoted
run's terminal bytes plus hashes/recipes for everything else.
"""

from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from slab import (
    DEFAULT_POLICY,
    ArtifactRole,
    ArtifactStore,
    IllegalTransitionError,
    LifecycleState,
    RetentionPolicy,
    Run,
    SQLiteRunStore,
    StateRule,
    expire_due,
    gc,
    utcnow,
)

Q = LifecycleState.QUARANTINED
V = LifecycleState.VERIFIED
E = LifecycleState.EXPIRED


@pytest.fixture()
def cas(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "cas")


def _aged(days: float, **kwargs: object) -> Run:
    return Run(created_at=utcnow() - timedelta(days=days), **kwargs)  # type: ignore[arg-type]


# -- policy as data --------------------------------------------------------------------


def test_default_policy_shape() -> None:
    assert DEFAULT_POLICY.quarantined.ttl_days == 30
    assert DEFAULT_POLICY.verified.ttl_days == 90
    assert DEFAULT_POLICY.promoted.ttl_days is None
    assert DEFAULT_POLICY.quarantined.keep == frozenset(ArtifactRole)  # alive: keep all
    assert DEFAULT_POLICY.promoted.keep == frozenset({ArtifactRole.TERMINAL})
    assert DEFAULT_POLICY.expired.keep == frozenset()


def test_policy_from_spec_shaped_dict() -> None:
    policy = RetentionPolicy.model_validate(
        {"quarantined": {"ttl_days": 30}, "promoted": {"keep": ["terminal"]}}
    )
    assert policy.quarantined.ttl_days == 30
    assert policy.promoted.keep == frozenset({ArtifactRole.TERMINAL})
    assert policy.verified.ttl_days == 90  # unmentioned states keep defaults


def test_ttl_on_promoted_rejected() -> None:
    with pytest.raises(ValidationError, match="never expires"):
        RetentionPolicy.model_validate({"promoted": {"ttl_days": 365}})


def test_ttl_on_archived_and_expired_rejected() -> None:
    for state in ("archived", "expired"):
        with pytest.raises(ValidationError):
            RetentionPolicy.model_validate({state: {"ttl_days": 1}})


def test_nonpositive_ttl_rejected() -> None:
    for bad in (0, -3):
        with pytest.raises(ValidationError):
            StateRule.model_validate({"ttl_days": bad})


def test_unknown_policy_keys_rejected() -> None:
    with pytest.raises(ValidationError):
        RetentionPolicy.model_validate({"quarantined": {"ttl_dayz": 30}})


def test_rule_for_accepts_enum_and_string() -> None:
    assert DEFAULT_POLICY.rule_for(Q) is DEFAULT_POLICY.quarantined
    assert DEFAULT_POLICY.rule_for("verified") is DEFAULT_POLICY.verified


# -- expire_due ------------------------------------------------------------------------


def test_expires_overdue_quarantined_only(store: SQLiteRunStore) -> None:
    stale = store.create(_aged(45, name="stale"))
    fresh = store.create(_aged(2, name="fresh"))
    expired = expire_due(store, DEFAULT_POLICY)
    assert [r.id for r in expired] == [stale.id]
    assert store.get(stale.id).state is E
    assert store.get(fresh.id).state is Q


def test_expiry_reason_and_actor_recorded(store: SQLiteRunStore) -> None:
    run = store.create(_aged(31))
    expire_due(store, DEFAULT_POLICY)
    (t,) = store.history(run.id)
    assert t.actor == "system"
    assert t.reason == "ttl: exceeded 30d in quarantined"


def test_verified_expires_on_its_own_clock(store: SQLiteRunStore) -> None:
    run = store.create(_aged(200))
    store.transition(run.id, V)  # state_entered_at resets to now
    assert expire_due(store, DEFAULT_POLICY) == []  # 200d in quarantine is irrelevant
    future = utcnow() + timedelta(days=91)
    expired = expire_due(store, DEFAULT_POLICY, now=future)
    assert [r.id for r in expired] == [run.id]
    assert "90d in verified" in (store.history(run.id)[-1].reason or "")


def test_promoted_never_swept(store: SQLiteRunStore) -> None:
    run = store.create(_aged(400))
    store.transition(run.id, "promoted", force=True)
    assert expire_due(store, DEFAULT_POLICY, now=utcnow() + timedelta(days=10_000)) == []
    assert store.get(run.id).state.value == "promoted"


def test_ttl_none_disables_expiry(store: SQLiteRunStore) -> None:
    policy = RetentionPolicy.model_validate({"quarantined": {"ttl_days": None}})
    store.create(_aged(10_000))
    assert expire_due(store, policy) == []


def test_expiry_skips_runs_that_changed_state_mid_sweep(
    store: SQLiteRunStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store.create(_aged(45, name="a"))
    store.create(_aged(45, name="b"))
    real = store.transition
    raised = {"done": False}

    def flaky(run_id: str, to_state: object, **kwargs: object) -> Run:
        if not raised["done"]:  # first target "changes state" under the sweep
            raised["done"] = True
            raise IllegalTransitionError(V, E, detail="simulated concurrent change")
        return real(run_id, to_state, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(store, "transition", flaky)
    assert len(expire_due(store, DEFAULT_POLICY)) == 1  # one skipped, sweep continues


def test_expected_guard_used_by_sweep(store: SQLiteRunStore) -> None:
    """The compare-and-swap primitive expire_due relies on, exercised directly."""
    run = store.create(Run())
    store.transition(run.id, V)
    with pytest.raises(IllegalTransitionError, match="expected state 'quarantined'"):
        store.transition(run.id, E, expected=Q)
    assert store.get(run.id).state is V


# -- gc --------------------------------------------------------------------------------


def _add(
    store: SQLiteRunStore,
    cas: ArtifactStore,
    run: Run,
    name: str,
    role: str,
    data: bytes,
) -> str:
    digest = cas.put_bytes(data)
    store.add_artifact(run.id, name=name, role=role, hash=digest, size_bytes=len(data))
    return digest


def test_gc_promoted_keeps_terminal_drops_intermediate(
    store: SQLiteRunStore, cas: ArtifactStore
) -> None:
    run = store.create(Run())
    final = _add(store, cas, run, "relaxed.xyz", "terminal", b"final structure")
    scratch = _add(store, cas, run, "wavecar", "intermediate", b"enormous wavefunction")
    store.transition(run.id, "promoted", force=True)

    report = gc(store, cas, DEFAULT_POLICY)

    assert report.dropped == [scratch]
    assert report.kept == [final]
    assert report.freed_bytes == len(b"enormous wavefunction")
    assert cas.has(final) and not cas.has(scratch)
    # hash-and-discard: both references survive, with recipe-bearing metadata intact
    names = {a.name for a in store.list_artifacts(run.id)}
    assert names == {"relaxed.xyz", "wavecar"}


def test_gc_quarantined_runs_keep_everything(store: SQLiteRunStore, cas: ArtifactStore) -> None:
    run = store.create(Run())
    _add(store, cas, run, "out", "terminal", b"t")
    _add(store, cas, run, "tmp", "intermediate", b"i")
    report = gc(store, cas, DEFAULT_POLICY)
    assert report.dropped == []
    assert len(report.kept) == 2


def test_gc_expired_runs_drop_everything(store: SQLiteRunStore, cas: ArtifactStore) -> None:
    run = store.create(Run())
    a = _add(store, cas, run, "out", "terminal", b"tt")
    b = _add(store, cas, run, "tmp", "intermediate", b"ii")
    store.transition(run.id, E)
    report = gc(store, cas, DEFAULT_POLICY)
    assert set(report.dropped) == {a, b}
    assert not cas.has(a) and not cas.has(b)
    assert len(store.list_artifacts(run.id)) == 2  # hashes + recipes remain


def test_gc_shared_hash_kept_if_any_reference_demands_it(
    store: SQLiteRunStore, cas: ArtifactStore
) -> None:
    data = b"shared bytes"
    keeper = store.create(Run(name="keeper"))
    goner = store.create(Run(name="goner"))
    digest = _add(store, cas, keeper, "out", "terminal", data)
    assert _add(store, cas, goner, "tmp", "intermediate", data) == digest
    store.transition(keeper.id, "promoted", force=True)
    store.transition(goner.id, E)

    report = gc(store, cas, DEFAULT_POLICY)
    assert report.dropped == []
    assert report.kept == [digest]
    assert cas.has(digest)


def test_gc_orphans_reported_not_deleted(store: SQLiteRunStore, cas: ArtifactStore) -> None:
    orphan = cas.put_bytes(b"never referenced")
    report = gc(store, cas, DEFAULT_POLICY)
    assert report.orphans == [orphan]
    assert report.dropped == []
    assert cas.has(orphan)


def test_gc_reports_missing_demanded_bytes(store: SQLiteRunStore, cas: ArtifactStore) -> None:
    run = store.create(Run())
    digest = _add(store, cas, run, "out", "terminal", b"precious")
    store.transition(run.id, "promoted", force=True)
    cas.discard(digest)  # out-of-band deletion, violating the policy's guarantee
    report = gc(store, cas, DEFAULT_POLICY)
    assert report.missing == [digest]
    assert report.dropped == [] and report.kept == []


def test_gc_dry_run_deletes_nothing(store: SQLiteRunStore, cas: ArtifactStore) -> None:
    run = store.create(Run())
    scratch = _add(store, cas, run, "tmp", "intermediate", b"junk")
    store.transition(run.id, E)
    report = gc(store, cas, DEFAULT_POLICY, dry_run=True)
    assert report.dropped == [scratch]
    assert report.dry_run is True
    assert cas.has(scratch)  # still there
    real = gc(store, cas, DEFAULT_POLICY)
    assert real.dropped == [scratch]
    assert not cas.has(scratch)


def test_gc_end_to_end_demo_story(store: SQLiteRunStore, cas: ArtifactStore) -> None:
    """5 variants land in quarantine; the best is promoted; expiry + gc leave
    exactly one run's terminal bytes plus hashes for everything else."""
    runs = [store.create(_aged(45, name=f"variant-{i}")) for i in range(5)]
    terminals, intermediates = [], []
    for i, run in enumerate(runs):
        terminals.append(_add(store, cas, run, "relaxed", "terminal", b"structure %d" % i))
        intermediates.append(_add(store, cas, run, "traj", "intermediate", b"traj %d" % i))

    best = runs[2]
    store.transition(best.id, "verified", actor="checks")
    store.transition(best.id, "promoted", reason="lowest energy of batch")

    assert len(expire_due(store, DEFAULT_POLICY)) == 4
    report = gc(store, cas, DEFAULT_POLICY)

    assert set(cas.hashes()) == {terminals[2]}  # exactly the promoted terminal bytes
    assert report.freed_bytes > 0
    for run in runs:  # every run still answers "what was made, and how?"
        assert {a.name for a in store.list_artifacts(run.id)} == {"relaxed", "traj"}
