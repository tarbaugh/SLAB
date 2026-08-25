"""Tests for retention policies, the TTL sweep, and gc.

The gc tests tell the demo's story in miniature: many runs land in quarantine,
one gets promoted, and after expiry + gc the archive holds exactly the promoted
run's terminal bytes plus hashes/recipes for everything else.
"""

from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from foundation import (
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
    # promoted keeps declared outputs and recompute roots; intermediates go hash-only
    assert DEFAULT_POLICY.promoted.keep == frozenset({ArtifactRole.TERMINAL, ArtifactRole.INPUT})
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


def test_expiry_never_touches_running_runs(store: SQLiteRunStore) -> None:
    run = store.create(_aged(400))
    store.set_status(run.id, "running")
    assert expire_due(store, DEFAULT_POLICY) == []
    assert store.get(run.id).state is Q


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


# -- gc over traced task data ----------------------------------------------------------


def test_gc_task_data_tiering_for_promoted_run(tmp_path: Path) -> None:
    """Promoted runs keep recompute roots and declared terminals; the middle drops."""
    from foundation import Workspace, task

    @task
    def stage_one(x: float) -> float:
        return x + 1

    @task
    def stage_two(y: float) -> float:
        return y * 2

    with Workspace(tmp_path / "ws") as ws:
        with ws.start_run(name="chain") as run:
            mid = stage_one(1.0)
            final = stage_two(mid)
            ref = run.keep("result", final)
        ws.runs.transition(run.id, "promoted", force=True)

        first, second = ws.runs.list_tasks(run.id)
        root_in = first.inputs["x"]  # entered from outside: recompute root
        mid_hash = first.outputs["return"]  # produced and consumed in-run: intermediate
        assert second.inputs["y"] == mid_hash

        report = ws.gc()
        assert mid_hash in report.dropped
        assert root_in in report.kept  # roots survive: recompute is a real promise
        assert ref.hash in report.kept  # declared terminal survives
        assert not ws.artifacts.has(mid_hash)
        assert ws.artifacts.has(root_in)


def test_gc_keeps_all_task_data_while_alive(tmp_path: Path) -> None:
    from foundation import Workspace, task

    @task
    def stage(x: float) -> float:
        return x + 1

    with Workspace(tmp_path / "ws") as ws:
        with ws.start_run() as run:
            stage(1.0)
        report = ws.gc()  # run is quarantined: everything stays
        assert report.dropped == []
        (record,) = ws.runs.list_tasks(run.id)
        assert all(ws.artifacts.has(h) for h in record.inputs.values())
        assert all(ws.artifacts.has(h) for h in record.outputs.values())


def test_gc_drops_all_task_data_of_expired_runs(tmp_path: Path) -> None:
    from foundation import Workspace, task

    @task
    def stage(x: float) -> float:
        return x + 2

    with Workspace(tmp_path / "ws") as ws:
        with ws.start_run() as run:
            stage(1.0)
        ws.runs.transition(run.id, "expired", actor="system")
        report = ws.gc()
        assert set(ws.artifacts.hashes()) == set()
        assert report.freed_bytes > 0
        assert len(ws.runs.list_tasks(run.id)) == 1  # records + hashes survive


def test_gc_fixed_point_task_keeps_external_input(tmp_path: Path) -> None:
    """A task returning its input unchanged (idempotent relax/canonicalize) must
    not launder the run's external input into a droppable intermediate."""
    from foundation import Workspace, fingerprint, task

    @task
    def identity(x: object) -> object:
        return x

    payload = {"structure": [1.0, 2.0, 3.0]}
    with Workspace(tmp_path / "ws") as ws:
        with ws.start_run() as run:
            identity(payload)
        ws.runs.transition(run.id, "promoted", force=True)
        report = ws.gc()
        root_hash = fingerprint(payload)
        assert root_hash in report.kept  # recompute root survives
        assert report.dropped == []
        assert ws.artifacts.has(root_hash)


def test_gc_during_task_execution_keeps_inflight_inputs(tmp_path: Path) -> None:
    """CAS dedup must not let gc drop bytes an executing run is using: the
    provisional task row makes in-flight inputs visible and demanded."""
    from foundation import Workspace, dumps, fingerprint, task

    root = tmp_path / "ws"
    payload = {"structure": [9, 9, 9]}
    with Workspace(root) as setup:
        with setup.start_run(name="old-attempt") as old:
            digest = setup.artifacts.put_bytes(dumps(payload))
            setup.runs.add_artifact(
                old.id, name="scratch", role="intermediate", hash=digest, size_bytes=1
            )
        setup.runs.transition(old.id, "expired", actor="system")

    seen: dict[str, object] = {}

    @task
    def slow(structure: object) -> int:
        with Workspace(root) as inner:  # a concurrent housekeeping sweep
            report = inner.gc()
            seen["dropped"] = report.dropped
            seen["still_there"] = inner.artifacts.has(fingerprint(structure))
        return 1

    with Workspace(root) as ws, ws.start_run(name="retry"):
        slow(payload)

    assert seen["still_there"] is True
    assert fingerprint(payload) not in seen["dropped"]  # type: ignore[operator]


def test_expire_include_running_recovers_hard_killed_runs(store: SQLiteRunStore) -> None:
    """A SIGKILLed process leaves its run at status 'running' forever; the
    opt-in sweep marks it failed and expires it instead of leaking it."""
    from foundation import ExecutionStatus

    dead = store.create(_aged(400))
    store.set_status(dead.id, "running")

    assert expire_due(store, DEFAULT_POLICY) == []  # protected by default
    (expired_run,) = expire_due(store, DEFAULT_POLICY, include_running=True)
    assert expired_run.id == dead.id
    assert expired_run.state is E
    assert expired_run.status is ExecutionStatus.FAILED
    assert expired_run.error is not None and "presumed dead" in expired_run.error
