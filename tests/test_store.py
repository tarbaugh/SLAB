import itertools
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from slab import (
    AmbiguousRunIdError,
    ArtifactExistsError,
    ArtifactNotFoundError,
    ArtifactRole,
    ExecutionStatus,
    IllegalStatusChangeError,
    IllegalTransitionError,
    LifecycleState,
    Run,
    RunExistsError,
    RunNotFoundError,
    RunStateError,
    RunStore,
    SchemaVersionError,
    SQLiteRunStore,
    StorageError,
    utcnow,
)

Q = LifecycleState.QUARANTINED
V = LifecycleState.VERIFIED
P = LifecycleState.PROMOTED
A = LifecycleState.ARCHIVED
E = LifecycleState.EXPIRED


# -- basics ----------------------------------------------------------------------------


def test_create_get_roundtrip(store: SQLiteRunStore) -> None:
    run = Run(
        name="si-relax",
        intent="baseline lattice constant",
        meta={"engine": "mace", "n_atoms": 64, "nested": {"k": [1, 2, 3]}},
    )
    store.create(run)
    loaded = store.get(run.id)
    assert loaded == run
    assert loaded.created_at.tzinfo is not None
    assert loaded.created_at.utcoffset() == timedelta(0)


def test_create_duplicate_id_raises(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    with pytest.raises(RunExistsError):
        store.create(Run(id=run.id))


def test_create_rejects_unserializable_meta(store: SQLiteRunStore) -> None:
    with pytest.raises(StorageError, match="JSON-serializable"):
        store.create(Run(meta={"obj": object()}))


def test_persists_across_reopen(db_path: Path) -> None:
    with SQLiteRunStore(db_path) as s1:
        run = s1.create(Run(name="survivor"))
        s1.transition(run.id, V, actor="checks")
    with SQLiteRunStore(db_path) as s2:
        loaded = s2.get(run.id)
        assert loaded.name == "survivor"
        assert loaded.state is V
        assert len(s2.history(run.id)) == 1


def test_memory_store_smoke() -> None:
    with SQLiteRunStore(":memory:") as s:
        run = s.create(Run())
        assert s.get(run.id).state is Q


def test_creates_parent_directories(tmp_path: Path) -> None:
    nested = tmp_path / "deeply" / "nested" / "runs.db"
    with SQLiteRunStore(nested) as s:
        s.create(Run())
    assert nested.exists()


def test_satisfies_runstore_protocol(store: SQLiteRunStore) -> None:
    assert isinstance(store, RunStore)


def test_repr_contains_path(store: SQLiteRunStore) -> None:
    assert "runs.db" in repr(store)


def test_operations_after_close_raise(db_path: Path) -> None:
    store = SQLiteRunStore(db_path)
    store.create(Run())
    store.close()
    with pytest.raises(sqlite3.ProgrammingError):
        store.list_runs()


# -- id resolution ---------------------------------------------------------------------


def test_get_by_unique_prefix(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    assert store.get(run.id[:8]).id == run.id
    assert store.resolve(run.id[:8]) == run.id


def test_get_unknown_id_raises(store: SQLiteRunStore) -> None:
    with pytest.raises(RunNotFoundError, match="zzznope"):
        store.get("zzznope")


def test_empty_prefix_rejected(store: SQLiteRunStore) -> None:
    store.create(Run())
    with pytest.raises(ValueError, match="non-empty"):
        store.get("")


def test_ambiguous_prefix_raises(store: SQLiteRunStore) -> None:
    store.create(Run(id="aaaa1"))
    store.create(Run(id="aaaa2"))
    with pytest.raises(AmbiguousRunIdError, match="2 matches") as excinfo:
        store.get("aaaa")
    assert sorted(excinfo.value.matches) == ["aaaa1", "aaaa2"]


def test_ambiguous_prefix_caps_matches(store: SQLiteRunStore) -> None:
    for i in range(7):
        store.create(Run(id=f"zzzz{i}"))
    with pytest.raises(AmbiguousRunIdError, match="6 or more") as excinfo:
        store.get("zzzz")
    assert len(excinfo.value.matches) == 6
    assert "..." in str(excinfo.value)


def test_exact_match_beats_prefix(store: SQLiteRunStore) -> None:
    store.create(Run(id="bbbb"))
    store.create(Run(id="bbbb1"))
    assert store.get("bbbb").id == "bbbb"


def test_like_wildcards_are_escaped(store: SQLiteRunStore) -> None:
    store.create(Run(id="cccc1"))
    with pytest.raises(RunNotFoundError):
        store.get("%")  # would match everything if % were not escaped
    with pytest.raises(RunNotFoundError):
        store.get("cccc_")  # would match cccc1 if _ were not escaped


# -- listing ---------------------------------------------------------------------------


def _run_at(minutes_ago: int, **kwargs: object) -> Run:
    return Run(created_at=utcnow() - timedelta(minutes=minutes_ago), **kwargs)  # type: ignore[arg-type]


def test_list_runs_newest_first(store: SQLiteRunStore) -> None:
    store.create(_run_at(3, name="oldest"))
    store.create(_run_at(1, name="newest"))
    store.create(_run_at(2, name="middle"))
    assert [r.name for r in store.list_runs()] == ["newest", "middle", "oldest"]


def test_list_runs_filter_by_state(store: SQLiteRunStore) -> None:
    kept = store.create(Run(name="kept"))
    store.create(Run(name="junk"))
    store.transition(kept.id, V)
    verified = store.list_runs(state=V)
    assert [r.name for r in verified] == ["kept"]
    assert [r.name for r in store.list_runs(state="quarantined")] == ["junk"]


def test_list_runs_filter_by_status(store: SQLiteRunStore) -> None:
    active = store.create(Run(name="active"))
    store.create(Run(name="idle"))
    store.set_status(active.id, ExecutionStatus.RUNNING)
    assert [r.name for r in store.list_runs(status="running")] == ["active"]


def test_list_runs_combined_filters_and_limit(store: SQLiteRunStore) -> None:
    for i in range(5):
        store.create(_run_at(i, name=f"r{i}"))
    got = store.list_runs(state=Q, status=ExecutionStatus.PENDING, limit=2)
    assert [r.name for r in got] == ["r0", "r1"]
    assert store.list_runs(limit=0) == []


def test_list_runs_rejects_negative_limit(store: SQLiteRunStore) -> None:
    with pytest.raises(ValueError, match="limit"):
        store.list_runs(limit=-1)


def test_list_runs_rejects_bogus_state(store: SQLiteRunStore) -> None:
    with pytest.raises(ValueError, match="LifecycleState"):
        store.list_runs(state="bogus")


def test_empty_store_lists_nothing(store: SQLiteRunStore) -> None:
    assert store.list_runs() == []


# -- transitions -----------------------------------------------------------------------


def test_transition_updates_state_and_records_history(store: SQLiteRunStore) -> None:
    run = store.create(_run_at(10))
    updated = store.transition(run.id, V, actor="checks", reason="fmax=0.03 < 0.05")
    assert updated.state is V
    assert updated.updated_at > updated.created_at

    (t,) = store.history(run.id)
    assert t.run_id == run.id
    assert t.from_state is Q
    assert t.to_state is V
    assert t.actor == "checks"
    assert t.reason == "fmax=0.03 < 0.05"
    assert t.forced is False
    assert t.at.tzinfo is not None


def test_transition_accepts_string_state(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    assert store.transition(run.id, "verified").state is V


def test_transition_rejects_bogus_state_string(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    with pytest.raises(ValueError, match="LifecycleState"):
        store.transition(run.id, "shipped")


def test_transition_accepts_prefix(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    assert store.transition(run.id[:8], V).state is V


def test_illegal_transition_leaves_no_trace(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    before = store.get(run.id)
    with pytest.raises(IllegalTransitionError):
        store.transition(run.id, A)  # quarantined -> archived is illegal
    assert store.get(run.id) == before
    assert store.history(run.id) == []


def test_force_promote_recorded_as_forced(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    updated = store.transition(run.id, P, force=True, reason="checks wrong; inspected by hand")
    assert updated.state is P
    (t,) = store.history(run.id)
    assert t.forced is True


def test_force_flag_on_normal_transition_not_recorded_as_forced(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    store.transition(run.id, V)
    store.transition(run.id, P, force=True)  # force was unnecessary here
    assert [t.forced for t in store.history(run.id)] == [False, False]


def test_promote_without_verification_requires_force(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    with pytest.raises(IllegalTransitionError) as excinfo:
        store.transition(run.id, P)
    assert excinfo.value.force_would_allow


def test_full_journey_history_chains(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    store.transition(run.id, V, actor="checks")
    store.transition(run.id, P, actor="agent", reason="best of batch")
    final = store.transition(run.id, A, actor="janitor")
    assert final.state is A

    hist = store.history(run.id)
    assert [(t.from_state, t.to_state) for t in hist] == [(Q, V), (V, P), (P, A)]
    for earlier, later in itertools.pairwise(hist):
        assert earlier.to_state is later.from_state
        assert earlier.at <= later.at


def test_expire_from_quarantine(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    assert store.transition(run.id, E, actor="system", reason="ttl").state is E


# -- concurrency -----------------------------------------------------------------------


def test_stale_handle_revalidates_against_current_state(db_path: Path) -> None:
    with SQLiteRunStore(db_path) as s1, SQLiteRunStore(db_path) as s2:
        run = s1.create(Run())
        s1.transition(run.id, V)
        # s2 never observed the change; the store reads fresh state, so this works...
        assert s2.transition(run.id, P).state is P
        # ...and s1's now-stale promotion attempt fails against the *current* state.
        with pytest.raises(IllegalTransitionError, match="already promoted"):
            s1.transition(run.id, P)


def test_concurrent_transition_single_winner(db_path: Path) -> None:
    with SQLiteRunStore(db_path) as setup:
        run = setup.create(Run())

    def attempt(i: int) -> str:
        with SQLiteRunStore(db_path) as s:
            try:
                s.transition(run.id, V, actor=f"thread-{i}")
                return "won"
            except IllegalTransitionError:
                return "lost"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt, range(8)))

    assert results.count("won") == 1
    assert results.count("lost") == 7
    with SQLiteRunStore(db_path) as s:
        assert s.get(run.id).state is V
        assert len(s.history(run.id)) == 1


# -- execution status ------------------------------------------------------------------


def test_set_status_stamps_started_and_finished(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    running = store.set_status(run.id, ExecutionStatus.RUNNING)
    assert running.status is ExecutionStatus.RUNNING
    assert running.started_at is not None
    assert running.finished_at is None

    done = store.set_status(run.id, "completed")
    assert done.status is ExecutionStatus.COMPLETED
    assert done.started_at == running.started_at
    assert done.finished_at is not None
    assert done.finished_at >= done.started_at


def test_cache_hit_finishes_without_starting(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    done = store.set_status(run.id, ExecutionStatus.COMPLETED)
    assert done.started_at is None
    assert done.finished_at is not None


def test_setup_failure_before_start(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    failed = store.set_status(run.id, ExecutionStatus.FAILED)
    assert failed.status is ExecutionStatus.FAILED
    assert failed.started_at is None
    assert failed.finished_at is not None


def test_illegal_status_change_rejected_and_unchanged(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    store.set_status(run.id, ExecutionStatus.RUNNING)
    store.set_status(run.id, ExecutionStatus.COMPLETED)
    with pytest.raises(IllegalStatusChangeError):
        store.set_status(run.id, ExecutionStatus.RUNNING)
    assert store.get(run.id).status is ExecutionStatus.COMPLETED


# -- intent ----------------------------------------------------------------------------


def test_set_intent_and_clear(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    assert store.set_intent(run.id, "probe k-mesh sensitivity").intent == (
        "probe k-mesh sensitivity"
    )
    assert store.set_intent(run.id, None).intent is None


def test_set_intent_allowed_after_promotion(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    store.transition(run.id, P, force=True)
    assert store.set_intent(run.id, "post-hoc: this was the good one").intent is not None


# -- schema versioning -----------------------------------------------------------------


def test_rejects_newer_schema_version(db_path: Path) -> None:
    SQLiteRunStore(db_path).close()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA user_version = 99")
    conn.close()
    with pytest.raises(SchemaVersionError, match="schema version 99"):
        SQLiteRunStore(db_path)


def test_reopen_same_version_is_idempotent(db_path: Path) -> None:
    with SQLiteRunStore(db_path) as s1:
        s1.create(Run())
    with SQLiteRunStore(db_path) as s2:
        assert len(s2.list_runs()) == 1


# -- fidelity --------------------------------------------------------------------------


def test_datetime_roundtrip_microseconds(store: SQLiteRunStore) -> None:
    created = datetime(2026, 8, 11, 12, 30, 45, 123456, tzinfo=UTC)
    run = store.create(Run(created_at=created))
    loaded = store.get(run.id)
    assert loaded.created_at == created
    assert loaded.updated_at == created


# -- state_entered_at ------------------------------------------------------------------


def test_state_entered_at_born_equal_to_created_at(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    assert store.get(run.id).state_entered_at == run.created_at


def test_transition_resets_state_clock(store: SQLiteRunStore) -> None:
    run = store.create(Run(created_at=utcnow() - timedelta(days=10)))
    updated = store.transition(run.id, V)
    assert updated.state_entered_at > run.state_entered_at
    assert updated.state_entered_at == updated.updated_at


def test_set_intent_and_status_do_not_touch_state_clock(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    store.set_intent(run.id, "note")
    store.set_status(run.id, ExecutionStatus.RUNNING)
    assert store.get(run.id).state_entered_at == run.state_entered_at


# -- transition expected= guard --------------------------------------------------------


def test_transition_with_matching_expected(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    assert store.transition(run.id, V, expected=Q).state is V
    assert store.transition(run.id, P, expected="verified").state is P


def test_transition_with_stale_expected_refused(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    store.transition(run.id, V)
    with pytest.raises(IllegalTransitionError, match="expected state 'quarantined'") as excinfo:
        store.transition(run.id, E, expected=Q)
    assert excinfo.value.from_state is V  # reports the actual state found
    assert store.get(run.id).state is V
    assert len(store.history(run.id)) == 1  # refused attempt left no trace


# -- artifact references ---------------------------------------------------------------

H1 = "ab" * 32
H2 = "cd" * 32


def test_add_and_get_artifact_roundtrip(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    recipe = {"task": "relax", "engine": "mace==0.3.5", "fmax": 0.05}
    ref = store.add_artifact(
        run.id, name="relaxed.xyz", role="terminal", hash=H1, size_bytes=1234, recipe=recipe
    )
    assert ref.run_id == run.id
    assert ref.role is ArtifactRole.TERMINAL
    loaded = store.get_artifact(run.id, "relaxed.xyz")
    assert loaded == ref
    assert loaded.recipe == recipe


def test_list_artifacts_ordered_and_filtered(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    store.add_artifact(run.id, name="first", role="intermediate", hash=H1, size_bytes=1)
    store.add_artifact(run.id, name="second", role="terminal", hash=H2, size_bytes=2)
    assert [a.name for a in store.list_artifacts(run.id)] == ["first", "second"]
    assert [a.name for a in store.list_artifacts(run.id, role=ArtifactRole.TERMINAL)] == ["second"]


def test_artifact_names_unique_per_run_but_not_across_runs(store: SQLiteRunStore) -> None:
    run_a = store.create(Run())
    run_b = store.create(Run())
    store.add_artifact(run_a.id, name="out", role="terminal", hash=H1, size_bytes=1)
    store.add_artifact(run_b.id, name="out", role="terminal", hash=H1, size_bytes=1)  # fine
    with pytest.raises(ArtifactExistsError, match="already has an artifact"):
        store.add_artifact(run_a.id, name="out", role="intermediate", hash=H2, size_bytes=2)


def test_add_artifact_to_terminal_run_refused(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    store.transition(run.id, E)
    with pytest.raises(RunStateError, match="expired"):
        store.add_artifact(run.id, name="late", role="terminal", hash=H1, size_bytes=1)


def test_add_artifact_validates_hash_and_role(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    with pytest.raises(ValidationError):
        store.add_artifact(run.id, name="x", role="terminal", hash="nothex", size_bytes=1)
    with pytest.raises(ValueError, match="ArtifactRole"):
        store.add_artifact(run.id, name="x", role="scratch", hash=H1, size_bytes=1)


def test_add_artifact_rejects_unserializable_recipe(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    with pytest.raises(StorageError, match="JSON-serializable"):
        store.add_artifact(
            run.id, name="x", role="terminal", hash=H1, size_bytes=1, recipe={"f": object()}
        )


def test_get_artifact_unknown_name(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    with pytest.raises(ArtifactNotFoundError, match="no artifact named"):
        store.get_artifact(run.id, "nope")


def test_artifacts_accept_run_prefix(store: SQLiteRunStore) -> None:
    run = store.create(Run())
    store.add_artifact(run.id[:8], name="out", role="terminal", hash=H1, size_bytes=1)
    assert store.get_artifact(run.id[:8], "out").hash == H1
    assert len(store.list_artifacts(run.id[:8])) == 1


def test_artifacts_persist_across_reopen(db_path: Path) -> None:
    with SQLiteRunStore(db_path) as s1:
        run = s1.create(Run())
        s1.add_artifact(run.id, name="out", role="terminal", hash=H1, size_bytes=7)
    with SQLiteRunStore(db_path) as s2:
        assert s2.get_artifact(run.id, "out").size_bytes == 7
