from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from slab import ExecutionStatus, LifecycleState, Run, Transition, utcnow


def test_run_defaults() -> None:
    run = Run()
    assert run.state is LifecycleState.QUARANTINED
    assert run.status is ExecutionStatus.PENDING
    assert run.name == ""
    assert run.intent is None
    assert run.meta == {}
    assert run.started_at is None
    assert run.finished_at is None
    assert len(run.id) == 26


def test_run_born_with_identical_timestamps() -> None:
    run = Run()
    assert run.created_at == run.updated_at
    assert run.created_at.tzinfo is not None


def test_run_explicit_created_at_propagates_to_updated_at() -> None:
    past = datetime(2026, 1, 1, tzinfo=UTC)
    run = Run(created_at=past)
    assert run.created_at == past
    assert run.updated_at == past


def test_run_explicit_timestamps_respected() -> None:
    created = datetime(2026, 1, 1, tzinfo=UTC)
    updated = created + timedelta(hours=1)
    run = Run(created_at=created, updated_at=updated)
    assert run.updated_at == updated


def test_run_is_frozen() -> None:
    run = Run()
    with pytest.raises(ValidationError):
        run.state = LifecycleState.PROMOTED  # type: ignore[misc]


def test_run_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Run(bogus="typo")  # type: ignore[call-arg]


def test_run_meta_nested() -> None:
    run = Run(meta={"engine": {"name": "mace", "version": "0.3.5"}, "n_atoms": 64})
    assert run.meta["engine"]["name"] == "mace"


def test_transition_defaults() -> None:
    t = Transition(
        run_id="abc", from_state=LifecycleState.QUARANTINED, to_state=LifecycleState.VERIFIED
    )
    assert t.actor == "user"
    assert t.reason is None
    assert t.forced is False
    assert t.at.tzinfo is not None


def test_utcnow_is_aware_utc() -> None:
    now = utcnow()
    assert now.tzinfo is UTC
