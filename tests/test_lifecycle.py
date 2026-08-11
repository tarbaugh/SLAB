"""Exhaustive tests of the lifecycle state machine.

The LEGAL/FORCE_ONLY sets below are written out independently of the
implementation on purpose: they are the spec. Any change to the transition
relation must be made in both places, deliberately.
"""

import pytest

from slab import (
    ExecutionStatus,
    IllegalStatusChangeError,
    IllegalTransitionError,
    LifecycleState,
    can_transition,
    is_terminal,
    requires_force,
    validate_status_change,
    validate_transition,
)

Q = LifecycleState.QUARANTINED
V = LifecycleState.VERIFIED
P = LifecycleState.PROMOTED
A = LifecycleState.ARCHIVED
E = LifecycleState.EXPIRED

STATES = list(LifecycleState)

# The spec: transitions legal without force...
LEGAL = {(Q, V), (Q, E), (V, P), (V, E), (P, A)}
# ...and the single transition force unlocks.
FORCE_ONLY = {(Q, P)}


@pytest.mark.parametrize("force", [False, True], ids=["no-force", "force"])
@pytest.mark.parametrize("to_state", STATES)
@pytest.mark.parametrize("from_state", STATES)
def test_transition_matrix(
    from_state: LifecycleState, to_state: LifecycleState, force: bool
) -> None:
    """Every (from, to, force) triple behaves exactly as the spec says — all 50."""
    pair = (from_state, to_state)
    expected = pair in LEGAL or (force and pair in FORCE_ONLY)

    assert can_transition(from_state, to_state, force=force) is expected

    if expected:
        validate_transition(from_state, to_state, force=force)  # must not raise
    else:
        with pytest.raises(IllegalTransitionError) as excinfo:
            validate_transition(from_state, to_state, force=force)
        err = excinfo.value
        assert err.from_state is from_state
        assert err.to_state is to_state
        assert err.force_would_allow is (pair in FORCE_ONLY and not force)


def test_promoted_never_expires_even_with_force() -> None:
    """The core asymmetry: promotion is permanent. No flag makes promoted data expire."""
    for from_state in (P, A):
        assert not can_transition(from_state, E, force=True)


def test_terminal_states_have_no_exits() -> None:
    for from_state in (A, E):
        assert is_terminal(from_state)
        for to_state in STATES:
            assert not can_transition(from_state, to_state, force=True)


def test_force_unlocks_exactly_one_transition() -> None:
    """force=True changes legality for (quarantined -> promoted) and nothing else."""
    changed = {
        (f, t)
        for f in STATES
        for t in STATES
        if can_transition(f, t, force=True) != can_transition(f, t, force=False)
    }
    assert changed == FORCE_ONLY


def test_requires_force() -> None:
    for f in STATES:
        for t in STATES:
            assert requires_force(f, t) is ((f, t) in FORCE_ONLY)


def test_non_terminal_states() -> None:
    for state in (Q, V, P):
        assert not is_terminal(state)


def test_error_message_suggests_force_for_unverified_promotion() -> None:
    with pytest.raises(IllegalTransitionError, match="force=True") as excinfo:
        validate_transition(Q, P)
    assert excinfo.value.force_would_allow


def test_error_message_self_transition() -> None:
    with pytest.raises(IllegalTransitionError, match="already promoted"):
        validate_transition(P, P)


def test_error_message_terminal() -> None:
    with pytest.raises(IllegalTransitionError, match="terminal"):
        validate_transition(E, V)


def test_error_message_promoted_permanence() -> None:
    with pytest.raises(IllegalTransitionError, match="permanent"):
        validate_transition(P, Q)


def test_error_message_lists_allowed_targets() -> None:
    with pytest.raises(IllegalTransitionError, match="expired, promoted"):
        validate_transition(V, A)


# -- execution status ------------------------------------------------------------------

PENDING = ExecutionStatus.PENDING
RUNNING = ExecutionStatus.RUNNING
COMPLETED = ExecutionStatus.COMPLETED
FAILED = ExecutionStatus.FAILED

STATUSES = list(ExecutionStatus)

LEGAL_STATUS = {
    (PENDING, RUNNING),
    (PENDING, COMPLETED),  # cache hit: result served without executing
    (PENDING, FAILED),  # setup failure before execution
    (RUNNING, COMPLETED),
    (RUNNING, FAILED),
}


@pytest.mark.parametrize("to_status", STATUSES)
@pytest.mark.parametrize("from_status", STATUSES)
def test_status_change_matrix(from_status: ExecutionStatus, to_status: ExecutionStatus) -> None:
    """Every (from, to) status pair behaves exactly as the spec says — all 16."""
    if (from_status, to_status) in LEGAL_STATUS:
        validate_status_change(from_status, to_status)  # must not raise
    else:
        with pytest.raises(IllegalStatusChangeError) as excinfo:
            validate_status_change(from_status, to_status)
        assert excinfo.value.from_status is from_status
        assert excinfo.value.to_status is to_status


def test_status_error_messages() -> None:
    with pytest.raises(IllegalStatusChangeError, match="already running"):
        validate_status_change(RUNNING, RUNNING)
    with pytest.raises(IllegalStatusChangeError, match="completed is final"):
        validate_status_change(COMPLETED, RUNNING)
    with pytest.raises(IllegalStatusChangeError, match="allowed from running"):
        validate_status_change(RUNNING, PENDING)
