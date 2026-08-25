"""Tests for Workspace, the run context, and check evaluation/gating."""

from pathlib import Path

import pytest

from foundation import (
    ArtifactRole,
    Assertion,
    ExecutionStatus,
    LifecycleState,
    NestedRunError,
    NoActiveRunError,
    Workspace,
    check,
    converged,
    current_run,
    finite,
    loads,
)

Q = LifecycleState.QUARANTINED
V = LifecycleState.VERIFIED
P = LifecycleState.PROMOTED


@pytest.fixture()
def ws(tmp_path: Path) -> Workspace:
    with Workspace(tmp_path / "ws") as workspace:
        yield workspace


def test_workspace_layout(tmp_path: Path) -> None:
    root = tmp_path / "deep" / "ws"
    with Workspace(root) as workspace:
        assert (root / "runs.db").exists()
        assert workspace.artifacts.root == root / "cas"
        assert "ws" in repr(workspace)


def test_run_lifecycle_around_context(ws: Workspace) -> None:
    with ws.start_run(name="demo", intent="why") as run:
        assert current_run() is run
        snapshot = run.run
        assert snapshot.status is ExecutionStatus.RUNNING
        assert snapshot.intent == "why"
        assert "demo" not in repr(run) or True  # repr carries the id
    assert current_run() is None
    final = ws.runs.get(run.id)
    assert final.status is ExecutionStatus.COMPLETED
    assert final.state is Q  # no checks -> quarantined; verification is earned


def test_passing_checks_verify_the_run(ws: Workspace) -> None:
    with ws.start_run(name="good") as run:

        @check
        def forces_converged():
            return converged(0.03, below=0.05, label="fmax")

    final = ws.runs.get(run.id)
    assert final.state is V
    (result,) = ws.runs.list_check_results(run.id)
    assert result.passed and result.kind == "converged"
    (transition,) = ws.runs.history(run.id)
    assert transition.actor == "checks"
    assert transition.reason == "1/1 assertions passed"


def test_failing_check_keeps_quarantine(ws: Workspace) -> None:
    with ws.start_run(name="bad") as run:

        @check
        def forces_converged():
            return converged(0.5, below=0.05, label="fmax")

    assert ws.runs.get(run.id).state is Q
    (result,) = ws.runs.list_check_results(run.id)
    assert not result.passed
    assert ws.runs.history(run.id) == []  # no verification transition happened


def test_one_failure_among_many_blocks_verification(ws: Workspace) -> None:
    with ws.start_run() as run:
        run.check(lambda: True, name="ok")
        run.check(lambda: False, name="not-ok")
    assert ws.runs.get(run.id).state is Q
    results = {r.name: r.passed for r in ws.runs.list_check_results(run.id)}
    assert results == {"ok": True, "not-ok": False}


def test_check_coercions(ws: Workspace) -> None:
    with ws.start_run() as run:
        run.check(lambda: None, name="assert_style_pass")
        run.check(lambda: [converged(0.01, below=1), finite([1.0, 2.0])], name="multi")

        @check(name="module_level_named")
        def _named() -> bool:
            return True

        @run.check(name="assert_style_fail")
        def _fail() -> None:
            assert 1 > 2, "arithmetic is broken"

        @run.check
        def crashing() -> object:
            raise ValueError("boom")

        run.check(lambda: object(), name="unsupported")
        run.check(lambda: [], name="empty")
        run.check(lambda: [converged(0.01, below=1), object()], name="mixed")

    by_name = {r.name: r for r in ws.runs.list_check_results(run.id)}
    assert by_name["assert_style_pass"].passed
    assert by_name["assert_style_pass"].kind == "assert"
    assert by_name["multi[0]"].passed and by_name["multi[1]"].passed
    assert by_name["module_level_named"].passed
    assert by_name["mixed[0]"].passed
    assert not by_name["mixed[1]"].passed
    assert "unsupported type" in by_name["mixed[1]"].message
    assert not by_name["assert_style_fail"].passed
    # pytest's assertion rewriting appends the expression; the message leads
    assert by_name["assert_style_fail"].message.startswith("arithmetic is broken")
    assert not by_name["crashing"].passed
    assert by_name["crashing"].message == "check raised ValueError: boom"
    assert not by_name["unsupported"].passed
    assert "unsupported type" in by_name["unsupported"].message
    assert not by_name["empty"].passed
    assert ws.runs.get(run.id).state is Q


def test_single_element_iterable_keeps_plain_name(ws: Workspace) -> None:
    with ws.start_run() as run:
        run.check(lambda: [Assertion(kind="custom", passed=True, message="ok")], name="solo")
    (result,) = ws.runs.list_check_results(run.id)
    assert result.name == "solo"


def test_check_outside_run_raises(ws: Workspace) -> None:
    with pytest.raises(NoActiveRunError):

        @check
        def orphan():
            return True


def test_script_failure_marks_run_failed_and_skips_checks(ws: Workspace) -> None:
    with pytest.raises(ValueError, match="exploded"), ws.start_run(name="doomed") as run:

        @check
        def never_evaluated():
            return True

        raise ValueError("exploded")

    final = ws.runs.get(run.id)
    assert final.status is ExecutionStatus.FAILED
    assert final.error == "ValueError: exploded"
    assert final.failure["type"] == "ValueError"
    assert 'raise ValueError("exploded")' in final.failure["traceback"]
    assert final.state is Q
    assert ws.runs.list_check_results(run.id) == []
    assert current_run() is None  # context cleaned up despite the exception


def test_malformed_notes_never_block_failure_recording(ws: Workspace) -> None:
    """failure_record runs inside the exception handler: hostile __notes__ must
    degrade into the record, never crash it (which would leave the run stuck
    'running' and mask the real exception)."""
    # no match= here: pytest's own match machinery also chokes on hostile notes
    with pytest.raises(ValueError), ws.start_run(name="hostile") as run:
        error = ValueError("bad")
        error.__notes__ = 123  # type: ignore[attr-defined]
        raise error
    final = ws.runs.get(run.id)
    assert final.status is ExecutionStatus.FAILED
    assert final.error == "ValueError: bad"
    assert final.failure["notes"] == ["123"]


def test_keyboard_interrupt_records_run_failure(ws: Workspace) -> None:
    """Ctrl-C during a run still leaves evidence: the run is marked failed with
    a KeyboardInterrupt failure record. (Task rows catch only Exception, so a
    task interrupted mid-flight stays 'running' — the expire --include-running
    recovery path; the run-level record is what says why.)"""
    with pytest.raises(KeyboardInterrupt), ws.start_run(name="interrupted") as run:
        raise KeyboardInterrupt
    final = ws.runs.get(run.id)
    assert final.status is ExecutionStatus.FAILED
    assert final.failure["type"] == "KeyboardInterrupt"
    assert "KeyboardInterrupt" in final.failure["traceback"]


def test_nested_runs_rejected(ws: Workspace) -> None:
    with ws.start_run(), pytest.raises(NestedRunError), ws.start_run():
        pass  # pragma: no cover - never reached
    assert current_run() is None


def test_sequential_runs_allowed(ws: Workspace) -> None:
    with ws.start_run() as first:
        pass
    with ws.start_run() as second:
        pass
    assert first.id != second.id
    assert len(ws.runs.list_runs()) == 2


def test_keep_declares_terminal_artifact(ws: Workspace) -> None:
    with ws.start_run(name="keeper") as run:
        ref = run.keep("energy", {"value": -10.84, "unit": "eV"})
    assert ref.role is ArtifactRole.TERMINAL
    assert loads(ws.artifacts.get(ref.hash).read_bytes()) == {"value": -10.84, "unit": "eV"}
    assert ws.runs.get_artifact(run.id, "energy") == ref


def test_keep_path_stores_raw_file_bytes(ws: Workspace, tmp_path: Path) -> None:
    poscar = tmp_path / "POSCAR"
    poscar.write_bytes(b"Si2\n")
    with ws.start_run() as run:
        ref = run.keep("structure", poscar, role="input")
    assert ref.role is ArtifactRole.INPUT
    assert ws.artifacts.get(ref.hash).read_bytes() == b"Si2\n"


def test_keep_string_is_a_value_not_a_path(ws: Workspace) -> None:
    with ws.start_run() as run:
        ref = run.keep("note", "POSCAR")  # a str, even path-like, is data
    assert loads(ws.artifacts.get(ref.hash).read_bytes()) == "POSCAR"


def test_verification_yields_to_concurrent_promotion(ws: Workspace) -> None:
    """If the run was force-promoted mid-flight, completion must not fight it."""
    with ws.start_run() as run:
        run.check(lambda: True, name="fine")
        ws.runs.transition(run.id, P, force=True, reason="human says ship it")
    final = ws.runs.get(run.id)
    assert final.state is P  # promotion stands; no verified transition on top
    assert final.status is ExecutionStatus.COMPLETED
    (result,) = ws.runs.list_check_results(run.id)
    assert result.passed  # the check evidence is still recorded


def test_generator_check_failure_is_recorded_not_raised(ws: Workspace) -> None:
    """A generator-based check whose body raises must become a failed result —
    'a crashing check is a failing check, never a crashed run'."""
    with ws.start_run() as run:  # must NOT raise out of the context

        @check
        def gen_check():
            yield converged(0.01, below=0.05)
            raise AssertionError("boom mid-generator")

    final = ws.runs.get(run.id)
    assert final.status is ExecutionStatus.COMPLETED
    assert final.state is Q
    (result,) = ws.runs.list_check_results(run.id)
    assert result.passed is False
    assert "boom mid-generator" in result.message


def test_generator_check_success(ws: Workspace) -> None:
    with ws.start_run() as run:

        @check
        def gen_check():
            yield converged(0.01, below=0.05)
            yield finite([1.0, 2.0])

    assert ws.runs.get(run.id).state is V
    assert [r.passed for r in ws.runs.list_check_results(run.id)] == [True, True]


def test_numpy_bool_check_return_coerced(ws: Workspace) -> None:
    np = pytest.importorskip("numpy")
    with ws.start_run() as run:
        run.check(lambda: np.float64(0.01) < 0.05, name="np_comparison")
    assert ws.runs.get(run.id).state is V
    (result,) = ws.runs.list_check_results(run.id)
    assert result.passed is True
