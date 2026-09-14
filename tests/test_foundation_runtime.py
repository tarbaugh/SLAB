"""Tests for Workspace, the run context, and check evaluation/gating."""

from pathlib import Path
from typing import Any

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
    resolve_session_id,
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
        assert run.id in repr(run)  # repr carries the id
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


def test_a_custom_check_may_carry_what_it_observed(ws: Workspace) -> None:
    """A record that said only ``returned False`` made one real session dig
    the artifact store by hand to learn what the lattice constant was."""
    with ws.start_run() as run:
        run.check(lambda: (False, 3.21), name="a0_reasonable")
        run.check(lambda: (True, 3.17, {"near": 3.17}), name="a0_with_expected")
        run.check(
            lambda: {"passed": True, "observed": 361.3, "expected": {"below": 720}}, name="sp_time"
        )
    by_name = {r.name: r for r in ws.runs.list_check_results(run.id)}
    a0 = by_name["a0_reasonable"]
    assert not a0.passed and a0.kind == "custom" and a0.observed == 3.21
    assert a0.message == "returned False; observed 3.21"
    assert by_name["a0_with_expected"].expected == {"near": 3.17}
    assert "expected {'near': 3.17}" in by_name["a0_with_expected"].message
    assert by_name["sp_time"].passed and by_name["sp_time"].observed == 361.3


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


# -- session stamps ----------------------------------------------------------


def test_start_run_stamps_explicit_session(ws: Workspace) -> None:
    with ws.start_run(name="probe", session="chat-1") as run:
        pass
    assert ws.runs.get(run.id).session == "chat-1"


def test_start_run_falls_back_to_the_environment(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLAB_SESSION", "from-env")
    with ws.start_run(name="probe") as run:
        pass
    assert ws.runs.get(run.id).session == "from-env"


def test_explicit_session_beats_the_environment(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLAB_SESSION", "from-env")
    with ws.start_run(name="probe", session="explicit") as run:
        pass
    assert ws.runs.get(run.id).session == "explicit"


def test_unstamped_run_has_no_session(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLAB_SESSION", raising=False)
    with ws.start_run(name="probe") as run:
        pass
    assert ws.runs.get(run.id).session is None


def test_blank_environment_session_never_stamps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLAB_SESSION", "")
    assert resolve_session_id() is None
    assert resolve_session_id("chat-1") == "chat-1"


def test_failed_run_keeps_its_session(ws: Workspace) -> None:
    """A session promote must be able to see the failures it refuses."""
    with pytest.raises(RuntimeError), ws.start_run(name="boom", session="chat-1") as run:
        raise RuntimeError("nope")
    failed = ws.runs.get(run.id)
    assert (failed.session, failed.status) == ("chat-1", ExecutionStatus.FAILED)


# -- run liveness --------------------------------------------------------------


def _vanished_pid() -> int:
    """The pid of a process that ran and exited: recorded, and gone."""
    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def test_start_run_stamps_the_owning_process(ws: Workspace) -> None:
    import os

    from foundation.runtime import run_liveness, this_host

    with ws.start_run(name="stamped") as run:
        live = ws.runs.get(run.id)
        assert (live.pid, live.host) == (os.getpid(), this_host())
        assert run_liveness(live) == "alive"
    assert run_liveness(ws.runs.get(run.id)) == "unrecorded"  # finished: nothing to check


def test_reap_dead_marks_a_run_whose_process_vanished(ws: Workspace) -> None:
    """A run stamped with this host and a pid that no longer exists is marked
    failed and says who did it. A live run, a run on another host, and a run
    from before the stamp are left as they are: nothing about them is known."""
    import os

    from foundation.models import Run
    from foundation.runtime import run_liveness, this_host

    gone = _vanished_pid()
    dead = ws.runs.create(Run(name="killed"))
    ws.runs.set_status(dead.id, "running", pid=gone, host=this_host())
    live = ws.runs.create(Run(name="live"))
    ws.runs.set_status(live.id, "running", pid=os.getpid(), host=this_host())
    away = ws.runs.create(Run(name="elsewhere"))
    ws.runs.set_status(away.id, "running", pid=1, host="another-node")
    old = ws.runs.create(Run(name="pre-stamp"))
    ws.runs.set_status(old.id, "running")

    reaped = ws.reap_dead(caller="the test")
    assert [r.id for r in reaped] == [dead.id]
    after = ws.runs.get(dead.id)
    assert after.status is ExecutionStatus.FAILED
    assert after.error == f"process {gone} on {this_host()} is gone; marked failed by the test"
    assert after.finished_at is not None
    for untouched in (live, away, old):
        assert ws.runs.get(untouched.id).status is ExecutionStatus.RUNNING
    assert run_liveness(ws.runs.get(live.id)) == "alive"
    assert run_liveness(ws.runs.get(away.id)) == "elsewhere"
    assert run_liveness(ws.runs.get(old.id)) == "unrecorded"
    assert ws.reap_dead(caller="again") == []  # nothing left to reap


def test_fail_run_retires_what_reap_cannot_judge_and_refuses_a_live_process(
    ws: Workspace,
) -> None:
    import os

    from foundation.errors import RunStateError
    from foundation.models import Run
    from foundation.runtime import this_host

    away = ws.runs.create(Run(name="elsewhere"))
    ws.runs.set_status(away.id, "running", pid=1, host="another-node")
    failed = ws.fail_run(away.id, reason="node drained")
    assert failed.status is ExecutionStatus.FAILED
    assert failed.error == "marked failed by the operator: node drained"

    live = ws.runs.create(Run(name="live"))
    ws.runs.set_status(live.id, "running", pid=os.getpid(), host=this_host())
    with pytest.raises(RunStateError, match="is alive"):
        ws.fail_run(live.id, reason="impatience")
    assert ws.runs.get(live.id).status is ExecutionStatus.RUNNING

    with pytest.raises(RunStateError, match="not 'running'"):
        ws.fail_run(away.id, reason="twice")


# -- reservations ----------------------------------------------------------------


def _budget(cpus: int = 4, gpus: int = 2):
    from slab.resources import Budget

    return Budget(cpus=tuple(range(cpus)), gpus=tuple(str(n) for n in range(gpus)))


def test_reserve_then_claim_round_trip(ws: Workspace) -> None:
    """A session reserves, the run claims: the slice lands on the run record and
    the reservation is the run's; when the run ends the reservation is released."""
    from foundation.runtime import this_host

    held = ws.reserve(ntasks=2, threads=1, gpus=1, budget=_budget())
    assert (held.cpus, held.gpus, held.host, held.run_id) == ((0, 1), ("0",), this_host(), None)
    with ws.start_run(name="sized", reservation=held) as run:
        live = ws.runs.get(run.id)
        assert live.resources == {
            "cpus": [0, 1], "gpus": ["0"], "ntasks": 2, "threads": 1, "reservation": held.id,
        }
        assert ws.runs.get_reservation(held.id).run_id == run.id
        assert ws.free_resources(budget=_budget())["free"] == {"cpus": [2, 3], "gpus": ["1"]}
        assert ws.runs.run_for_reservation(held.id).id == run.id
    assert ws.runs.get(run.id).resources["reservation"] == held.id  # provenance stays
    assert ws.runs.list_reservations() == []  # the run ended: released
    assert ws.free_resources(budget=_budget())["free"] == {"cpus": [0, 1, 2, 3], "gpus": ["0", "1"]}
    assert ws.runs.run_for_reservation(held.id).id == run.id


def test_start_run_accepts_the_reservation_id(ws: Workspace) -> None:
    held = ws.reserve(ntasks=1, budget=_budget())
    with ws.start_run(name="by-id", reservation=held.id) as run:
        assert ws.runs.get(run.id).resources["cpus"] == [0]


def test_a_reservation_that_does_not_fit_is_refused_with_the_free_amounts(ws: Workspace) -> None:
    from foundation.errors import ResourcesError

    ws.reserve(ntasks=3, budget=_budget())
    with pytest.raises(ResourcesError, match=r"2 rank\(s\) x 1 thread\(s\) = 2 cpus asked") as e:
        ws.reserve(ntasks=2, budget=_budget())
    assert e.value.free == {"cpus": [3], "gpus": ["0", "1"]}
    with pytest.raises(ResourcesError, match=r"3 gpu\(s\) asked, but only 2") as e:
        ws.reserve(ntasks=1, gpus=3, budget=_budget())
    assert e.value.free["cpus"] == [3]


def test_an_unsized_reservation_takes_the_whole_free_budget(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a gpu in the budget an unsized launch takes every free cpu.
    With one it holds one plain rank (see the store's tests)."""
    for name in ("SLAB_CPUS", "SLAB_GPUS", "SLAB_NTASKS", "SLAB_THREADS", "SLURM_CPUS_PER_TASK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    first = ws.reserve(ntasks=1, budget=_budget(gpus=0))
    whole = ws.reserve(budget=_budget(gpus=0))
    assert whole.cpus == (1, 2, 3) and whole.gpus == () and whole.ntasks == 2
    assert whole.threads == 1
    from foundation.errors import ResourcesError

    with pytest.raises(ResourcesError, match="no cpu is free"):
        ws.reserve(budget=_budget(gpus=0))
    ws.runs.release_reservation(first.id)
    ws.runs.release_reservation(whole.id)
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    shrunk = ws.reserve(budget=_budget(gpus=0))
    assert (shrunk.ntasks, shrunk.threads, shrunk.cpus) == (4, 1, (0, 1, 2, 3))


def test_a_reservation_whose_holder_died_is_released_by_reap(ws: Workspace) -> None:
    """The holder's death frees the slice at once (free is derived), and the
    next reap deletes the row; a claim of the released reservation is refused."""
    from foundation.errors import ResourcesError

    gone = _vanished_pid()
    dead = ws.reserve(ntasks=2, budget=_budget(), holder_pid=gone)
    assert ws.free_resources(budget=_budget())["free"]["cpus"] == [0, 1, 2, 3]
    again = ws.reserve(ntasks=2, budget=_budget())  # overlaps the dead row, by design
    assert again.cpus == (0, 1)
    assert ws.reap_dead(caller="test") == []
    assert [r.id for r in ws.runs.list_reservations()] == [again.id]
    with (
        pytest.raises(ResourcesError, match=f"no reservation '{dead.id}'"),
        ws.start_run(name="late", reservation=dead.id),
    ):
        pass
    assert ws.runs.list_runs() == []  # refused before any run existed


def test_a_claimed_or_foreign_reservation_is_refused(ws: Workspace) -> None:
    from foundation.errors import ResourcesError
    from foundation.models import Run

    held = ws.reserve(ntasks=1, budget=_budget())
    other = ws.runs.create(Run(name="other"))
    ws.runs.claim_reservation(held.id, other.id, host=held.host)
    with (
        pytest.raises(ResourcesError, match="already claimed by run"),
        ws.start_run(name="second", reservation=held.id),
    ):
        pass
    away = ws.runs.reserve(
        host="another-node", holder_pid=1, budget_cpus=range(2), budget_gpus=(), ntasks=1
    )
    with (
        pytest.raises(ResourcesError, match="made for host 'another-node'"),
        ws.start_run(name="elsewhere", reservation=away.id),
    ):
        pass
    with pytest.raises(ResourcesError, match="a claimed reservation belongs to its run"):
        ws.runs.transfer_reservation(held.id, holder_pid=2)


def test_free_shrinks_and_grows_as_runs_start_and_end(ws: Workspace) -> None:
    def free() -> list[int]:
        return ws.free_resources(budget=_budget(cpus=6, gpus=0))["free"]["cpus"]

    assert free() == [0, 1, 2, 3, 4, 5]
    a = ws.reserve(ntasks=2, budget=_budget(cpus=6, gpus=0))
    assert free() == [2, 3, 4, 5]
    with ws.start_run(name="a", reservation=a):
        b = ws.reserve(ntasks=3, budget=_budget(cpus=6, gpus=0))
        assert b.cpus == (2, 3, 4) and free() == [5]
    assert free() == [0, 1, 5]  # a ended; b is still held, unclaimed
    with pytest.raises(RuntimeError), ws.start_run(name="b", reservation=b):
        assert free() == [0, 1, 5]
        raise RuntimeError("boom")
    assert free() == [0, 1, 2, 3, 4, 5]  # a failed run releases too


def test_a_reaped_run_releases_its_reservation(ws: Workspace) -> None:
    from foundation.models import Run
    from foundation.runtime import this_host

    held = ws.reserve(ntasks=1, budget=_budget())
    run = ws.runs.create(Run(name="killed"))
    ws.runs.claim_reservation(held.id, run.id, host=held.host)
    ws.runs.set_status(run.id, "running", pid=_vanished_pid(), host=this_host())
    assert ws.free_resources(budget=_budget())["free"]["cpus"] == [0, 1, 2, 3]
    assert [r.id for r in ws.reap_dead(caller="test")] == [run.id]
    assert ws.runs.list_reservations() == []
    assert ws.runs.get(run.id).resources["reservation"] == held.id


def test_start_run_with_a_reservation_claims_and_starts_in_one_call(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The claim sets the run running itself; a separate set_status(running)
    would open a window in which the slice counts as free."""
    import os

    from slab.resources import Budget

    budget = Budget(cpus=(0, 1), gpus=())
    held = ws.reserve(ntasks=1, budget=budget)
    started_separately: list[object] = []
    original = ws.runs.set_status

    def spy(run_id: str, status: object, **kwargs: object) -> object:
        started_separately.append(status)
        return original(run_id, status, **kwargs)

    monkeypatch.setattr(ws.runs, "set_status", spy)
    with ws.start_run(name="sized", reservation=held) as run:
        record = ws.runs.get(run.id)
        assert record.status.value == "running" and record.pid == os.getpid()
        assert ws.free_resources(budget=budget)["free"]["cpus"] == [1]
        assert ws.release_dead() == []
    assert started_separately == [ExecutionStatus.COMPLETED]  # never RUNNING on its own
    assert ws.runs.list_reservations() == []


def test_a_hard_killed_child_is_reaped_and_its_slice_freed(ws: Workspace) -> None:
    """A background launch abandons its Popen, so a child that SIGKILL (or an
    OOM kill) ended stays a zombie, and a zombie still answers signal 0.
    process_alive reaps it and reports it dead, so reap_dead and
    release_dead free its slice without waiting for an unrelated Popen."""
    import contextlib
    import os
    import signal
    import subprocess
    import time

    from foundation.models import Run
    from foundation.runtime import this_host
    from foundation.store import process_alive
    from slab.resources import Budget

    budget = Budget(cpus=(0, 1), gpus=())
    child = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        held = ws.reserve(ntasks=1, holder_pid=child.pid, budget=budget)
        assert process_alive(child.pid)
        assert ws.free_resources(budget=budget)["free"]["cpus"] == [1]
        os.kill(child.pid, signal.SIGKILL)
        deadline = time.monotonic() + 5.0
        while process_alive(child.pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not process_alive(child.pid)  # reaped by the probe, no wait() here
        assert [r.id for r in ws.release_dead()] == [held.id]
        assert ws.free_resources(budget=budget)["free"]["cpus"] == [0, 1]
        run = ws.runs.create(Run(name="killed"))
        ws.runs.set_status(run.id, "running", pid=child.pid, host=this_host())
        assert [r.id for r in ws.reap_dead(caller="test")] == [run.id]
    finally:
        with contextlib.suppress(ProcessLookupError):
            child.kill()
        child.wait()


def test_start_run_exports_the_run_id_and_restores_it(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every scratch a calculation makes inside the run is stamped with the
    run's id, because the variable is set for the block and restored after."""
    import os

    monkeypatch.delenv("SLAB_RUN_ID", raising=False)
    with ws.start_run(name="stamped") as run:
        assert os.environ["SLAB_RUN_ID"] == run.id
    assert "SLAB_RUN_ID" not in os.environ
    monkeypatch.setenv("SLAB_RUN_ID", "outer")
    with ws.start_run(name="nested-value") as run:
        assert os.environ["SLAB_RUN_ID"] == run.id
    assert os.environ["SLAB_RUN_ID"] == "outer"


# -- a run is live only while its job is -----------------------------------------


def test_a_run_of_another_job_is_not_alive_here_and_holds_no_slice(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two runs on this host with this very pid, one from this job and one
    from another: the other is 'other-job', never 'alive', and its slice
    does not count against this job's budget."""
    import os

    from foundation.models import Run
    from foundation.runtime import describe_liveness, run_liveness, this_host
    from slab.resources import Budget

    monkeypatch.setenv("SLURM_JOB_ID", "7")
    budget = Budget(cpus=(0, 1, 2, 3))
    mine = ws.runs.create(Run(name="mine", job_id="7"))
    theirs = ws.runs.create(Run(name="theirs", job_id="8"))
    held_mine = ws.reserve(ntasks=2, budget=budget)
    assert held_mine.job_id == "7"
    held_theirs = ws.runs.reserve(
        host=this_host(), holder_pid=os.getpid(), budget_cpus=budget.cpus, budget_gpus=(),
        ntasks=2, job_id="8",
    )
    ws.runs.claim_reservation(held_mine.id, mine.id, host=this_host(), pid=os.getpid())
    ws.runs.claim_reservation(held_theirs.id, theirs.id, host=this_host(), pid=os.getpid())
    assert run_liveness(ws.runs.get(mine.id)) == "alive"
    assert run_liveness(ws.runs.get(theirs.id)) == "other-job"
    assert describe_liveness(ws.runs.get(theirs.id)) == (
        f"process {os.getpid()} on {this_host()} belongs to job 8, not this job (7); "
        "liveness not checked from here"
    )
    free = ws.free_resources(budget=budget)
    assert free["reservations"] == [held_mine.id]
    assert free["free"]["cpus"] == [2, 3]
    # No scheduler here: neither run is reaped, and neither slice is released.
    assert ws.reap_dead(caller="the test") == []
    assert {r.id for r in ws.runs.list_reservations()} == {held_mine.id, held_theirs.id}


def _job_answer(state: str) -> Any:
    from slab.hpc import JobState, JobStatus

    def answer(job_id: str) -> JobStatus:
        return JobStatus(job_id=job_id, state=JobState(state), raw=state.upper())

    return answer


def test_reap_dead_fails_the_runs_of_an_ended_job(
    ws: Workspace, scratch_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scheduler's word closes a run wherever it ran: a cancelled job's
    run is failed, its reservation released, its scratch removed. A running
    job, an undetermined answer, and no scheduler at all leave it alone."""
    import os

    from conftest import seed_scratch
    from foundation import runtime
    from foundation.models import Run
    from foundation.runtime import run_liveness, this_host
    from slab.hpc import SchedulerNotAvailableError

    monkeypatch.setenv("SLURM_JOB_ID", "7")
    jobbed = ws.runs.create(Run(name="in-job-8", job_id="8"))
    held = ws.runs.reserve(
        host=this_host(), holder_pid=os.getpid(), budget_cpus=range(2), budget_gpus=(),
        ntasks=1, job_id="8",
    )
    ws.runs.claim_reservation(held.id, jobbed.id, host=this_host(), pid=os.getpid())
    scratch = seed_scratch(scratch_root, "slab-lammps-8", run_id=jobbed.id)
    asked: list[str] = []

    def unavailable(job_id: str) -> Any:
        asked.append(job_id)
        raise SchedulerNotAvailableError("no squeue")

    monkeypatch.setattr(runtime, "job_state", unavailable)
    assert ws.reap_dead(caller="the test") == []
    assert asked == ["8"]
    monkeypatch.setattr(runtime, "job_state", _job_answer("running"))
    assert ws.reap_dead(caller="the test") == []
    monkeypatch.setattr(runtime, "job_state", _job_answer("undetermined"))
    assert ws.reap_dead(caller="the test") == []
    assert ws.runs.get(jobbed.id).status is ExecutionStatus.RUNNING
    assert run_liveness(ws.runs.get(jobbed.id)) == "other-job"
    assert scratch.is_dir()

    monkeypatch.setattr(runtime, "job_state", _job_answer("cancelled"))
    reaped = ws.reap_dead(caller="the test")
    assert [r.id for r in reaped] == [jobbed.id]
    after = ws.runs.get(jobbed.id)
    assert after.status is ExecutionStatus.FAILED
    assert after.error == "job 8 is cancelled; marked failed by the test"
    assert ws.runs.list_reservations() == []
    assert not scratch.exists()


def test_a_session_record_is_stale_once_its_runs_were_failed_by_job_end(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A harness record whose only running run belongs to an ended job turns
    stale when the reap closes that run, so purge may delete it."""
    from foundation import runtime
    from foundation.models import Run
    from foundation.session_record import SessionRecord, stale_records

    records = tmp_path / "records"
    SessionRecord(records, "sess-old", client="mcp").record({"type": "start"})
    run = ws.runs.create(Run(name="in-job", job_id="8", session="sess-old"))
    ws.runs.set_status(run.id, "running", pid=1, host="compute-7")
    assert stale_records(records, runs=ws.runs, keep_newest=False) == []
    monkeypatch.setattr(runtime, "job_state", _job_answer("timeout"))
    assert [r.id for r in ws.reap_dead(caller="the test")] == [run.id]
    assert ws.runs.get(run.id).error == "job 8 is timeout; marked failed by the test"
    stale = stale_records(records, runs=ws.runs, keep_newest=False)
    assert [r.session_id for r in stale] == ["sess-old"]


def test_reap_dead_settles_a_run_an_ended_job_left_on_another_host(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run of a job that died on another node, seen from a new job: the
    scheduler says the job is terminal, so the run is failed and its slice
    released, though its host and pid cannot be checked from here."""
    from foundation import runtime
    from foundation.models import Run

    monkeypatch.setenv("SLURM_JOB_ID", "9")
    left = ws.runs.create(Run(name="left-by-8", job_id="8"))
    held = ws.runs.reserve(
        host="compute-7", holder_pid=1, budget_cpus=range(4), budget_gpus=("0",),
        gpus=1, job_id="8",
    )
    ws.runs.claim_reservation(held.id, left.id, host="compute-7", pid=1)
    asked: list[str] = []

    def answer(job_id: str) -> Any:
        asked.append(job_id)
        return _job_answer("cancelled")(job_id)

    monkeypatch.setattr(runtime, "job_state", answer)
    reaped = ws.reap_dead(caller="slab runs reap")
    assert [r.id for r in reaped] == [left.id]
    assert asked == ["8"]
    after = ws.runs.get(left.id)
    assert after.status is ExecutionStatus.FAILED
    assert after.error == "job 8 is cancelled; marked failed by slab runs reap"
    assert ws.runs.list_reservations() == []


def test_settle_ended_jobs_asks_only_the_scheduler(
    ws: Workspace, scratch_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The settle behind 'slab purge': a run of a terminal job is failed, its
    slice released, its scratch removed. A run of a running job, a run with
    no job (even one whose pid is gone here), and every run where the
    scheduler cannot be reached stay running. A job named as ended is taken
    at its word, without a question to the scheduler."""
    from conftest import seed_scratch
    from foundation import runtime
    from foundation.models import Run
    from foundation.runtime import this_host
    from slab.hpc import JobState, JobStatus, SchedulerNotAvailableError

    ended = ws.runs.create(Run(name="ended", job_id="8"))
    held = ws.runs.reserve(
        host="compute-7", holder_pid=1, budget_cpus=range(2), budget_gpus=(),
        ntasks=1, job_id="8",
    )
    ws.runs.claim_reservation(held.id, ended.id, host="compute-7", pid=1)
    scratch = seed_scratch(scratch_root, "slab-lammps-8", run_id=ended.id)
    live = ws.runs.create(Run(name="live", job_id="9"))
    ws.runs.set_status(live.id, "running", pid=1, host="compute-7")
    lost = ws.runs.create(Run(name="lost", job_id="10"))
    ws.runs.set_status(lost.id, "running", pid=1, host="compute-7")
    bare = ws.runs.create(Run(name="bare"))
    ws.runs.set_status(bare.id, "running", pid=2**22 - 1, host=this_host())

    def unavailable(job_id: str) -> Any:
        raise SchedulerNotAvailableError("no squeue")

    monkeypatch.setattr(runtime, "job_state", unavailable)
    assert ws.settle_ended_jobs(caller="the test") == []

    states = {"8": JobState.COMPLETED, "9": JobState.RUNNING, "10": JobState.UNDETERMINED}
    asked: list[str] = []

    def answer(job_id: str) -> JobStatus:
        asked.append(job_id)
        return JobStatus(job_id=job_id, state=states[job_id], raw=states[job_id].value)

    monkeypatch.setattr(runtime, "job_state", answer)
    dry = ws.settle_ended_jobs(caller="the test", dry_run=True)
    assert [r.id for r in dry] == [ended.id]
    assert ws.runs.get(ended.id).status is ExecutionStatus.RUNNING
    assert scratch.is_dir()

    asked.clear()
    settled = ws.settle_ended_jobs(caller="the test")
    assert [r.id for r in settled] == [ended.id]
    assert sorted(asked) == ["10", "8", "9"]
    assert ws.runs.get(ended.id).error == "job 8 is completed; marked failed by the test"
    assert ws.runs.list_reservations() == []
    assert not scratch.exists()
    for run in (live, lost, bare):
        assert ws.runs.get(run.id).status is ExecutionStatus.RUNNING

    asked.clear()
    forced = ws.settle_ended_jobs(caller="the test", ended=["10"])
    assert [r.id for r in forced] == [lost.id]
    assert "10" not in asked
    assert ws.runs.get(lost.id).error == "job 10 ended; marked failed by the test"


# -- evidence on a failed check ---------------------------------------------------


def test_a_bare_false_records_the_check_source_and_the_keys_it_read(ws: Workspace) -> None:
    """'returned False' alone made a delegate re-read artifacts by hand: the
    record now carries the check's text and the keys of the result it read."""
    result = {"rows": [{"Temp": 300.0}], "steps": 0}
    with ws.start_run() as run:

        @check
        def enough_steps():
            return result["steps"] > 10

    (record,) = ws.runs.list_check_results(run.id)
    assert (record.passed, record.message, record.observed) == (False, "returned False", None)
    assert record.evidence is not None
    assert record.evidence["source"].startswith("@check\ndef enough_steps():")
    assert record.evidence["keys"] == {"result": ["rows", "steps"]}
    assert "raised" not in record.evidence


def test_a_raising_check_records_the_exception_and_the_line(ws: Workspace) -> None:
    result = {"rows": [], "steps": 1000}
    with ws.start_run() as run:

        @check
        def fcc_fraction():
            return result["n_fcc"] / result["steps"] > 0.9

    (record,) = ws.runs.list_check_results(run.id)
    assert record.kind == "error"
    assert record.evidence is not None
    assert record.evidence["raised"] == "KeyError: 'n_fcc'"
    line = record.evidence["line"]
    assert line.startswith("test_foundation_runtime.py:")
    assert line.endswith('return result["n_fcc"] / result["steps"] > 0.9')
    assert record.evidence["keys"] == {"result": ["rows", "steps"]}


def test_a_check_that_says_what_it_saw_records_no_evidence(ws: Workspace) -> None:
    with ws.start_run() as run:
        run.check(lambda: (False, 0.42, 0.9), name="fraction")
        run.check(lambda: True, name="fine")
    by_name = {r.name: r for r in ws.runs.list_check_results(run.id)}
    assert by_name["fraction"].observed == 0.42
    assert by_name["fraction"].evidence is None
    assert by_name["fine"].evidence is None


def test_the_source_in_the_evidence_stops_at_forty_lines(tmp_path: Path) -> None:
    import runpy

    from foundation.runtime import EVIDENCE_SOURCE_LINES, check_evidence

    body = "\n".join(f"    x{i} = {i}" for i in range(50))
    module = tmp_path / "long_check.py"
    module.write_text(f"def long_check():\n{body}\n    return False\n")
    fn = runpy.run_path(str(module))["long_check"]
    lines = check_evidence(fn)["source"].splitlines()
    assert len(lines) == EVIDENCE_SOURCE_LINES + 1
    assert lines[-1] == "... (12 more lines)"
