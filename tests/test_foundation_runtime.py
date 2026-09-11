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
    for name in ("SLAB_CPUS", "SLAB_GPUS", "SLAB_NTASKS", "SLAB_THREADS", "SLURM_CPUS_PER_TASK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SLURM_NTASKS", "2")
    first = ws.reserve(ntasks=1, budget=_budget())
    whole = ws.reserve(budget=_budget())
    assert whole.cpus == (1, 2, 3) and whole.gpus == () and whole.ntasks == 2
    assert whole.threads == 1
    from foundation.errors import ResourcesError

    with pytest.raises(ResourcesError, match="no cpu is free"):
        ws.reserve(budget=_budget())
    ws.runs.release_reservation(first.id)
    ws.runs.release_reservation(whole.id)
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    shrunk = ws.reserve(budget=_budget())
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
