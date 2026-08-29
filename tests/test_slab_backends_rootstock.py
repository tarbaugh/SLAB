"""Rootstock failure-evidence tests: fake the calculator and the WorkerDiedError.

Rootstock's worker holds its state in memory, not on disk, so
``collect_failure_evidence`` cannot produce files for it — but the
``WorkerDiedError`` message carries the post-mortem tails, and the tests here
pin that we surface those as a standard ``engine error (rootstock worker):``
note, the same pattern the file-IO engines use. No rootstock install is
required; the seam is class-name + module-name + exception-chain walk.
"""

from __future__ import annotations

from slab.backends import collect_failure_evidence, get_calculator


class _FakeRootstockCalc:
    """Duck-typed to look like ``rootstock.calculator.RootstockCalculator``."""


_FakeRootstockCalc.__module__ = "rootstock.calculator"
_FakeRootstockCalc.__name__ = "RootstockCalculator"


class _FakeWorkerDiedError(RuntimeError):
    """Named to match rootstock's own exception; the note collector uses
    class-name matching, so no rootstock import is needed."""


_FakeWorkerDiedError.__name__ = "WorkerDiedError"


def test_no_exception_produces_no_note() -> None:
    """The seam is best-effort: without an exception in hand, no note fires."""
    assert collect_failure_evidence(_FakeRootstockCalc()) == ([], [])
    assert collect_failure_evidence(_FakeRootstockCalc(), None) == ([], [])


def test_worker_died_error_surfaces_as_a_standard_note() -> None:
    exc = _FakeWorkerDiedError(
        "worker died: exit=139; stderr tail: RuntimeError: CUDA out of memory."
    )
    notes, files = collect_failure_evidence(_FakeRootstockCalc(), exc)
    assert files == []
    (note,) = notes
    assert note.startswith("engine error (rootstock worker): ")
    assert "exit=139" in note
    assert "CUDA out of memory" in note


def test_worker_died_error_in_the_exception_chain_still_surfaces() -> None:
    """The exception the task actually raises is often not the rootstock one:
    an optimizer's own RuntimeError wraps the WorkerDiedError as __cause__.
    The walker must traverse the chain to find the story."""
    root_cause = _FakeWorkerDiedError("worker died: exit=1; stderr: bad checkpoint hash")
    outer = RuntimeError("BFGS step failed")
    outer.__cause__ = root_cause
    notes, _ = collect_failure_evidence(_FakeRootstockCalc(), outer)
    (note,) = notes
    assert "bad checkpoint hash" in note


def test_long_worker_message_is_capped_not_dropped() -> None:
    """A worker that dumped ten kilobytes of stderr must not crowd the record;
    the tail is capped so the traceback and the other notes still fit."""
    exc = _FakeWorkerDiedError("worker died: " + ("x" * 20_000))
    (note,) = collect_failure_evidence(_FakeRootstockCalc(), exc)[0]
    assert " [...]" in note
    assert len(note) < 5_000


def test_non_rootstock_calculators_are_unaffected_by_the_new_arg() -> None:
    """EMT and LJ still return the empty evidence tuple, exception or not."""
    exc = _FakeWorkerDiedError("stray")
    assert collect_failure_evidence(get_calculator("emt"), exc) == ([], [])
    assert collect_failure_evidence(get_calculator("lj"), None) == ([], [])
