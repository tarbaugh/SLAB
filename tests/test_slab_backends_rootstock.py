"""Rootstock failure-evidence tests: fake the calculator and the WorkerDiedError.

Rootstock's worker holds its state in memory, not on disk, so
``collect_failure_evidence`` cannot produce files for it — but the
``WorkerDiedError`` message carries the post-mortem tails, and the tests here
pin that we surface those as a standard ``engine error (rootstock worker):``
note, the same pattern the file-IO engines use. No rootstock install is
required; the seam is class-name + module-name + exception-chain walk.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

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


# -- setup lines --------------------------------------------------------------


def _fake_rootstock(monkeypatch: object, seen: dict[str, object]) -> None:
    """A rootstock package whose calculator records the environment it was
    spawned into, so a test can prove the setup acted before construction."""
    import os
    import sys
    import types

    class RootstockCalculator:
        def __init__(self, **options: object) -> None:
            seen["options"] = options
            seen["env"] = dict(os.environ)

    fake = types.ModuleType("rootstock")
    fake.RootstockCalculator = RootstockCalculator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rootstock", fake)  # type: ignore[attr-defined]


def _rootstock_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setup: list[str]) -> None:
    import json

    (tmp_path / "slab.toml").write_text(
        f'[engines.rootstock]\nroot = "{tmp_path}"\nsetup = {json.dumps(setup)}\n'
    )
    monkeypatch.chdir(tmp_path)
    from slab.backends import _setup_environment

    _setup_environment.cache_clear()


def test_rootstock_setup_lines_act_before_the_worker_is_spawned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real MACE worker died importing torchvision inside the sandbox
    container: the user's shell had loaded the CUDA module, and nothing had
    run it for the worker. The setup's variables must be in this process
    when the calculator is built, and recorded so job submission can
    restore the environment it started from."""
    from slab import engines as engines_module
    from slab.engines import applied_env

    seen: dict[str, object] = {}
    _fake_rootstock(monkeypatch, seen)
    monkeypatch.delenv("SLAB_TEST_ROOTSTOCK_SETUP", raising=False)
    _rootstock_project(tmp_path, monkeypatch, ["export SLAB_TEST_ROOTSTOCK_SETUP=applied"])
    try:
        get_calculator("rootstock", checkpoint="mace-mp-0-medium")
        env = seen["env"]
        assert isinstance(env, dict) and env["SLAB_TEST_ROOTSTOCK_SETUP"] == "applied"
        assert applied_env()["SLAB_TEST_ROOTSTOCK_SETUP"] == (None, "applied")
        options = seen["options"]
        assert isinstance(options, dict)
        assert options["root"] == str(tmp_path) and "setup" not in options
    finally:
        os.environ.pop("SLAB_TEST_ROOTSTOCK_SETUP", None)
        engines_module._APPLIED_ENV.pop("SLAB_TEST_ROOTSTOCK_SETUP", None)


def test_a_failing_rootstock_setup_names_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slab.errors import EngineNotAvailableError

    seen: dict[str, object] = {}
    _fake_rootstock(monkeypatch, seen)
    _rootstock_project(tmp_path, monkeypatch, ["echo no such module >&2", "false"])
    with pytest.raises(EngineNotAvailableError) as info:
        get_calculator("rootstock", checkpoint="mace-mp-0-medium")
    assert "[engines.rootstock] setup exited 1" in str(info.value)
    assert "no such module" in str(info.value)
    assert "env" not in seen  # nothing was spawned


def test_no_rootstock_setup_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slab.backends import rootstock_setup_env

    seen: dict[str, object] = {}
    _fake_rootstock(monkeypatch, seen)
    (tmp_path / "slab.toml").write_text(f'[engines.rootstock]\nroot = "{tmp_path}"\n')
    monkeypatch.chdir(tmp_path)
    assert rootstock_setup_env() == {}
    get_calculator("rootstock", checkpoint="mace-mp-0-medium")
    assert "env" in seen
