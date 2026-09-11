"""The scratch owner marker and the inventory that reads it.

Ownership is recorded, never inferred from age: every slab-managed scratch
directory carries a marker naming its process, host, and run, and
:func:`slab.scratch.leftovers` reads it back. Nothing here deletes.
"""

import json
import os
from pathlib import Path

import pytest

from conftest import seed_scratch, vanished_pid
from slab.backends import _scratch_dir
from slab.scratch import OWNER_MARKER, leftovers, mark_owner, this_host


def test_a_scratch_dir_carries_its_owner(
    scratch_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLAB_RUN_ID", "run-abc")
    monkeypatch.setenv("SLAB_SESSION", "chat-1")
    made = _scratch_dir("slab-qe-")
    assert made.parent == scratch_root
    marker = json.loads((made / OWNER_MARKER).read_text())
    assert marker["pid"] == os.getpid()
    assert marker["host"] == this_host()
    assert marker["run_id"] == "run-abc"
    assert marker["session"] == "chat-1"
    assert marker["prefix"] == "slab-qe-"


def test_a_scratch_dir_outside_a_run_names_no_run(
    scratch_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLAB_RUN_ID", raising=False)
    monkeypatch.delenv("SLAB_SESSION", raising=False)
    made = _scratch_dir("slab-lammps-")
    marker = json.loads((made / OWNER_MARKER).read_text())
    assert marker["run_id"] is None and marker["session"] is None


def test_leftovers_parse_the_marker_and_report_liveness(tmp_path: Path) -> None:
    gone = vanished_pid()
    seed_scratch(tmp_path, "slab-qe-live", run_id="run-1")
    seed_scratch(tmp_path, "slab-qe-dead", pid=gone, run_id="run-2")
    seed_scratch(tmp_path, "slab-qe-away", pid=1, host="another-node", run_id="run-3")
    seed_scratch(tmp_path, "slab-qe-noown", marker=False)
    seed_scratch(tmp_path, "slab-qe-norun", pid=gone)
    (tmp_path / "other-tool-x").mkdir()  # not slab's: never listed
    (tmp_path / "slab-not-a-dir").write_text("a file with the prefix\n")

    found = {item.path.name: item for item in leftovers(tmp_path)}
    assert sorted(found) == [
        "slab-qe-away", "slab-qe-dead", "slab-qe-live", "slab-qe-noown", "slab-qe-norun"
    ]
    assert found["slab-qe-live"].alive is True
    assert found["slab-qe-live"].unowned is False
    assert found["slab-qe-live"].owner is not None
    assert found["slab-qe-live"].owner.run_id == "run-1"
    assert found["slab-qe-dead"].alive is False
    assert found["slab-qe-away"].alive is None  # another host: nothing to see from here
    assert found["slab-qe-noown"].owner is None
    assert found["slab-qe-noown"].unowned is True
    assert found["slab-qe-noown"].alive is None
    assert found["slab-qe-norun"].unowned is True  # a marker, but no run
    assert found["slab-qe-norun"].alive is False
    # The payload plus the marker; a directory with no marker is the payload alone.
    assert found["slab-qe-noown"].size_bytes == 32
    assert all(item.size_bytes > 32 for name, item in found.items() if name != "slab-qe-noown")


def test_an_unreadable_marker_is_unowned(tmp_path: Path) -> None:
    made = seed_scratch(tmp_path, "slab-qe-broken", marker=False)
    (made / OWNER_MARKER).write_text("not json\n")
    (item,) = leftovers(tmp_path)
    assert item.owner is None and item.unowned


def test_leftovers_of_a_missing_root_is_empty(tmp_path: Path) -> None:
    assert leftovers(tmp_path / "nowhere") == []


def test_mark_owner_returns_the_marker_path(tmp_path: Path) -> None:
    assert mark_owner(tmp_path, prefix="slab-x-") == tmp_path / OWNER_MARKER


def test_scratch_root_is_the_configured_one_or_none(
    scratch_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from slab import scratch

    assert scratch.scratch_root() == scratch_root
    empty = tmp_path / "empty-project"
    empty.mkdir()
    monkeypatch.chdir(empty)
    assert scratch.scratch_root() is None  # the platform temp directory is never swept
