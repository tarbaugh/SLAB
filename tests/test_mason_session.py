"""The session layout facts ``slab purge`` relies on: transcript groups,
the files no group claims, and the locks no process holds."""

from __future__ import annotations

import fcntl
from pathlib import Path

from mason.session import stale_locks, transcript_groups, unrecognised_session_files


def _sessions(root: Path) -> Path:
    sessions = root / "mason" / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    return sessions


def test_an_orphan_delegation_is_a_group_of_its_own_for_the_sweep(tmp_path: Path) -> None:
    """A delegation transcript whose conversation is gone is reached by the
    sweep as its own group; the readers, which resume and report
    conversations, never see it."""
    sessions = _sessions(tmp_path)
    (sessions / "20260826-120000-111.jsonl").write_text("{}\n")
    (sessions / "20260826-120000-111-crystal-1.jsonl").write_text("{}\n")
    (sessions / "20260820-080000-9-md-expert-2.jsonl").write_text("{}\n")  # orphan
    (sessions / "20260820-080000-9-md-expert-2-critic-1.jsonl").write_text("{}\n")  # orphan too
    groups = transcript_groups(tmp_path, include_orphans=True)
    assert [(c.name, [s.name for s in siblings]) for c, siblings in groups] == [
        ("20260820-080000-9-md-expert-2-critic-1.jsonl", []),
        ("20260820-080000-9-md-expert-2.jsonl", []),
        ("20260826-120000-111.jsonl", ["20260826-120000-111-crystal-1.jsonl"]),
    ]
    assert [c.name for c, _ in transcript_groups(tmp_path)] == ["20260826-120000-111.jsonl"]


def test_unrecognised_session_files_are_listed_not_swept(tmp_path: Path) -> None:
    sessions = _sessions(tmp_path)
    (sessions / "20260826-120000-111.jsonl").write_text("{}\n")
    (sessions / "20260826-120000-111.compactions.md").write_text("#\n")
    (sessions / "20260826-120000-111-crystal-1.jsonl").write_text("{}\n")
    (sessions / "20260826-120000-111-crystal-1.compactions.md").write_text("#\n")
    (sessions / "20260801-000000-5-md-expert-1.jsonl").write_text("{}\n")  # orphan: grouped
    (sessions / "20260701-000000-3.compactions.md").write_text("#\n")  # its transcript is gone
    (sessions / "transcript.jsonl.bak").write_text("{}\n")
    (sessions / "notes.txt").write_text("stray\n")
    (sessions / "a-dir").mkdir()
    reviews = tmp_path / "mason" / "reviews"
    reviews.mkdir()
    (reviews / "20260826-120000-111-review-1.md").write_text("---\n")  # claimed
    (reviews / "20260801-000000-5-md-expert-1-review-1.md").write_text("---\n")  # the orphan's
    (reviews / "20260601-000000-2-review-1.md").write_text("---\n")  # its transcript is gone
    assert [p.name for p in unrecognised_session_files(tmp_path)] == [
        "20260601-000000-2-review-1.md",
        "20260701-000000-3.compactions.md",
        "notes.txt",
        "transcript.jsonl.bak",
    ]
    assert unrecognised_session_files(tmp_path / "elsewhere") == []


def test_a_stale_lock_is_detected_and_a_held_one_is_not(tmp_path: Path) -> None:
    locks = tmp_path / "mason" / "locks"
    locks.mkdir(parents=True)
    stale = locks / "1111111111111111.lock"
    stale.write_text("pid 1, cwd /gone\n")
    held = locks / "2222222222222222.lock"
    held.write_text("pid 2, cwd /here\n")
    (locks / "notes.txt").write_text("not a lock\n")
    with open(held, "a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert stale_locks(tmp_path) == [stale]
    assert stale_locks(tmp_path) == [stale, held]  # released with the handle
    assert stale_locks(tmp_path / "elsewhere") == []
