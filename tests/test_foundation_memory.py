"""The machine-memory store: round trips, refusals, and provenance."""

from __future__ import annotations

import os
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from foundation import memory as memory_store
from foundation.errors import MemoryStoreError


def _utc_today() -> date:
    """Today in UTC, the day the memory store dates outages by."""
    return datetime.now(UTC).date()



@pytest.fixture()
def memory_root(tmp_path: Path) -> Path:
    """A memory directory of its own: the root conftest also uses tmp_path."""
    root = tmp_path / "memory"
    root.mkdir()
    return root


def test_a_written_memory_reads_back_whole(memory_root: Path) -> None:
    written = memory_store.write(
        "vllm-mamba-cache",
        "vLLM refuses hybrid-Mamba models at the default batch size.",
        "Set [agent.serve] args = [\"--max-num-seqs\", \"32\"] before serving one.",
        agent="pi",
        model="qwen3-30b",
        directory=memory_root,
        evidence="checked by hand",
    )
    assert written.name == "vllm-mamba-cache"
    assert written.path == (memory_root / "vllm-mamba-cache.md").resolve()

    found = memory_store.discover(memory_root)
    assert list(found) == ["vllm-mamba-cache"]
    memory = found["vllm-mamba-cache"]
    assert memory.description.startswith("vLLM refuses hybrid-Mamba")
    assert "--max-num-seqs" in memory.body()
    assert memory.agent == "pi"
    assert memory.model == "qwen3-30b"
    assert memory.created == memory.updated
    assert memory.provenance().startswith("recorded by pi on ")


def test_the_directory_is_created_on_first_write_only(tmp_path: Path) -> None:
    root = tmp_path / "not-yet"
    assert memory_store.discover(root) == {}
    assert not root.exists()
    memory_store.write("a-fact", "A fact.", "The body.", directory=root, evidence="checked by hand")
    assert root.is_dir()


def test_rewriting_a_memory_keeps_its_creation_date(memory_root: Path) -> None:
    (memory_root / "a-fact.md").write_text(
        "---\ndescription: First reading.\ncreated: 2020-01-02\nupdated: 2020-01-02\n"
        "agent: pi\n---\nBody one.\n",
        encoding="utf-8",
    )
    first = memory_store.discover(memory_root)["a-fact"]
    assert first.created == "2020-01-02"

    second = memory_store.write(
        "a-fact", "Second reading.", "Body two.", agent="md-expert", directory=memory_root,
        evidence="checked by hand",
    )
    assert second.created == "2020-01-02"  # the fact is as old as it was
    assert second.updated != second.created
    assert second.agent == "md-expert"
    assert second.body() == "Body two.\n"
    assert len(memory_store.discover(memory_root)) == 1
    assert "updated 2020" not in second.provenance()


def test_a_malformed_memory_is_loud_not_absent(memory_root: Path) -> None:
    (memory_root / "broken.md").write_text("no frontmatter here\n", encoding="utf-8")
    with pytest.raises(MemoryStoreError) as excinfo:
        memory_store.discover(memory_root)
    assert "broken.md" in str(excinfo.value)
    assert "frontmatter" in str(excinfo.value)


def test_a_memory_without_a_description_is_refused(memory_root: Path) -> None:
    (memory_root / "quiet.md").write_text("---\nagent: pi\n---\nA fact.\n", encoding="utf-8")
    with pytest.raises(MemoryStoreError, match="required 'description'"):
        memory_store.discover(memory_root)


def test_a_memory_with_an_empty_body_is_refused(memory_root: Path) -> None:
    (memory_root / "hollow.md").write_text("---\ndescription: d\n---\n\n", encoding="utf-8")
    with pytest.raises(MemoryStoreError, match="body is empty"):
        memory_store.discover(memory_root)


def test_a_frontmatter_name_that_disagrees_with_the_file_is_refused(memory_root: Path) -> None:
    (memory_root / "here.md").write_text(
        "---\nname: elsewhere\ndescription: d\n---\nA fact.\n", encoding="utf-8"
    )
    with pytest.raises(MemoryStoreError, match="disagrees with the file name"):
        memory_store.discover(memory_root)


def test_a_badly_named_file_is_refused(memory_root: Path) -> None:
    (memory_root / "Not_A_Name.md").write_text(
        "---\ndescription: d\n---\nA fact.\n", encoding="utf-8"
    )
    with pytest.raises(MemoryStoreError, match="not a valid memory name"):
        memory_store.discover(memory_root)


def test_hidden_files_and_other_suffixes_are_not_memories(memory_root: Path) -> None:
    (memory_root / ".draft.md").write_text("garbage", encoding="utf-8")
    (memory_root / "_scratch.md").write_text("garbage", encoding="utf-8")
    (memory_root / "README.txt").write_text("garbage", encoding="utf-8")
    memory_store.write(
        "real-fact", "Real.", "Body.", directory=memory_root, evidence="checked by hand"
    )
    assert list(memory_store.discover(memory_root)) == ["real-fact"]


@pytest.mark.parametrize(
    ("name", "description", "body", "expected"),
    [
        ("Not A Name", "d", "b", "not a valid memory name"),
        ("-leading", "d", "b", "not a valid memory name"),
        ("fine-name", "   ", "b", "needs a description"),
        ("fine-name", "d", "  \n ", "needs a body"),
        ("fine-name", "x" * 1025, "b", "over the 1024-character limit"),
        ("fine-name", "d", "x" * 4001, "over the 4000-character limit"),
    ],
)
def test_write_refuses_what_it_cannot_store(
    memory_root: Path, name: str, description: str, body: str, expected: str
) -> None:
    with pytest.raises(MemoryStoreError, match=expected):
        memory_store.write(
            name, description, body, directory=memory_root, evidence="checked by hand"
        )
    assert list(memory_root.iterdir()) == []


def test_the_hundredth_memory_is_the_last_new_one(memory_root: Path) -> None:
    for index in range(memory_store.MAX_MEMORIES):
        memory_store.write(
            f"fact-{index:03d}", "A fact.", "Body.", directory=memory_root,
            evidence="checked by hand",
        )
    with pytest.raises(MemoryStoreError, match="the limit"):
        memory_store.write(
            "one-too-many", "A fact.", "Body.", directory=memory_root, evidence="checked by hand"
        )
    # Updating one that already exists still works: consolidation is the way out.
    memory_store.write(
        "fact-000", "Consolidated.", "Body.", directory=memory_root, evidence="checked by hand"
    )
    assert len(memory_store.discover(memory_root)) == memory_store.MAX_MEMORIES


def test_a_failed_write_leaves_no_partial_file(
    memory_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory_store.write(
        "a-fact", "Original.", "Original body.", directory=memory_root, evidence="checked by hand"
    )

    def explode(source: str, destination: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(memory_store.os, "replace", explode)
    with pytest.raises(MemoryStoreError, match="disk full"):
        memory_store.write(
            "a-fact", "Replacement.", "New body.", directory=memory_root, evidence="checked by hand"
        )

    monkeypatch.undo()
    assert [p.name for p in sorted(memory_root.iterdir())] == ["a-fact.md"]
    assert memory_store.discover(memory_root)["a-fact"].body() == "Original body.\n"


def test_forgetting_a_memory_removes_exactly_one(memory_root: Path) -> None:
    memory_store.write(
        "keep-me", "Keep.", "Body.", directory=memory_root, evidence="checked by hand"
    )
    memory_store.write(
        "drop-me", "Drop.", "Body.", directory=memory_root, evidence="checked by hand"
    )
    removed = memory_store.delete("drop-me", memory_root)
    assert removed.name == "drop-me.md"
    assert list(memory_store.discover(memory_root)) == ["keep-me"]


def test_forgetting_an_unknown_memory_names_what_exists(memory_root: Path) -> None:
    memory_store.write(
        "keep-me", "Keep.", "Body.", directory=memory_root, evidence="checked by hand"
    )
    with pytest.raises(MemoryStoreError, match=r"no memory named 'ghost'.*keep-me"):
        memory_store.delete("ghost", memory_root)


def test_the_memory_directory_follows_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("SLAB_MEMORY_DIR", raising=False)
    assert memory_store.memory_dir() == tmp_path / "xdg" / "slab" / "memory"

    monkeypatch.setenv("SLAB_MEMORY_DIR", str(tmp_path / "elsewhere"))
    assert memory_store.memory_dir() == tmp_path / "elsewhere"
    # The override is what the sandbox exports, so writes must land there.
    memory_store.write("a-fact", "A fact.", "Body.", evidence="checked by hand")
    assert (tmp_path / "elsewhere" / "a-fact.md").is_file()
    assert list(memory_store.discover()) == ["a-fact"]


def test_a_home_relative_override_expands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLAB_MEMORY_DIR", "~/memories")
    assert memory_store.memory_dir() == Path(os.path.expanduser("~/memories"))


def test_the_catalog_block_lists_one_line_per_memory(memory_root: Path) -> None:
    memory_store.write(
        "b-fact", "The second fact.", "Body.", directory=memory_root, evidence="checked by hand"
    )
    memory_store.write(
        "a-fact", "The first fact.", "Body.", directory=memory_root, evidence="checked by hand"
    )
    block = memory_store.catalog_block(memory_store.discover(memory_root))
    lines = block.splitlines()
    assert lines[0] == "# Memory"
    assert lines[-2:] == ["- a-fact: The first fact.", "- b-fact: The second fact."]
    assert "recall" in block and "remember" in block


def test_empty_store_still_carries_the_remember_doctrine() -> None:
    """A machine with no memories yet must still be told this surface exists
    and when to write to it — otherwise the agent has no reason to reach for
    the remember tool the first time it solves a machine-level blocker."""
    block = memory_store.catalog_block({})
    assert block.startswith("# Memory")
    assert "remember" in block
    assert "No machine facts recorded on this machine yet" in block
    assert not any(line.startswith("- ") for line in block.splitlines())


def test_the_file_stays_readable_by_the_person_editing_it(memory_root: Path) -> None:
    """Dates unquoted, no YAML anchors: a memory is edited by hand sometimes."""
    memory_store.write(
        "a-fact", "A fact.", "Body.", agent="pi", directory=memory_root, evidence="checked by hand"
    )
    text = (memory_root / "a-fact.md").read_text(encoding="utf-8")
    stamp = memory_store.discover(memory_root)["a-fact"].created
    assert f"created: {stamp}\n" in text
    assert f"updated: {stamp}\n" in text  # not collapsed into an alias
    assert "&id" not in text and "*id" not in text


def test_a_multiline_description_becomes_one_line(memory_root: Path) -> None:
    memory_store.write(
        "wrapped",
        "A description that the writer\n  broke across lines.",
        "Body.",
        directory=memory_root,
        evidence="checked by hand",
    )
    memory = memory_store.discover(memory_root)["wrapped"]
    assert memory.description == "A description that the writer broke across lines."


def test_bodies_with_frontmatter_markers_survive_the_round_trip(memory_root: Path) -> None:
    body = "The fix:\n\n---\n\nRun it twice: once for 'x: y' and once for \"z\".\n"
    memory_store.write(
        "tricky", "Quotes and rules.", body, directory=memory_root, evidence="checked by hand"
    )
    assert memory_store.discover(memory_root)["tricky"].body() == body


# -- version stamps ----------------------------------------------------------


def test_a_stamp_survives_the_round_trip_as_text(memory_root: Path) -> None:
    written = memory_store.write(
        "grace-gpu-growth",
        "gracemaker needs TF_FORCE_GPU_ALLOW_GROWTH on the GPU nodes.",
        "Without it TensorFlow grabs the whole card and the second fit fails.",
        against={"gracemaker": "1.10", "slab-stack": "0.1.0"},
        directory=memory_root,
        evidence="checked by hand",
    )
    text = written.path.read_text(encoding="utf-8")
    assert "against:\n  gracemaker: '1.10'\n  slab-stack: 0.1.0\n" in text
    again = memory_store.discover(memory_root)["grace-gpu-growth"]
    assert again.against == {"gracemaker": "1.10", "slab-stack": "0.1.0"}
    assert again.provenance().endswith(
        ", against gracemaker 1.10, slab-stack 0.1.0, evidence: checked by hand"
    )


def test_a_replacement_carries_its_own_stamp(memory_root: Path) -> None:
    memory_store.write(
        "a-fact", "About gracemaker.", "Body.", against={"gracemaker": "0.5.2"},
        directory=memory_root,
        evidence="checked by hand",
    )
    memory_store.write(
        "a-fact", "About nothing stamped.", "Body.", directory=memory_root,
        evidence="checked by hand",
    )
    again = memory_store.discover(memory_root)["a-fact"]
    assert again.against == {}
    assert "against:" not in again.path.read_text(encoding="utf-8")


def test_a_hand_typed_stamp_reads_as_text(memory_root: Path) -> None:
    (memory_root / "mp-release.md").write_text(
        "---\ndescription: The mp snapshot lacks alloys.\nagainst:\n  mp: 2024.11\n"
        "  slab-stack: 1\n---\nOnly elements and binaries.\n",
        encoding="utf-8",
    )
    memory = memory_store.discover(memory_root)["mp-release"]
    assert memory.against == {"mp": "2024.11", "slab-stack": "1"}


@pytest.mark.parametrize(
    "stamp",
    ["against: 0.6.0\n", "against:\n  - gracemaker\n", "against:\n  gracemaker: [0, 6]\n"],
)
def test_a_malformed_stamp_is_refused(memory_root: Path, stamp: str) -> None:
    (memory_root / "bad-stamp.md").write_text(
        f"---\ndescription: d\n{stamp}---\nBody.\n", encoding="utf-8"
    )
    with pytest.raises(MemoryStoreError, match="'against'"):
        memory_store.discover(memory_root)


def test_stamp_names_only_the_software_the_text_mentions() -> None:
    live = {"atomsk": "0.13.1", "gracemaker": "0.6.0", "qe": "7.3", "slab-stack": "0.1.0"}
    assert memory_store.stamp("pw.x hangs on the login node.", live) == {"qe": "7.3"}
    assert memory_store.stamp("Set it in slab.toml.", live) == {"slab-stack": "0.1.0"}
    assert memory_store.stamp("A GRACE fit needs a GPU.", live) == {"gracemaker": "0.6.0"}
    # Whole words only: 'atomskit' is not atomsk, and nothing here names slab.
    assert memory_store.stamp("atomskit and the vLLM cache.", live) == {}


def test_drift_is_the_catalog_note_and_nothing_else(memory_root: Path) -> None:
    memory_store.write(
        "grace-gpu", "gracemaker needs X.", "Body.", against={"gracemaker": "0.5.2"},
        directory=memory_root,
        evidence="checked by hand",
    )
    memory_store.write(
        "atomsk-path", "atomsk wants an absolute path.", "Body.", against={"atomsk": "0.13.1"},
        directory=memory_root,
        evidence="checked by hand",
    )
    memory_store.write(
        "vllm-cache", "vLLM refuses a big batch.", "Body.", directory=memory_root,
        evidence="checked by hand",
    )
    memories = memory_store.discover(memory_root)

    same = {"gracemaker": "0.5.2", "atomsk": "0.13.1"}
    unchanged = memory_store.catalog_block(memories, live=same)
    assert "changed since" not in unchanged

    block = memory_store.catalog_block(memories, live={"gracemaker": "0.6.0"})
    lines = block.splitlines()
    assert (
        "- atomsk-path: atomsk wants an absolute path. "
        "[changed since: atomsk was 0.13.1, not found now]"
    ) in lines
    assert (
        "- grace-gpu: gracemaker needs X. [changed since: gracemaker was 0.5.2, now 0.6.0]"
    ) in lines
    assert "- vllm-cache: vLLM refuses a big batch." in lines
    # Without a live map nothing is judged, so nothing is flagged.
    assert "changed since" not in memory_store.catalog_block(memories)


# -- evidence, versions, review ----------------------------------------------


def test_a_memory_without_evidence_is_refused(memory_root: Path) -> None:
    with pytest.raises(MemoryStoreError, match="a memory needs evidence"):
        memory_store.write("gpu-build", "The gpu build's command works.", "It ran.",
                           directory=memory_root)
    with pytest.raises(MemoryStoreError, match="a memory needs evidence"):
        memory_store.write("gpu-build", "The gpu build's command works.", "It ran.",
                           evidence="   ", directory=memory_root)
    assert list(memory_root.iterdir()) == []
    with pytest.raises(MemoryStoreError, match="over the 500-character limit"):
        memory_store.write("gpu-build", "d", "b", evidence="x" * 501, directory=memory_root)


def test_an_unverified_memory_is_stamped_and_marked(memory_root: Path) -> None:
    written = memory_store.write(
        "scratch-quota", "The scratch filesystem here fills at 80 percent.",
        "Seen in one probe.", unverified=True, directory=memory_root,
    )
    assert written.unverified is True and written.evidence is None
    assert "unverified: true\n" in written.path.read_text(encoding="utf-8")
    assert written.provenance().endswith("no evidence recorded")
    block = memory_store.catalog_block(memory_store.discover(memory_root))
    assert (
        "- scratch-quota: The scratch filesystem here fills at 80 percent. [unverified]"
        in block.splitlines()
    )

    checked = memory_store.write(
        "scratch-quota", "The scratch filesystem here fills at 80 percent.",
        "Writes fail above it.",
        evidence="run 01k2x7abcd completed after the sweep", directory=memory_root,
    )
    assert checked.unverified is False
    assert "unverified" not in checked.path.read_text(encoding="utf-8")
    assert checked.provenance().endswith("evidence: run 01k2x7abcd completed after the sweep")


def test_a_memory_with_no_evidence_reads_as_unverified(memory_root: Path) -> None:
    """A file from before the rule, or written by hand, is a claim until confirmed."""
    (memory_root / "old-claim.md").write_text(
        "---\ndescription: tables are empty on cache hits.\ncreated: 2026-09-01\n---\nSo.\n",
        encoding="utf-8",
    )
    (memory_root / "flagged.md").write_text(
        "---\ndescription: d\nevidence: run 01k2x7abcd\nunverified: true\n---\nSo.\n",
        encoding="utf-8",
    )
    memories = memory_store.discover(memory_root)
    assert memories["old-claim"].unverified is True
    assert memories["flagged"].unverified is True  # the flag outranks the evidence
    (memory_root / "bad.md").write_text(
        "---\ndescription: d\nunverified: maybe\n---\nSo.\n", encoding="utf-8"
    )
    with pytest.raises(MemoryStoreError, match="'unverified' must be true or false"):
        memory_store.discover(memory_root)


def test_reusing_a_name_keeps_the_version_it_replaces(memory_root: Path) -> None:
    first = memory_store.write(
        "newton-order", "newton on must precede read_data.", "From a probe file.",
        unverified=True, directory=memory_root,
    )
    assert first.replaced is None
    assert memory_store.versions("newton-order", memory_root) == []

    second = memory_store.write(
        "newton-order", "newton order does not matter.", "Both orders ran.",
        evidence="runs 01k2x7abcd and 01k2x7efgh", directory=memory_root,
    )
    assert second.replaced is not None and second.replaced.is_file()
    (kept,) = memory_store.versions("newton-order", memory_root)
    assert kept.path == second.replaced
    assert kept.body == "From a probe file."
    assert kept.description == "newton on must precede read_data."
    assert kept.unverified is True and kept.evidence is None
    # The history is not a memory: the catalog still holds one.
    assert list(memory_store.discover(memory_root)) == ["newton-order"]

    memory_store.write(
        "newton-order", "Third.", "Third body.", evidence="e", directory=memory_root
    )
    bodies = [v.body for v in memory_store.versions("newton-order", memory_root)]
    assert bodies == ["Both orders ran.", "From a probe file."]  # newest first


def test_the_history_keeps_a_bounded_number_of_versions(
    memory_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(memory_store, "MAX_VERSIONS", 3)
    for index in range(6):
        memory_store.write("a-fact", "A fact.", f"Body {index}.", evidence="e",
                           directory=memory_root)
    bodies = [v.body for v in memory_store.versions("a-fact", memory_root)]
    assert bodies == ["Body 4.", "Body 3.", "Body 2."]


def test_restore_puts_an_earlier_version_back_and_keeps_the_rejected_one(
    memory_root: Path,
) -> None:
    memory_store.write("a-fact", "Right.", "The right body.", evidence="e",
                       directory=memory_root)
    wrong = memory_store.write("a-fact", "Wrong.", "The wrong body.", unverified=True,
                               directory=memory_root)
    assert wrong.replaced is not None
    restored = memory_store.restore("a-fact", wrong.replaced, memory_root)
    assert restored.body() == "The right body.\n" and restored.description == "Right."
    assert memory_store.versions("a-fact", memory_root)[0].body == "The wrong body."

    other = memory_store.write("b-fact", "B.", "B body.", evidence="e", directory=memory_root)
    with pytest.raises(MemoryStoreError, match="is not a kept version"):
        memory_store.restore("b-fact", wrong.replaced, memory_root)
    assert other.path.is_file()


def test_forgetting_a_memory_forgets_its_history(memory_root: Path) -> None:
    memory_store.write("a-fact", "One.", "One.", evidence="e", directory=memory_root)
    memory_store.write("a-fact", "Two.", "Two.", evidence="e", directory=memory_root)
    assert memory_store.versions("a-fact", memory_root)
    memory_store.delete("a-fact", memory_root)
    assert memory_store.versions("a-fact", memory_root) == []
    # A new memory under the old name starts with no history.
    fresh = memory_store.write("a-fact", "Three.", "Three.", evidence="e", directory=memory_root)
    assert fresh.replaced is None


def test_review_lists_the_unverified_and_the_drifted(memory_root: Path) -> None:
    memory_store.write("checked", "Checked.", "Body.", evidence="run 01k2x7abcd",
                       against={"lammps": "22Jul2025"}, directory=memory_root)
    memory_store.write("claim", "A claim.", "Body.", unverified=True, directory=memory_root)
    memory_store.write("drifted", "gracemaker needs X.", "Body.", evidence="run 01k2x7efgh",
                       against={"gracemaker": "0.5.2"}, directory=memory_root)
    memory_store.write("both", "Both.", "Body.", unverified=True,
                       against={"gracemaker": "0.5.2"}, directory=memory_root)
    live = {"lammps": "22Jul2025", "gracemaker": "0.6.0"}
    listed = memory_store.needs_review(memory_store.discover(memory_root), live)
    assert [(m.name, reasons) for m, reasons in listed] == [
        ("both", ["unverified", "gracemaker was 0.5.2, now 0.6.0"]),
        ("claim", ["unverified"]),
        ("drifted", ["gracemaker was 0.5.2, now 0.6.0"]),
    ]


def test_run_ids_are_read_out_of_the_evidence() -> None:
    evidence = "run 01k2x7abcdefghjkmnpqrstv failed; see 01k2x7ab too; 5000000 steps"
    assert memory_store.run_ids(evidence) == ["01k2x7abcdefghjkmnpqrstv", "01k2x7ab"]
    assert memory_store.run_ids("checked by hand") == []


def test_a_persons_confirmation_is_stamped_and_a_later_write_drops_it(
    memory_root: Path,
) -> None:
    checked = memory_store.write(
        "scratch-quota", "The scratch filesystem fills at 80 percent.", "Seen.",
        evidence="checked by hand", confirmed=date(2026, 9, 17), directory=memory_root,
    )
    assert checked.confirmed == "2026-09-17"
    assert "confirmed: 2026-09-17\n" in checked.path.read_text(encoding="utf-8")
    assert "confirmed by a person on 2026-09-17" in checked.provenance()
    assert memory_store.discover(memory_root)["scratch-quota"].confirmed == "2026-09-17"

    # An agent's rewrite is a new claim, so the person's stamp goes with the
    # version it confirmed.
    rewritten = memory_store.write(
        "scratch-quota", "The scratch filesystem fills at 90 percent.", "Seen again.",
        agent="pi", evidence="run 01k2x7abcd", directory=memory_root,
    )
    assert rewritten.confirmed is None
    assert "confirmed:" not in rewritten.path.read_text(encoding="utf-8")


# -- kinds, outages, and documented behaviour --------------------------------


def test_a_memory_carries_its_kind_and_an_unmarked_file_is_a_build_memory(
    memory_root: Path,
) -> None:
    written = memory_store.write(
        "gpu-bandwidth", "The GPUs here share one link.", "So two fits contend.",
        evidence="run 01k2x7abcd", kind="resource", directory=memory_root,
    )
    assert written.kind == "resource"
    assert "kind: resource\n" in written.path.read_text(encoding="utf-8")

    (memory_root / "from-before.md").write_text(
        "---\ndescription: An older fact.\ncreated: 2026-09-01\n---\nThe fact.\n",
        encoding="utf-8",
    )
    found = memory_store.discover(memory_root)
    assert found["from-before"].kind == "build"
    assert found["from-before"].expires_at is None
    # A build memory says nothing about kinds in its file, so nothing moves.
    assert "kind:" not in found["gpu-bandwidth"].path.read_text(encoding="utf-8").replace(
        "kind: resource", ""
    )


def test_an_unknown_kind_is_refused_on_the_way_in_and_out(memory_root: Path) -> None:
    with pytest.raises(MemoryStoreError, match="not a kind of memory"):
        memory_store.write("a-fact", "A fact.", "Body.", evidence="e", kind="rumour",
                           directory=memory_root)
    (memory_root / "odd.md").write_text(
        "---\ndescription: A fact.\nkind: rumour\n---\nBody.\n", encoding="utf-8"
    )
    with pytest.raises(MemoryStoreError, match="'kind' must be one of"):
        memory_store.discover(memory_root)


def test_an_outage_expires_a_week_out_and_names_its_host(memory_root: Path) -> None:
    written = memory_store.write(
        "device-init-fails", "One node refuses to initialise its GPUs.",
        "Every launch there dies before the first step.",
        evidence="run 01k2x7abcd", kind="outage", where="n1", directory=memory_root,
    )
    today = datetime.now(UTC).date()
    assert written.expires_at == (today + timedelta(days=memory_store.OUTAGE_DAYS)).isoformat()
    assert written.where == "n1"
    assert written.outage_note().startswith("outage recorded ")
    assert "on n1;" in written.outage_note()
    assert not written.expired()


def test_only_an_outage_may_expire(memory_root: Path) -> None:
    with pytest.raises(MemoryStoreError, match="only an outage expires"):
        memory_store.write("a-fact", "A fact.", "Body.", evidence="e",
                           expires_at="2026-10-01", directory=memory_root)
    with pytest.raises(MemoryStoreError, match="not a date"):
        memory_store.write("a-fact", "A fact.", "Body.", evidence="e", kind="outage",
                           expires_at="next tuesday", directory=memory_root)


def test_the_catalog_drops_an_expired_outage_and_review_lists_it(memory_root: Path) -> None:
    memory_store.write(
        "device-init-fails", "One node refuses to initialise its GPUs.", "Body.",
        evidence="run 01k2x7abcd", kind="outage", where="n1",
        expires_at=_utc_today() - timedelta(days=1), directory=memory_root,
    )
    memory_store.write("a-build-fact", "A build fact.", "Body.", evidence="run 01k2x7efgh",
                       directory=memory_root)
    found = memory_store.discover(memory_root)
    assert found["device-init-fails"].expired()

    block = memory_store.catalog_block(found)
    assert "- a-build-fact: A build fact." in block
    assert "device-init-fails" not in block

    listed = memory_store.needs_review(found, {})
    assert [(m.name, reasons) for m, reasons in listed] == [
        ("device-init-fails", [f"expired outage, recorded {_utc_today().isoformat()}"]),
    ]


def test_a_live_outage_carries_its_host_and_expiry_into_the_catalog(memory_root: Path) -> None:
    memory_store.write(
        "device-init-fails", "One node refuses to initialise its GPUs.", "Body.",
        evidence="run 01k2x7abcd", kind="outage", where="n1", directory=memory_root,
    )
    line = memory_store.catalog_block(memory_store.discover(memory_root)).splitlines()[-1]
    assert line.startswith("- device-init-fails: One node refuses to initialise its GPUs.")
    assert "[outage recorded " in line and "on n1;" in line


def test_a_memory_that_restates_the_documented_codes_is_refused(memory_root: Path) -> None:
    with pytest.raises(MemoryStoreError) as raised:
        memory_store.write(
            "cna-codes",
            "The LAMMPS build here shifts the cna/atom codes.",
            "On this build 1 is hcp and 2 is fcc, not the documented mapping.",
            evidence="run 01k2x7abcd", directory=memory_root,
        )
    message = str(raised.value)
    assert "documented behaviour, not a fact about this machine" in message
    assert "two-phase-melting section 3" in message
    assert memory_store.discover(memory_root) == {}


@pytest.mark.parametrize(
    ("description", "body", "skill"),
    [
        ("dilate all in fix nph.", "dilate all remaps every atom's position.",
         "two-phase-melting section 2"),
        ("fix_modify on a thermostat.", "fix_modify energy yes is refused; use econserve.",
         "lammps-scripting section 8"),
        ("velocity create here.", "velocity all create sets the temperature of the group.",
         "lammps-scripting section 3"),
        ("fix halt syntax.", "fix guard all halt 10 v_x > 5 error continue stops the run.",
         "lammps-scripting section 7"),
    ],
)
def test_each_documented_subject_is_refused_with_its_skill(
    memory_root: Path, description: str, body: str, skill: str
) -> None:
    with pytest.raises(MemoryStoreError, match=re.escape(skill)):
        memory_store.write("a-fact", description, body, evidence="run 01k2x7abcd",
                           directory=memory_root)


def test_a_crash_the_skills_do_not_document_is_recorded(memory_root: Path) -> None:
    written = memory_store.write(
        "cna-threads",
        "compute cna/atom segfaults above eight threads on this build.",
        "Run the compute on one thread until the build is replaced.",
        evidence="run 01k2x7abcd", directory=memory_root,
    )
    assert written.name == "cna-threads"


def test_every_documented_entry_names_a_skill_section_that_exists() -> None:
    from foundation.documented import DOCUMENTED
    from foundation.skills import discover_skills

    catalog = discover_skills(Path(__file__).resolve().parent)
    for entry in DOCUMENTED:
        assert entry.skill in catalog, f"{entry.command} names a skill that is gone"
        body = catalog[entry.skill].body()
        assert f"\n## {entry.section}. " in body, (
            f"{entry.skill} has no section {entry.section} for {entry.command}"
        )
        assert entry.command in body


def test_dry_run_ids_are_read_out_of_the_evidence() -> None:
    evidence = "dry-20260917-121314-ab12 kept the log; run 01k2x7abcd is still going"
    assert memory_store.dry_run_ids(evidence) == ["dry-20260917-121314-ab12"]
    assert memory_store.run_ids(evidence) == ["01k2x7abcd"]
