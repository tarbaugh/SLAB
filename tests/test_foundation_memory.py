"""The machine-memory store: round trips, refusals, and provenance."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from foundation import memory as memory_store
from foundation.errors import MemoryStoreError


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
    memory_store.write("a-fact", "A fact.", "The body.", directory=root)
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
        "a-fact", "Second reading.", "Body two.", agent="md-expert", directory=memory_root
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
    memory_store.write("real-fact", "Real.", "Body.", directory=memory_root)
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
        memory_store.write(name, description, body, directory=memory_root)
    assert list(memory_root.iterdir()) == []


def test_the_hundredth_memory_is_the_last_new_one(memory_root: Path) -> None:
    for index in range(memory_store.MAX_MEMORIES):
        memory_store.write(f"fact-{index:03d}", "A fact.", "Body.", directory=memory_root)
    with pytest.raises(MemoryStoreError, match="the limit"):
        memory_store.write("one-too-many", "A fact.", "Body.", directory=memory_root)
    # Updating one that already exists still works: consolidation is the way out.
    memory_store.write("fact-000", "Consolidated.", "Body.", directory=memory_root)
    assert len(memory_store.discover(memory_root)) == memory_store.MAX_MEMORIES


def test_a_failed_write_leaves_no_partial_file(
    memory_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory_store.write("a-fact", "Original.", "Original body.", directory=memory_root)

    def explode(source: str, destination: str) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(memory_store.os, "replace", explode)
    with pytest.raises(MemoryStoreError, match="disk full"):
        memory_store.write("a-fact", "Replacement.", "New body.", directory=memory_root)

    monkeypatch.undo()
    assert [p.name for p in sorted(memory_root.iterdir())] == ["a-fact.md"]
    assert memory_store.discover(memory_root)["a-fact"].body() == "Original body.\n"


def test_forgetting_a_memory_removes_exactly_one(memory_root: Path) -> None:
    memory_store.write("keep-me", "Keep.", "Body.", directory=memory_root)
    memory_store.write("drop-me", "Drop.", "Body.", directory=memory_root)
    removed = memory_store.delete("drop-me", memory_root)
    assert removed.name == "drop-me.md"
    assert list(memory_store.discover(memory_root)) == ["keep-me"]


def test_forgetting_an_unknown_memory_names_what_exists(memory_root: Path) -> None:
    memory_store.write("keep-me", "Keep.", "Body.", directory=memory_root)
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
    memory_store.write("a-fact", "A fact.", "Body.")
    assert (tmp_path / "elsewhere" / "a-fact.md").is_file()
    assert list(memory_store.discover()) == ["a-fact"]


def test_a_home_relative_override_expands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SLAB_MEMORY_DIR", "~/memories")
    assert memory_store.memory_dir() == Path(os.path.expanduser("~/memories"))


def test_the_catalog_block_lists_one_line_per_memory(memory_root: Path) -> None:
    memory_store.write("b-fact", "The second fact.", "Body.", directory=memory_root)
    memory_store.write("a-fact", "The first fact.", "Body.", directory=memory_root)
    block = memory_store.catalog_block(memory_store.discover(memory_root))
    lines = block.splitlines()
    assert lines[0] == "# Memory"
    assert lines[-2:] == ["- a-fact: The first fact.", "- b-fact: The second fact."]
    assert "recall" in block and "remember" in block
    assert memory_store.catalog_block({}) == ""


def test_a_multiline_description_becomes_one_line(memory_root: Path) -> None:
    memory_store.write(
        "wrapped",
        "A description that the writer\n  broke across lines.",
        "Body.",
        directory=memory_root,
    )
    memory = memory_store.discover(memory_root)["wrapped"]
    assert memory.description == "A description that the writer broke across lines."


def test_bodies_with_frontmatter_markers_survive_the_round_trip(memory_root: Path) -> None:
    body = "The fix:\n\n---\n\nRun it twice: once for 'x: y' and once for \"z\".\n"
    memory_store.write("tricky", "Quotes and rules.", body, directory=memory_root)
    assert memory_store.discover(memory_root)["tricky"].body() == body
