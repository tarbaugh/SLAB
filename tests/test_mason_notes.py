"""The software-notes context provider: selection, overrides, and the prompt gate."""

from pathlib import Path

import pytest

from mason.config import AgentConfig
from mason.notes import _CANONICAL, enabled_notes, note_text, notes_block
from mason.prompts import system_messages
from mason.session import MasonSession
from slab.config import EnginesConfig, HpcConfig


def _session(tmp_path: Path, agent: AgentConfig | None = None) -> MasonSession:
    return MasonSession(
        tmp_path,
        agent=agent or AgentConfig(),
        hpc=HpcConfig(),
        workspace_root=tmp_path / ".slab",
    )


def test_every_canonical_note_ships_and_is_substantive() -> None:
    root = Path(__file__).resolve().parent.parent / "src" / "mason" / "notes"
    shipped = sorted(p.stem for p in root.glob("*.md"))
    assert shipped == sorted(_CANONICAL)
    for name in _CANONICAL:
        assert len(note_text(name)) > 100, f"the {name} note is too thin to help"


def test_selection_follows_the_engines_tables() -> None:
    assert enabled_notes(EnginesConfig()) == ("mace", "emt", "lj")
    cfg = EnginesConfig.model_validate(
        {"qe": {"command": "srun pw.x"}, "lammps": {"command": "lmp"}}
    )
    assert enabled_notes(cfg) == ("mace", "qe", "lammps", "emt", "lj")
    served = EnginesConfig.model_validate({"rootstock": {"cluster": "delta"}})
    assert "rootstock" in enabled_notes(served)


def test_a_user_note_replaces_the_packaged_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xdg = tmp_path / "xdg"
    (xdg / "slab" / "notes").mkdir(parents=True)
    (xdg / "slab" / "notes" / "mace.md").write_text(
        "Our MACE runs on the H100 partition only.\n"
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    assert note_text("mace") == "Our MACE runs on the H100 partition only."
    block = notes_block(EnginesConfig())
    assert "H100 partition" in block
    assert "~/.cache/mace" not in block  # the packaged note is fully replaced
    # Other notes still come from the package.
    assert "effective-medium theory" in block


def test_the_system_prompt_carries_the_notes_by_default(tmp_path: Path) -> None:
    (content,) = [m["content"] for m in system_messages(_session(tmp_path))]
    assert "# Software notes" in content
    assert "## mace" in content
    # No [engines.qe] table in this project, so no qe note.
    assert "## qe" not in content


def test_a_configured_engine_brings_its_note(tmp_path: Path) -> None:
    (tmp_path / "slab.toml").write_text('[engines.qe]\ncommand = "pw.x"\n')
    (content,) = [m["content"] for m in system_messages(_session(tmp_path))]
    assert "## qe" in content
    assert "qe_protocol_options" in content


def test_software_notes_false_removes_the_block(tmp_path: Path) -> None:
    agent = AgentConfig(software_notes=False)
    (content,) = [m["content"] for m in system_messages(_session(tmp_path, agent))]
    assert "# Software notes" not in content


def test_working_bounds_ride_the_fence(tmp_path: Path) -> None:
    (fenced,) = [m["content"] for m in system_messages(_session(tmp_path))]
    assert "# Working bounds" in fenced
    assert "ask the user before you touch it" in fenced
    lifted = AgentConfig(file_scope="anywhere")
    (content,) = [m["content"] for m in system_messages(_session(tmp_path, lifted))]
    assert "# Working bounds" not in content


def test_roster_tables_cannot_override_software_notes() -> None:
    with pytest.raises(Exception, match="software_notes"):
        AgentConfig.model_validate({"roster": {"pi": {"software_notes": False}}})
