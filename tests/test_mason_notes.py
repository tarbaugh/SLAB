"""The software-notes context provider: selection, overrides, and the prompt gate."""

from pathlib import Path

import pytest

from mason.config import AgentConfig
from mason.notes import _CANONICAL, enabled_notes, note_text, notes_block
from mason.prompts import system_messages
from mason.session import MasonSession
from slab.config import HpcConfig, SlabConfig


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


def test_selection_follows_the_config_tables() -> None:
    assert enabled_notes(SlabConfig()) == ("rootstock", "emt", "lj")
    cfg = SlabConfig.model_validate(
        {"engines": {"qe": {"command": "srun pw.x"}, "lammps": {"command": "lmp"}}}
    )
    assert enabled_notes(cfg) == ("rootstock", "qe", "lammps", "emt", "lj")
    served = SlabConfig.model_validate({"engines": {"rootstock": {"cluster": "delta"}}})
    assert "rootstock" in enabled_notes(served)
    snapshot = SlabConfig.model_validate({"builders": {"mp": {"root": "/data/mp"}}})
    assert enabled_notes(snapshot) == ("rootstock", "emt", "lj", "mp")


def test_a_user_note_replaces_the_packaged_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    xdg = tmp_path / "xdg"
    (xdg / "slab" / "notes").mkdir(parents=True)
    (xdg / "slab" / "notes" / "rootstock.md").write_text(
        "Our rootstock install serves mace-mp-0-medium only.\n"
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    assert note_text("rootstock") == "Our rootstock install serves mace-mp-0-medium only."
    block = notes_block(SlabConfig())
    assert "mace-mp-0-medium only" in block
    # A sentence from the packaged note must not survive the override.
    assert "worker subprocess" not in block
    # Other notes still come from the package.
    assert "effective-medium theory" in block


def test_the_system_prompt_carries_the_notes_by_default(tmp_path: Path) -> None:
    (content,) = [m["content"] for m in system_messages(_session(tmp_path))]
    assert "# Software notes" in content
    assert "## rootstock" in content
    # No [engines.qe] table in this project, so no qe note.
    assert "## qe" not in content


def test_a_configured_engine_brings_its_note(tmp_path: Path) -> None:
    (tmp_path / "slab.toml").write_text('[engines.qe]\ncommand = "pw.x"\n')
    (content,) = [m["content"] for m in system_messages(_session(tmp_path))]
    assert "## qe" in content
    assert "qe_protocol_options" in content


def test_a_configured_snapshot_brings_the_mp_note(tmp_path: Path) -> None:
    (tmp_path / "slab.toml").write_text('[builders.mp]\nroot = "/data/mp"\n')
    (content,) = [m["content"] for m in system_messages(_session(tmp_path))]
    assert "## mp" in content
    assert "Absence is absence" in content
    assert "(snapshot release, material_id)" in content


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
