"""The ``slab`` front door: one command tree over all four packages.

These tests pin the composed surface, not the behavior behind it — each
verb's behavior is tested where it is implemented. What must hold here is
that the tree is exactly the planned one, that mounting re-uses the same
functions rather than copies, and that a command reached through the front
door actually answers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from foundation import cli as foundation_cli
from slab._version import __version__
from slab_stack.cli import _command_name, app

runner = CliRunner()

LIFECYCLE = {"run", "list", "show", "promote", "sessions"}
HOUSEKEEPING_COMMANDS = {"expire", "gc", "fast-forward", "purge"}
GROUPS = {"memory", "mason", "engines", "pseudos", "protocols", "hpc", "config"}


def test_the_command_tree_is_exactly_the_planned_one() -> None:
    commands = [_command_name(info) for info in app.registered_commands]
    assert sorted(commands) == sorted(LIFECYCLE | HOUSEKEEPING_COMMANDS | {"mcp"})
    assert len(set(commands)) == len(commands)
    groups = [info.name for info in app.registered_groups]
    assert sorted(groups) == sorted(GROUPS)  # type: ignore[type-var]


def test_every_lifecycle_verb_is_the_foundation_function_itself() -> None:
    front = {_command_name(info): info.callback for info in app.registered_commands}
    for info in foundation_cli.app.registered_commands:
        assert front[_command_name(info)] is info.callback


def test_help_panels_group_by_intent() -> None:
    panels = {_command_name(info): info.rich_help_panel for info in app.registered_commands}
    for name in LIFECYCLE:
        assert panels[name] == "Runs and lifecycle"
    for name in HOUSEKEEPING_COMMANDS:
        assert panels[name] == "Housekeeping"
    assert panels["mcp"] == "Integration"
    group_panels = {info.name: info.rich_help_panel for info in app.registered_groups}
    assert group_panels["mason"] == "The resident agent"
    assert group_panels["memory"] == "Housekeeping"
    for name in ("engines", "pseudos", "protocols", "hpc", "config"):
        assert group_panels[name] == "This machine"


def test_version_speaks_for_the_whole_stack() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"slab {__version__}"


def test_list_answers_through_the_front_door(tmp_path: Path) -> None:
    result = runner.invoke(app, ["list", "-w", str(tmp_path / "ws")])
    assert result.exit_code == 0
    assert "no runs" in result.output


def test_memory_answers_through_the_front_door(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLAB_MEMORY_DIR", str(tmp_path / "memory"))
    result = runner.invoke(app, ["memory", "list"])
    assert result.exit_code == 0
    assert "no memories recorded yet" in result.output


@pytest.mark.parametrize("group", sorted(GROUPS))
def test_every_group_renders_its_help(group: str) -> None:
    result = runner.invoke(app, [group, "--help"])
    assert result.exit_code == 0
