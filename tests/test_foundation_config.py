"""Configuration as Foundation sees it: the ``[workspace]`` table it owns."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from foundation._ops import resolve_root
from foundation.cli import app
from foundation.config import (
    FoundationConfig,
    WorkspaceConfig,
    config_value,
    load_config,
)
from slab.config import ConfigError
from slab.config import load_config as load_slab_config

runner = CliRunner()


# -- the table Foundation owns -----------------------------------------------


def test_no_files_yields_pure_defaults(tmp_path: Path) -> None:
    config = load_config(tmp_path)
    assert config == FoundationConfig()
    assert config.workspace.root is None


def test_workspace_root_is_read_from_the_shared_file(tmp_path: Path) -> None:
    (tmp_path / "slab.toml").write_text('[workspace]\nroot = "/scratch/ws"\n')
    assert load_config(tmp_path).workspace.root == "/scratch/ws"
    assert config_value("workspace.root", tmp_path) == "/scratch/ws"


def test_workspace_root_expands_home_and_variables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SLAB_TEST_WS", "/scratch/tom")
    (tmp_path / "slab.toml").write_text('[workspace]\nroot = "${SLAB_TEST_WS}/ws"\n')
    assert load_config(tmp_path).workspace.root == "/scratch/tom/ws"
    (tmp_path / "slab.toml").write_text('[workspace]\nroot = "~/ws"\n')
    assert load_config(tmp_path).workspace.root == str(Path("~/ws").expanduser())


def test_unset_variable_in_workspace_root_is_refused(tmp_path: Path) -> None:
    (tmp_path / "slab.toml").write_text('[workspace]\nroot = "/scratch/$SLAB_NO_SUCH_VAR/ws"\n')
    with pytest.raises(ConfigError, match=r"references \$SLAB_NO_SUCH_VAR"):
        load_config(tmp_path)


def test_a_typo_inside_workspace_is_refused_naming_the_file(tmp_path: Path) -> None:
    (tmp_path / "slab.toml").write_text('[workspace]\nrooot = "/scratch/ws"\n')
    with pytest.raises(ConfigError, match=r"workspace\.rooot"):
        load_config(tmp_path)


def test_tables_foundation_does_not_own_are_ignored_here(tmp_path: Path) -> None:
    """One file, three packages: a valid [hpc] must not trip Foundation."""
    (tmp_path / "slab.toml").write_text(
        '[workspace]\nroot = "/scratch/ws"\n\n[hpc]\ncluster = "delta"\n\n'
        '[agent]\nmodel = "llama3.1:8b"\n'
    )
    config = load_config(tmp_path)
    assert config.workspace.root == "/scratch/ws"
    assert not hasattr(config, "hpc")


# -- the move from paths.workspace -------------------------------------------


def test_the_old_paths_workspace_key_is_refused_with_a_pointer(tmp_path: Path) -> None:
    """The refusal is the migration; an unknown key would read as a typo."""
    (tmp_path / "slab.toml").write_text('[paths]\nworkspace = "/scratch/ws"\n')
    with pytest.raises(ConfigError, match=r"paths.workspace moved to \[workspace\] root"):
        load_slab_config(tmp_path)


def test_the_old_key_is_refused_even_when_the_new_one_is_present(tmp_path: Path) -> None:
    (tmp_path / "slab.toml").write_text(
        '[paths]\nworkspace = "/old/ws"\n\n[workspace]\nroot = "/new/ws"\n'
    )
    with pytest.raises(ConfigError, match=r"paths.workspace moved"):
        load_slab_config(tmp_path)


# -- resolution chain ---------------------------------------------------------


def test_resolve_root_uses_config_below_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "slab.toml").write_text('[workspace]\nroot = "/config/ws"\n')
    monkeypatch.chdir(project)
    monkeypatch.delenv("SLAB_WORKSPACE", raising=False)
    assert resolve_root(None) == Path("/config/ws")
    monkeypatch.setenv("SLAB_WORKSPACE", "/env/ws")
    assert resolve_root(None) == Path("/env/ws")  # env beats config
    assert resolve_root("/explicit/ws") == Path("/explicit/ws")


def test_workspace_default_is_dot_slab(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLAB_WORKSPACE", raising=False)
    assert resolve_root(None) == Path(".slab")


# -- CLI ----------------------------------------------------------------------


def test_cli_broken_config_fails_workspace_commands_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "slab.toml").write_text("[paths\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLAB_WORKSPACE", raising=False)
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 1
    assert "not valid TOML" in result.output


def test_cli_reads_the_workspace_root_from_the_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ws = tmp_path / "from-config"
    (tmp_path / "slab.toml").write_text(f'[workspace]\nroot = "{ws}"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLAB_WORKSPACE", raising=False)
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert ws.is_dir()


def test_cli_refuses_the_moved_key_rather_than_ignoring_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "slab.toml").write_text('[paths]\nworkspace = "/scratch/ws"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLAB_WORKSPACE", raising=False)
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 1
    assert "[workspace] root" in result.output


def test_workspace_config_model_forbids_unknown_keys() -> None:
    with pytest.raises(Exception, match="rooot"):
        WorkspaceConfig.model_validate({"rooot": "/x"})
