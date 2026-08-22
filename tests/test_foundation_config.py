"""Configuration as Foundation sees it: the workspace root and its origins."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from foundation.cli import app

runner = CliRunner()

def test_cli_broken_config_fails_workspace_commands_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "slab.toml").write_text("[paths\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SLAB_WORKSPACE", raising=False)
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 1
    assert "not valid TOML" in result.output
