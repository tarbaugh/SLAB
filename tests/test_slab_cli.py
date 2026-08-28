"""SLAB CLI tests via typer's CliRunner: engines, pseudos, protocols."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from slab.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.startswith("slab ")


# -- engines ---------------------------------------------------------------------------


EMT_ENTRY = {
    "calculator": "ase.calculators.emt.EMT",
    "version": "ase-built-in",
    "description": "cluster-declared EMT for tests",
}


def _write_engines(tmp_path: Path, engines: dict, cluster: str = "delta") -> Path:
    path = tmp_path / "engines.json"
    path.write_text(json.dumps({"cluster": cluster, "engines": engines}))
    return path


def test_engines_list_without_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SLAB_ENGINES", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    result = runner.invoke(app, ["engines", "list"])
    assert result.exit_code == 0
    assert "built-in: emt, lammps, lj, qe, rootstock" in result.output
    assert "none configured" in result.output


def test_engines_list_with_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    registry = _write_engines(tmp_path, {"emt-cluster": EMT_ENTRY})
    monkeypatch.setenv("SLAB_ENGINES", str(registry))
    result = runner.invoke(app, ["engines", "list"])
    assert result.exit_code == 0
    assert "registry [delta]" in result.output
    assert "emt-cluster" in result.output
    assert "ase-built-in" in result.output
    assert "cluster-declared EMT" in result.output


def test_engines_list_invalid_registry_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bad = tmp_path / "engines.json"
    bad.write_text(json.dumps({"engines": {"qe": EMT_ENTRY}}))  # shadows a built-in
    result = runner.invoke(app, ["engines", "list", "--registry", str(bad)])
    assert result.exit_code == 1
    assert "built-in engine name" in result.output


def test_engines_verify(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sys as _sys

    registry = _write_engines(
        tmp_path,
        {
            "good": {**EMT_ENTRY, "probe": [_sys.executable, "-c", "pass"]},
            "bad": {**EMT_ENTRY, "probe": [_sys.executable, "-c", "raise SystemExit(2)"]},
        },
    )
    result = runner.invoke(app, ["engines", "verify", "--registry", str(registry)])
    assert result.exit_code == 1
    assert "[+] good" in result.output
    assert "[x] bad" in result.output
    assert "1/2 engines verified" in result.output

    ok_registry = _write_engines(tmp_path, {"good": EMT_ENTRY})
    ok = runner.invoke(app, ["engines", "verify", "--registry", str(ok_registry)])
    assert ok.exit_code == 0
    assert "1/1 engines verified" in ok.output


def test_engines_verify_without_registry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SLAB_ENGINES", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    result = runner.invoke(app, ["engines", "verify"])
    assert result.exit_code == 1
    assert "no engine registry configured" in result.output


def test_engines_list_shows_rootstock_checkpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("rootstock", reason="rootstock extra not installed")
    root = tmp_path / "rootstock-install"
    env_dir = root / "envs" / "fake-mace"
    env_dir.mkdir(parents=True)
    (env_dir / "env_source.py").write_text('CHECKPOINTS = {"fake-mace-checkpoint": "small"}\n')
    monkeypatch.delenv("SLAB_ENGINES", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ROOTSTOCK_ROOT", str(root))
    result = runner.invoke(app, ["engines", "list"])
    assert result.exit_code == 0
    assert "rootstock checkpoints (usable directly as engine=)" in result.output
    assert "fake-mace: fake-mace-checkpoint" in result.output


# -- pseudos and protocols ---------------------------------------------------------------


def test_pseudos_list_empty_and_populated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from conftest import make_family

    monkeypatch.setenv("SLAB_PSEUDOS", str(tmp_path / "pseudos"))
    empty = runner.invoke(app, ["pseudos", "list"])
    assert empty.exit_code == 0
    assert "no families installed" in empty.output

    make_family(tmp_path / "pseudos")
    populated = runner.invoke(app, ["pseudos", "list"])
    assert populated.exit_code == 0
    assert "SSSP/1.3.0/PBEsol/efficiency" in populated.output
    assert "2 elements" in populated.output


def test_pseudos_verify_happy_and_tampered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from conftest import make_family

    monkeypatch.setenv("SLAB_PSEUDOS", str(tmp_path / "pseudos"))
    _family, directory = make_family(tmp_path / "pseudos")
    ok = runner.invoke(app, ["pseudos", "verify", "SSSP/1.3/PBEsol/efficiency"])
    assert ok.exit_code == 0
    assert "all 2 files match" in ok.output

    (directory / "Si.test.upf").write_text("tampered")
    bad = runner.invoke(app, ["pseudos", "verify", "SSSP/1.3/PBEsol/efficiency"])
    assert bad.exit_code == 1
    assert "checksum mismatch" in bad.output


def test_pseudos_verify_unknown_family_teaches_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SLAB_PSEUDOS", str(tmp_path / "pseudos"))
    result = runner.invoke(app, ["pseudos", "verify", "SSSP/1.3/PBEsol/efficiency"])
    assert result.exit_code == 1
    assert "slab pseudos install" in result.output


def test_pseudos_install_refuses_unknown_kind() -> None:
    result = runner.invoke(app, ["pseudos", "install", "pseudo-dojo"])
    assert result.exit_code == 1
    assert "only 'sssp'" in result.output


def test_protocols_list_and_show() -> None:
    listed = runner.invoke(app, ["protocols", "list"])
    assert listed.exit_code == 0
    for name in ("fast", "balanced", "stringent"):
        assert name in listed.output

    shown = runner.invoke(app, ["protocols", "show", "balanced"])
    assert shown.exit_code == 0
    assert "kpoints_distance: 0.15" in shown.output
    assert "forc_conv_thr_ev_per_ang" in shown.output

    as_json = runner.invoke(app, ["protocols", "show", "balanced", "--json"])
    assert json.loads(as_json.output)["degauss"] == 0.02

    unknown = runner.invoke(app, ["protocols", "show", "extreme"])
    assert unknown.exit_code == 1
    assert "available: balanced, fast, stringent" in unknown.output


def test_engines_list_includes_protocols_and_families(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from conftest import make_family

    monkeypatch.delenv("SLAB_ENGINES", raising=False)
    monkeypatch.setenv("SLAB_PSEUDOS", str(tmp_path / "pseudos"))
    empty = runner.invoke(app, ["engines", "list"])
    assert empty.exit_code == 0
    assert "qe protocols: balanced, fast, stringent" in empty.output
    assert "pseudo families: none installed" in empty.output

    make_family(tmp_path / "pseudos")
    populated = runner.invoke(app, ["engines", "list"])
    assert "pseudo families: SSSP/1.3.0/PBEsol/efficiency" in populated.output
