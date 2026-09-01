"""SLAB CLI tests via typer's CliRunner: engines, pseudos, protocols."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from slab.cli import app

runner = CliRunner()


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


# -- the mp snapshot group ---------------------------------------------------------------


def _configure_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from conftest import build_mp_snapshot

    root = build_mp_snapshot(tmp_path / "mp-snapshot")
    (tmp_path / "slab.toml").write_text(f'[builders.mp]\nroot = "{root}"\n')
    monkeypatch.chdir(tmp_path)
    return root


def test_mp_info_unconfigured_fails_with_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["mp", "info"])
    assert result.exit_code == 1
    assert "[builders.mp] root" in result.output


def test_mp_info_reports_the_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _configure_snapshot(tmp_path, monkeypatch)
    result = runner.invoke(app, ["mp", "info"])
    assert result.exit_code == 0
    assert f"root: {root}" in result.output
    assert "release: 2025.11.1" in result.output
    assert "materials: 4" in result.output


def test_mp_search_filters_and_shows_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_snapshot(tmp_path, monkeypatch)
    result = runner.invoke(
        app,
        ["mp", "search", "-e", "Fe", "-f", "energy_above_hull__lte=0.05"],
    )
    assert result.exit_code == 0
    assert "material_id=mp-13" in result.output
    assert "mp-1271068" not in result.output
    as_json = runner.invoke(app, ["mp", "search", "-e", "Fe", "--json"])
    ids = {row["material_id"] for row in json.loads(as_json.output)}
    assert ids == {"mp-13", "mp-1271068"}


def test_mp_search_refuses_a_malformed_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_snapshot(tmp_path, monkeypatch)
    result = runner.invoke(app, ["mp", "search", "-f", "no-equals-sign"])
    assert result.exit_code == 1
    assert "key=value" in result.output


def test_mp_show_renders_the_record_and_absence_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_snapshot(tmp_path, monkeypatch)
    result = runner.invoke(app, ["mp", "show", "mp-22862"])
    assert result.exit_code == 0
    assert "formula_pretty: NaCl" in result.output
    assert "elements: Cl, Na" in result.output
    assert "cif_file: " in result.output
    missing = runner.invoke(app, ["mp", "show", "mp-404"])
    assert missing.exit_code == 1
    assert "no online fallback" in missing.output


def test_engines_list_names_the_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLAB_ENGINES", raising=False)
    _configure_snapshot(tmp_path, monkeypatch)
    result = runner.invoke(app, ["engines", "list"])
    assert result.exit_code == 0
    assert "mp snapshot: release 2025.11.1, 4 materials ('slab mp info')" in result.output


def test_engines_list_names_the_gracemaker_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SLAB_ENGINES", raising=False)
    bin_dir = tmp_path / "grace-bin"
    bin_dir.mkdir()
    for name, body in (("python", 'echo "0.5.1"'), ("gracemaker", 'echo training')):
        script = bin_dir / name
        script.write_text(f"#!/bin/sh\n{body}\n")
        script.chmod(0o755)
    (tmp_path / "slab.toml").write_text(
        f'[builders.gracemaker]\ncommand = "{bin_dir / "gracemaker"}"\n'
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["engines", "list"])
    assert result.exit_code == 0
    assert (
        f"gracemaker trainer: tensorpotential 0.5.1 via {bin_dir / 'gracemaker'}"
        in result.output
    )
