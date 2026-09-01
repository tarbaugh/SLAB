"""fetch_structure: the traced route from a material id to an Atoms.

The task reads the configured snapshot only (``[builders.mp] root``), so
every test writes a project ``slab.toml`` and chdirs into it — the same
resolution a workflow script sees. The snapshot itself is the miniature
real one from ``conftest.build_mp_snapshot``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from conftest import build_mp_snapshot
from foundation import Workspace
from foundation.tasks import fetch_structure
from slab.errors import BuilderError, BuilderNotAvailableError


@pytest.fixture()
def ws(tmp_path: Path) -> Iterator[Workspace]:
    with Workspace(tmp_path / "ws") as workspace:
        yield workspace


def _point_config_at(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, snapshot: Path
) -> None:
    (tmp_path / "slab.toml").write_text(f'[builders.mp]\nroot = "{snapshot}"\n')
    monkeypatch.chdir(tmp_path)


@pytest.fixture()
def configured_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mp_snapshot: Path
) -> Path:
    _point_config_at(tmp_path, monkeypatch, mp_snapshot)
    return mp_snapshot


def test_fetch_returns_the_archived_structure(configured_snapshot: Path) -> None:
    atoms, info = fetch_structure("mp-149")
    assert atoms.get_chemical_formula() == "Si2"
    assert all(atoms.pbc)
    assert info["builder"] == "mp"
    assert info["material_id"] == "mp-149"
    assert info["release"] == "2025.11.1"
    assert info["cif_path"] == "cifs/00/mp-149.cif"
    assert info["source"] == "cif"
    assert (info["n_atoms"], info["formula"]) == (2, "Si2")


def test_traced_fetch_keeps_the_cif_it_consumed(
    ws: Workspace, configured_snapshot: Path
) -> None:
    with ws.start_run(name="fetch") as run:
        fetch_structure("mp-22862", label="rocksalt")
    (artifact,) = ws.runs.list_artifacts(run.id)
    assert artifact.name == "rocksalt.cif"
    kept = ws.artifacts.get(artifact.hash).read_text()
    assert kept == (configured_snapshot / "cifs/02/mp-22862.cif").read_text()


def test_identical_fetches_cache_and_a_new_release_misses(
    ws: Workspace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _point_config_at(tmp_path, monkeypatch, build_mp_snapshot(tmp_path / "snap-a"))
    with ws.start_run() as first:
        fetch_structure("mp-149")
    # The same release mounted at a different path is the same data: hit.
    _point_config_at(tmp_path, monkeypatch, build_mp_snapshot(tmp_path / "snap-b"))
    with ws.start_run() as second:
        fetch_structure("mp-149")
    # A newer release is different data, whatever the path: miss.
    _point_config_at(
        tmp_path,
        monkeypatch,
        build_mp_snapshot(tmp_path / "snap-c", release="2026.03.1"),
    )
    with ws.start_run() as third:
        fetch_structure("mp-149")
    assert ws.runs.list_tasks(first.id)[0].cache_hit is False
    assert ws.runs.list_tasks(second.id)[0].cache_hit is True
    assert ws.runs.list_tasks(third.id)[0].cache_hit is False


def test_absence_raises_inside_a_run_with_the_contract(
    ws: Workspace, configured_snapshot: Path
) -> None:
    with pytest.raises(BuilderError, match="no online fallback"), ws.start_run(name="doomed"):
        fetch_structure("mp-404")


def test_unconfigured_snapshot_names_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(BuilderNotAvailableError, match=r"\[builders.mp\] root"):
        fetch_structure("mp-149")


def test_fetch_feeds_relax_directly(configured_snapshot: Path) -> None:
    from foundation.tasks import relax

    atoms, _ = fetch_structure("mp-13")
    relaxed, info = relax(atoms, engine="lj", fmax=0.1)
    assert info["converged"] is True
    assert len(relaxed) == 1
