"""The mp builder seam: read-only queries over an offline snapshot.

Every test runs against a miniature but structurally real snapshot built by
``conftest.build_mp_snapshot``: real CIFs written by ASE, the ``materials``
and ``material_elements`` tables, key/value ``dataset_info``, ``units``, and
``manifest.json``. The contract under test comes from the snapshot handoff:
read-only, no online fallback, identity is (release, material_id), and a
stored ``cif_path`` must resolve below the snapshot root.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from conftest import build_mp_snapshot
from slab.errors import BuilderError, BuilderNotAvailableError
from slab.mp import (
    connect,
    describe_mp,
    get_material,
    mp_root,
    query_materials,
    search_materials,
    snapshot_info,
    structure_path,
)

# -- root resolution ----------------------------------------------------------


def test_unconfigured_root_is_refused_naming_the_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(BuilderNotAvailableError, match=r"\[builders.mp\] root"):
        mp_root()


def test_root_resolves_from_config_and_per_call_wins(
    tmp_path: Path, mp_snapshot: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "slab.toml").write_text(f'[builders.mp]\nroot = "{mp_snapshot}"\n')
    monkeypatch.chdir(tmp_path)
    assert mp_root() == mp_snapshot
    other = build_mp_snapshot(tmp_path / "other-snapshot")
    assert mp_root(other) == other


def test_a_directory_without_the_database_is_not_a_snapshot(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(BuilderError, match=r"metadata\.sqlite is missing"):
        mp_root(tmp_path / "empty")


# -- read-only ----------------------------------------------------------------


def test_the_connection_refuses_writes(mp_snapshot: Path) -> None:
    connection = connect(mp_snapshot)
    try:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("INSERT INTO dataset_info VALUES ('k', 'v')")
    finally:
        connection.close()


# -- provenance and identity --------------------------------------------------


def test_snapshot_info_reports_release_count_and_manifest(mp_snapshot: Path) -> None:
    info = snapshot_info(mp_snapshot)
    assert info["release"] == "2025.11.1"
    assert info["materials"] == 4
    assert info["manifest"]["license_labels"] == ["CC-BY-4.0"]
    assert {"key": "build_timestamp", "value": "2026-08-30T12:00:00Z"} in info[
        "dataset_info"
    ]


def test_release_survives_a_missing_manifest(tmp_path: Path) -> None:
    root = build_mp_snapshot(tmp_path / "snap", manifest=False)
    info = snapshot_info(root)
    assert "manifest" not in info
    assert info["release"] == "2025.11.1"


def test_describe_stamps_release_and_count_without_the_path(mp_snapshot: Path) -> None:
    identity = describe_mp(mp_snapshot)
    assert identity == {"builder": "mp", "release": "2025.11.1", "materials": 4}
    assert "root" not in identity


def test_describe_degrades_to_a_file_fingerprint_without_a_release(
    tmp_path: Path,
) -> None:
    root = build_mp_snapshot(tmp_path / "snap", release=None, manifest=False)
    identity = describe_mp(root)
    assert identity["release"] is None
    stat = (root / "metadata.sqlite").stat()
    assert identity["snapshot_fingerprint"] == [stat.st_size, stat.st_mtime_ns]


# -- search -------------------------------------------------------------------


def test_search_by_element_membership(mp_snapshot: Path) -> None:
    rows = search_materials({"elements": ["Fe"]}, root=mp_snapshot)
    assert {row["material_id"] for row in rows} == {"mp-13", "mp-1271068"}


def test_search_combines_elements_with_a_range_suffix(mp_snapshot: Path) -> None:
    rows = search_materials(
        {"elements": ["Fe"], "energy_above_hull__lte": 0.05}, root=mp_snapshot
    )
    assert [row["material_id"] for row in rows] == ["mp-13"]


def test_search_excludes_elements(mp_snapshot: Path) -> None:
    rows = search_materials(
        {"exclude_elements": ["Fe", "Cl"]}, columns=["material_id"], root=mp_snapshot
    )
    assert rows == [{"material_id": "mp-149"}]


def test_search_null_means_not_populated(mp_snapshot: Path) -> None:
    rows = search_materials({"band_gap": None}, root=mp_snapshot)
    assert [row["material_id"] for row in rows] == ["mp-22862"]
    rows = search_materials(
        {"band_gap__ne": None}, columns=["material_id"], root=mp_snapshot
    )
    assert len(rows) == 3


def test_search_orders_and_limits(mp_snapshot: Path) -> None:
    rows = search_materials(
        {},
        columns=["material_id", "energy_above_hull"],
        order_by="-energy_above_hull",
        root=mp_snapshot,
    )
    assert rows[0]["material_id"] == "mp-1271068"
    assert len(search_materials({}, limit=0, root=mp_snapshot)) == 1
    assert len(search_materials({}, limit=2, root=mp_snapshot)) == 2


def test_search_refuses_an_unknown_column_naming_the_schema(mp_snapshot: Path) -> None:
    with pytest.raises(BuilderError, match=r"'bandgap'.*band_gap"):
        search_materials({"bandgap": 1.0}, root=mp_snapshot)
    with pytest.raises(BuilderError, match=r"'nope'.*material_id"):
        search_materials({}, order_by="nope", root=mp_snapshot)
    with pytest.raises(BuilderError, match=r"'nope'.*material_id"):
        search_materials({}, columns=["nope"], root=mp_snapshot)


def test_search_refuses_bad_suffixes_elements_and_none_comparisons(
    mp_snapshot: Path,
) -> None:
    with pytest.raises(BuilderError, match="unknown filter suffix"):
        search_materials({"band_gap__within": 1.0}, root=mp_snapshot)
    with pytest.raises(BuilderError, match="not an element symbol"):
        search_materials({"elements": ["Fe; DROP"]}, root=mp_snapshot)
    with pytest.raises(BuilderError, match="NULL orders with nothing"):
        search_materials({"band_gap__lte": None}, root=mp_snapshot)


# -- one material -------------------------------------------------------------


def test_get_material_returns_row_elements_and_cif_path(mp_snapshot: Path) -> None:
    record = get_material("mp-22862", root=mp_snapshot)
    assert record["formula_pretty"] == "NaCl"
    assert record["elements"] == ["Cl", "Na"]
    assert record["band_gap"] is None
    cif = Path(record["cif_file"])
    assert cif.is_absolute() and cif.is_file()
    assert structure_path("mp-22862", root=mp_snapshot) == cif


def test_absence_is_absence_and_names_the_release(mp_snapshot: Path) -> None:
    with pytest.raises(BuilderError, match=r"release 2025\.11\.1") as excinfo:
        get_material("mp-404", root=mp_snapshot)
    assert "no online fallback" in str(excinfo.value)


def test_a_nonsense_material_id_is_refused_before_the_query(mp_snapshot: Path) -> None:
    with pytest.raises(BuilderError, match="does not look like a material id"):
        get_material("mp-149 OR 1=1", root=mp_snapshot)


def test_a_cif_path_escaping_the_root_is_refused(tmp_path: Path) -> None:
    root = build_mp_snapshot(
        tmp_path / "snap",
        extra_materials=(
            {"material_id": "mp-666", "cif_path": "../outside/mp-666.cif"},
        ),
    )
    with pytest.raises(BuilderError, match="escapes the snapshot root"):
        get_material("mp-666", root=root)


def test_a_listed_but_missing_cif_names_the_transfer(tmp_path: Path) -> None:
    root = build_mp_snapshot(
        tmp_path / "snap",
        extra_materials=({"material_id": "mp-777", "cif_path": "cifs/zz/mp-777.cif"},),
    )
    with pytest.raises(BuilderError, match="transferred completely"):
        structure_path("mp-777", root=root)


def test_a_record_with_no_cif_says_so(tmp_path: Path) -> None:
    root = build_mp_snapshot(
        tmp_path / "snap", extra_materials=({"material_id": "mp-888"},)
    )
    with pytest.raises(BuilderError, match="records no CIF"):
        structure_path("mp-888", root=root)
    assert "cif_file" not in get_material("mp-888", root=root)


# -- raw SQL ------------------------------------------------------------------


def test_query_runs_select_and_with(mp_snapshot: Path) -> None:
    result = query_materials(
        "SELECT material_id FROM materials WHERE is_magnetic = 1 "
        "ORDER BY material_id",
        root=mp_snapshot,
    )
    assert result["rows"] == [{"material_id": "mp-1271068"}, {"material_id": "mp-13"}]
    assert result["truncated"] is False
    result = query_materials(
        "-- stable count\nWITH s AS (SELECT * FROM materials WHERE is_stable = 1) "
        "SELECT count(*) AS n FROM s",
        root=mp_snapshot,
    )
    assert result["rows"] == [{"n": 3}]


def test_query_refuses_anything_but_select(mp_snapshot: Path) -> None:
    with pytest.raises(BuilderError, match="only read-only queries"):
        query_materials("DROP TABLE materials", root=mp_snapshot)
    with pytest.raises(BuilderError, match="only read-only queries"):
        query_materials("  ", root=mp_snapshot)
    with pytest.raises(BuilderError, match="snapshot query failed"):
        query_materials("SELECT 1; DROP TABLE materials", root=mp_snapshot)


def test_query_caps_rows_and_says_it_truncated(mp_snapshot: Path) -> None:
    result = query_materials("SELECT * FROM materials", limit=2, root=mp_snapshot)
    assert result["row_count"] == 2 and len(result["rows"]) == 2
    assert result["truncated"] is True


# -- against a real snapshot --------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("SLAB_TEST_MP_SNAPSHOT"),
    reason="set SLAB_TEST_MP_SNAPSHOT to a real snapshot root",
)
def test_real_snapshot_answers_end_to_end() -> None:
    root = os.environ["SLAB_TEST_MP_SNAPSHOT"]
    info = snapshot_info(root)
    assert info["materials"] > 0
    rows = search_materials(
        {"elements": ["Fe"]}, columns=["material_id", "cif_path"], limit=5, root=root
    )
    assert rows
    cif = structure_path(rows[0]["material_id"], root=root)
    from ase.io import read as ase_read

    atoms = ase_read(cif)
    assert len(atoms) > 0
