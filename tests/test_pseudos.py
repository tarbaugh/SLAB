"""Pseudopotential family tests — installs run against a mocked archive."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from conftest import _upf_bytes, make_family
from slab.pseudos import (
    PseudoFamilyError,
    family_cutoffs,
    family_digest,
    family_pseudos,
    find_family,
    install_sssp,
    list_families,
    pseudos_root,
    verify_family,
)


@pytest.fixture()
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SLAB_PSEUDOS", str(tmp_path / "pseudos"))
    return tmp_path / "pseudos"


# -- roots and discovery ---------------------------------------------------------------


def test_root_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SLAB_PSEUDOS", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert pseudos_root() == tmp_path / "xdg" / "slab" / "pseudos"
    monkeypatch.delenv("XDG_DATA_HOME")
    assert pseudos_root() == Path("~/.local/share").expanduser() / "slab" / "pseudos"
    monkeypatch.setenv("SLAB_PSEUDOS", str(tmp_path / "envroot"))
    assert pseudos_root() == tmp_path / "envroot"
    assert pseudos_root(tmp_path / "explicit") == tmp_path / "explicit"  # explicit wins


def test_list_and_find(root: Path) -> None:
    assert list_families() == []
    family, directory = make_family(root)
    make_family(root, name="SSSP/1.2.1/PBE/precision", symbols=("Cu",))
    names = [f.name for f, _ in list_families()]
    assert names == ["SSSP/1.2.1/PBE/precision", "SSSP/1.3.0/PBEsol/efficiency"]
    found, where = find_family("SSSP/1.3.0/PBEsol/efficiency")
    assert found.name == family.name
    assert where == directory


def test_find_family_version_prefix_git_style(root: Path) -> None:
    make_family(root)
    found, _ = find_family("SSSP/1.3/PBEsol/efficiency")
    assert found.name == "SSSP/1.3.0/PBEsol/efficiency"
    found, _ = find_family("SSSP/1/PBEsol/efficiency")  # any dot-boundary prefix
    assert found.name == "SSSP/1.3.0/PBEsol/efficiency"
    # but only at dot boundaries, and only in the version segment
    with pytest.raises(PseudoFamilyError, match="not installed"):
        find_family("SSSP/13/PBEsol/efficiency")
    with pytest.raises(PseudoFamilyError, match="not installed"):
        find_family("SSSP/1.3/PBEsol/eff")


def test_find_family_ambiguity_refused(root: Path) -> None:
    make_family(root, name="SSSP/1.3.0/PBEsol/efficiency")
    make_family(root, name="SSSP/1.3.1/PBEsol/efficiency")
    with pytest.raises(PseudoFamilyError, match="ambiguous"):
        find_family("SSSP/1.3/PBEsol/efficiency")
    found, _ = find_family("SSSP/1.3.1/PBEsol/efficiency")  # exact still works
    assert found.name == "SSSP/1.3.1/PBEsol/efficiency"


def test_missing_family_error_teaches_install(root: Path) -> None:
    with pytest.raises(PseudoFamilyError, match="slab pseudos install"):
        find_family("SSSP/1.3/PBEsol/efficiency")


def test_corrupt_manifest_is_loud(root: Path) -> None:
    _, directory = make_family(root)
    (directory / "family.json").write_text("{not json")
    with pytest.raises(PseudoFamilyError, match="invalid family manifest"):
        list_families()


def test_newer_layout_version_refused(root: Path) -> None:
    family, directory = make_family(root)
    payload = family.model_dump(mode="json")
    payload["layout_version"] = 99
    (directory / "family.json").write_text(json.dumps(payload))
    with pytest.raises(PseudoFamilyError, match="upgrade slab"):
        list_families()


# -- cutoffs, mappings, digests ---------------------------------------------------------


def test_family_cutoffs_elementwise_maxima(root: Path) -> None:
    family, _ = make_family(root)
    assert family_cutoffs(family, ["Si"]) == (30.0, 240.0)
    assert family_cutoffs(family, ["Si", "O", "Si"]) == (75.0, 600.0)


def test_family_cutoffs_accepts_atoms(root: Path) -> None:
    from ase.build import bulk

    family, _ = make_family(root)
    assert family_cutoffs(family, bulk("Si", "diamond", a=5.43)) == (30.0, 240.0)


def test_family_missing_element_is_loud(root: Path) -> None:
    family, _ = make_family(root)
    with pytest.raises(PseudoFamilyError, match="no pseudopotential for element 'W'"):
        family_cutoffs(family, ["Si", "W"])
    with pytest.raises(PseudoFamilyError, match="'W'"):
        family_pseudos(family, ["W"])


def test_family_pseudos_full_and_subset(root: Path) -> None:
    family, _ = make_family(root)
    assert family_pseudos(family) == {"O": "O.test.upf", "Si": "Si.test.upf"}
    assert family_pseudos(family, ["Si", "Si"]) == {"Si": "Si.test.upf"}


def test_family_digest_is_content_derived(root: Path) -> None:
    family, _ = make_family(root)
    same, _ = make_family(root / "elsewhere", name="SSSP/1.3.0/PBEsol/efficiency")
    assert family_digest(family) == family_digest(same)  # same content, any location
    other, _ = make_family(root / "other", symbols=("Si",))
    assert family_digest(family) != family_digest(other)


def test_verify_family_detects_tampering(root: Path) -> None:
    family, directory = make_family(root)
    assert verify_family(family, directory) == []
    (directory / "Si.test.upf").write_text("tampered")
    (directory / "O.test.upf").unlink()
    problems = verify_family(family, directory)
    assert any("mismatch" in p for p in problems)
    assert any("missing" in p for p in problems)


# -- installation against a mocked archive ----------------------------------------------


def _mock_archive(monkeypatch: pytest.MonkeyPatch, corrupt: bool = False) -> dict[str, int]:
    """Serve a fake Materials Cloud record through slab.pseudos' fetchers."""
    import slab.pseudos as pseudos

    payloads = {"Si": _upf_bytes("Si"), "O": _upf_bytes("O")}
    metadata = {
        symbol: {
            "filename": f"{symbol}.test.upf",
            "md5": hashlib.md5(payload).hexdigest(),
            "cutoff_wfc": 30.0 if symbol == "Si" else 75.0,
            "cutoff_rho": 240.0 if symbol == "Si" else 600.0,
        }
        for symbol, payload in payloads.items()
    }
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w:gz") as archive:
        for symbol, payload in payloads.items():
            content = b"corrupted!" if corrupt and symbol == "O" else payload
            info = tarfile.TarInfo(name=f"nested/{symbol}.test.upf")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))

    calls = {"json": 0, "download": 0}

    def fetch_json(url: str):
        calls["json"] += 1
        if url.endswith("versions/latest"):
            return {"id": "rec-test"}
        if url.endswith("/files"):
            return {
                "entries": [
                    {"key": "SSSP_1.3.0_PBEsol_efficiency.tar.gz"},
                    {"key": "SSSP_1.3.0_PBEsol_efficiency.json"},
                    {"key": "SSSP_1.2.1_PBEsol_efficiency.tar.gz"},
                    {"key": "SSSP_1.3.0_PBE_precision.tar.gz"},
                ]
            }
        if url.endswith(".json/content"):
            # pin the exact metadata URL construction, not just the suffix
            assert url.endswith("/files/SSSP_1.3.0_PBEsol_efficiency.json/content"), url
            return metadata
        raise AssertionError(f"unexpected json fetch: {url}")

    def download(url: str, destination: Path) -> None:
        calls["download"] += 1
        assert url.endswith("SSSP_1.3.0_PBEsol_efficiency.tar.gz/content")
        destination.write_bytes(tar_bytes.getvalue())

    monkeypatch.setattr(pseudos, "_fetch_json", fetch_json)
    monkeypatch.setattr(pseudos, "_download", download)
    return calls


def test_install_sssp_end_to_end(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_archive(monkeypatch)
    family, directory = install_sssp(version="1.3", functional="PBEsol", precision="efficiency")
    assert family.name == "SSSP/1.3.0/PBEsol/efficiency"  # patch resolved from the record
    assert directory == root / "SSSP_1.3.0_PBEsol_efficiency"
    assert sorted(family.elements) == ["O", "Si"]
    assert verify_family(family, directory) == []
    found, _ = find_family("SSSP/1.3/PBEsol/efficiency")
    assert found.name == family.name

    with pytest.raises(PseudoFamilyError, match="already installed"):
        install_sssp(version="1.3", functional="PBEsol", precision="efficiency")
    replaced, _ = install_sssp(
        version="1.3", functional="PBEsol", precision="efficiency", force=True
    )
    assert replaced.name == family.name


def test_install_sssp_checksum_mismatch_aborts_cleanly(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_archive(monkeypatch, corrupt=True)
    with pytest.raises(PseudoFamilyError, match="checksum mismatch"):
        install_sssp(version="1.3", functional="PBEsol", precision="efficiency")
    assert list_families() == []  # nothing half-landed


def test_install_sssp_unknown_configuration_lists_offers(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_archive(monkeypatch)
    with pytest.raises(PseudoFamilyError, match="record offers"):
        install_sssp(version="9.9", functional="PBEsol", precision="efficiency")


def test_land_family_failure_rolls_back_the_old_install(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """force-replace must never destroy the old family before the new one has
    landed: a failing rename rolls the old install back."""
    import os as _os

    import slab.pseudos as pseudos

    _family, destination = make_family(root)
    old_manifest = (destination / "family.json").read_text()
    staged = root / ".staging-test"
    staged.mkdir()
    (staged / "family.json").write_text("{}")

    real_rename = _os.rename

    def failing_rename(src, dst):
        if Path(src) == staged:
            raise OSError(28, "No space left on device")
        return real_rename(src, dst)

    monkeypatch.setattr(pseudos.os, "rename", failing_rename)
    with pytest.raises(PseudoFamilyError, match="could not land"):
        pseudos._land_family(staged, destination)
    assert destination.exists()
    assert (destination / "family.json").read_text() == old_manifest  # rolled back
    assert [f.name for f, _ in list_families()] == ["SSSP/1.3.0/PBEsol/efficiency"]


def test_install_sssp_garbage_archive_is_a_teaching_error(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An HTML error page served as the tar.gz must surface as a loud
    PseudoFamilyError, not a raw tarfile traceback."""
    import slab.pseudos as pseudos

    _mock_archive(monkeypatch)

    def garbage_download(url: str, destination: Path) -> None:
        destination.write_text("<html><body>502 Bad Gateway</body></html>")

    monkeypatch.setattr(pseudos, "_download", garbage_download)
    with pytest.raises(PseudoFamilyError, match=r"not a readable tar\.gz"):
        install_sssp(version="1.3", functional="PBEsol", precision="efficiency")
    assert list_families() == []


def test_pseudo_entry_refuses_path_traversal_filenames() -> None:
    from pydantic import ValidationError

    from slab.pseudos import PseudoEntry

    for hostile in ("sub/Si.upf", "../Si.upf", "/etc/passwd", ".."):
        with pytest.raises(ValidationError, match="bare name"):
            PseudoEntry(filename=hostile, md5="0" * 32, cutoff_wfc=30.0, cutoff_rho=240.0)


def test_list_families_skips_staging_debris(root: Path) -> None:
    make_family(root)
    debris = root / ".staging-crashed"
    debris.mkdir()
    (debris / "family.json").write_text("{not even json")
    assert [f.name for f, _ in list_families()] == ["SSSP/1.3.0/PBEsol/efficiency"]
