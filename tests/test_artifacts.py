import hashlib
import stat
from pathlib import Path

import pytest

from slab import AmbiguousHashError, ArtifactNotFoundError, ArtifactStore

SAMPLE = b"Si2\n1.0\n2.7 2.7 0.0\n"
SAMPLE_HASH = hashlib.sha256(SAMPLE).hexdigest()


@pytest.fixture()
def cas(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "cas")


def _plant(cas: ArtifactStore, fake_hash: str, data: bytes = b"planted") -> Path:
    """Plant a blob under an arbitrary (fake) hash, bypassing put()."""
    path = cas.root / fake_hash[:2] / fake_hash[2:4] / fake_hash
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# -- put / get -------------------------------------------------------------------------


def test_put_returns_sha256_and_lays_out_fanout(cas: ArtifactStore, tmp_path: Path) -> None:
    src = tmp_path / "POSCAR"
    src.write_bytes(SAMPLE)
    digest = cas.put(src)
    assert digest == SAMPLE_HASH
    stored = cas.root / digest[:2] / digest[2:4] / digest
    assert stored.is_file()
    assert stored.read_bytes() == SAMPLE


def test_put_is_idempotent_and_deduplicates(cas: ArtifactStore, tmp_path: Path) -> None:
    a = tmp_path / "a.xyz"
    b = tmp_path / "b.xyz"  # same content, different name
    a.write_bytes(SAMPLE)
    b.write_bytes(SAMPLE)
    assert cas.put(a) == cas.put(b) == SAMPLE_HASH
    assert len(list(cas.hashes())) == 1


def test_put_bytes_roundtrip(cas: ArtifactStore) -> None:
    digest = cas.put_bytes(SAMPLE)
    assert digest == SAMPLE_HASH
    assert cas.get(digest).read_bytes() == SAMPLE


def test_put_large_file_streams(cas: ArtifactStore, tmp_path: Path) -> None:
    big = tmp_path / "trajectory.bin"
    data = b"\xab" * (3 << 20)  # 3 MiB: exercises the chunked read loop
    big.write_bytes(data)
    digest = cas.put(big)
    assert digest == hashlib.sha256(data).hexdigest()
    assert cas.size(digest) == len(data)


def test_stored_blobs_are_read_only(cas: ArtifactStore) -> None:
    digest = cas.put_bytes(SAMPLE)
    mode = cas.get(digest).stat().st_mode
    assert not (mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def test_no_incoming_temp_files_left_behind(cas: ArtifactStore, tmp_path: Path) -> None:
    src = tmp_path / "f"
    src.write_bytes(SAMPLE)
    cas.put(src)
    cas.put_bytes(b"other")
    assert not list(cas.root.glob(".incoming-*"))


# -- has / size / resolve --------------------------------------------------------------


def test_has(cas: ArtifactStore) -> None:
    assert not cas.has(SAMPLE_HASH)
    cas.put_bytes(SAMPLE)
    assert cas.has(SAMPLE_HASH)


def test_size_of_missing_raises(cas: ArtifactStore) -> None:
    with pytest.raises(ArtifactNotFoundError):
        cas.size("0" * 64)


def test_get_missing_raises_with_helpful_message(cas: ArtifactStore) -> None:
    with pytest.raises(ArtifactNotFoundError, match="discarded by retention"):
        cas.get("0" * 64)


def test_get_by_unique_prefix(cas: ArtifactStore) -> None:
    digest = cas.put_bytes(SAMPLE)
    assert cas.get(digest[:8]).read_bytes() == SAMPLE
    assert cas.resolve(digest[:8]) == digest


def test_uppercase_input_normalized(cas: ArtifactStore) -> None:
    digest = cas.put_bytes(SAMPLE)
    assert cas.has(digest.upper())
    assert cas.resolve(digest.upper()) == digest


def test_ambiguous_prefix_raises(cas: ArtifactStore) -> None:
    _plant(cas, "abcdef" + "0" * 58)
    _plant(cas, "abcdef" + "1" * 58)
    with pytest.raises(AmbiguousHashError, match="2 matches") as excinfo:
        cas.resolve("abcdef")
    assert len(excinfo.value.matches) == 2


def test_prefix_too_short_or_malformed_rejected(cas: ArtifactStore) -> None:
    cas.put_bytes(SAMPLE)
    with pytest.raises(ValueError, match=">= 6 hex"):
        cas.resolve(SAMPLE_HASH[:5])
    with pytest.raises(ValueError, match=">= 6 hex"):
        cas.resolve("not-hex!")


def test_prefix_with_no_fanout_dir_raises_not_found(cas: ArtifactStore) -> None:
    with pytest.raises(ArtifactNotFoundError):
        cas.resolve("deadbeef")


def test_prefix_fanout_exists_but_no_match(cas: ArtifactStore) -> None:
    _plant(cas, "abab" + "0" * 60)
    with pytest.raises(ArtifactNotFoundError):
        cas.resolve("ababff")


@pytest.mark.parametrize("bad", ["", "xyz", "0" * 63, "0" * 65, "g" * 64])
def test_invalid_full_hashes_rejected(cas: ArtifactStore, bad: str) -> None:
    with pytest.raises(ValueError):
        cas.has(bad)


# -- discard ---------------------------------------------------------------------------


def test_discard_drops_bytes_and_prunes_dirs(cas: ArtifactStore) -> None:
    digest = cas.put_bytes(SAMPLE)
    fan_out = cas.root / digest[:2]
    assert cas.discard(digest) is True
    assert not cas.has(digest)
    assert not fan_out.exists()  # empty ab/cd dirs pruned
    with pytest.raises(ArtifactNotFoundError):
        cas.get(digest)


def test_discard_missing_returns_false(cas: ArtifactStore) -> None:
    assert cas.discard("0" * 64) is False


def test_put_after_discard_restores(cas: ArtifactStore) -> None:
    digest = cas.put_bytes(SAMPLE)
    cas.discard(digest)
    assert cas.put_bytes(SAMPLE) == digest
    assert cas.has(digest)


def test_discard_keeps_sibling_blobs(cas: ArtifactStore) -> None:
    doomed = "cafe" + "0" * 60
    sibling = "cafe" + "1" * 60  # same ab/cd fan-out dirs
    _plant(cas, doomed)
    _plant(cas, sibling)
    assert cas.discard(doomed) is True
    assert cas.has(sibling)


# -- enumeration -----------------------------------------------------------------------


def test_hashes_lists_exactly_stored_set(cas: ArtifactStore) -> None:
    digests = {cas.put_bytes(bytes([i])) for i in range(5)}
    assert set(cas.hashes()) == digests


def test_hashes_ignores_stray_files(cas: ArtifactStore) -> None:
    digest = cas.put_bytes(SAMPLE)
    (cas.root / "README").write_text("not a blob")
    (cas.root / "ab").mkdir(exist_ok=True)
    (cas.root / "ab" / "junk.txt").write_text("also not a blob")
    assert list(cas.hashes()) == [digest]


def test_empty_store(cas: ArtifactStore) -> None:
    assert list(cas.hashes()) == []


def test_repr_contains_root(cas: ArtifactStore) -> None:
    assert "cas" in repr(cas)
