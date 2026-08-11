"""Content-addressed artifact storage (SHA-256, ``root/ab/cd/<hash>`` layout).

The artifact store holds *bytes only*, keyed by content hash. Which runs
reference which artifacts — and with what name, role, and recipe — lives in the
run database (:class:`~slab.models.ArtifactRef`). The two lifetimes are
independent by design: retention can drop bytes (``discard``) while every
reference, hash, and recipe survives, which is SLAB's hash-and-discard
provenance tier.

Storage properties:

* **Deduplicating** — identical content is stored once, whatever it is named.
* **Immutable** — stored blobs are read-only on disk; writes go through a
  temp file and an atomic rename, so concurrent writers of the same content
  are harmless.
* **Verifiable** — the filename *is* the SHA-256 of the content.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from collections.abc import Iterator
from pathlib import Path

from slab.errors import AmbiguousHashError, ArtifactNotFoundError

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX = re.compile(r"^[0-9a-f]+$")
_CHUNK = 1 << 20  # 1 MiB read chunks: hash/copy large files without loading them


def _validate_digest(digest: str) -> str:
    normalized = digest.lower()
    if not _HEX64.fullmatch(normalized):
        raise ValueError(
            f"not a valid artifact hash: {digest!r} (expected 64 hex characters, SHA-256)"
        )
    return normalized


class ArtifactStore:
    """Filesystem content-addressed store for artifact bytes.

    Args:
        root: Directory for the store (created, parents included, if missing).

    Examples:
        >>> import tempfile
        >>> cas = ArtifactStore(tempfile.mkdtemp())
        >>> digest = cas.put_bytes(b"relaxed structure")
        >>> cas.has(digest), cas.get(digest).read_bytes()
        (True, b'relaxed structure')
        >>> cas.discard(digest)
        True
        >>> cas.has(digest)  # bytes gone; any recorded hash+recipe survives in the run DB
        False
    """

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self._root = Path(root).expanduser()
        self._root.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"ArtifactStore({str(self._root)!r})"

    @property
    def root(self) -> Path:
        """The store's root directory."""
        return self._root

    # -- writes -----------------------------------------------------------------------

    def put(self, path: str | os.PathLike[str]) -> str:
        """Store the file at *path*; return its SHA-256 hash.

        Hashes and copies in a single streaming pass (safe for files larger
        than memory), so the stored bytes always match the returned hash even
        if the source file is being modified. Idempotent: content that is
        already stored is not copied again.

        Examples:
            >>> import pathlib, tempfile
            >>> tmp = pathlib.Path(tempfile.mkdtemp())
            >>> _ = (tmp / "POSCAR").write_text("Si2\\n")
            >>> cas = ArtifactStore(tmp / "cas")
            >>> digest = cas.put(tmp / "POSCAR")
            >>> cas.get(digest).read_text()
            'Si2\\n'
        """
        hasher = hashlib.sha256()
        fd, tmp_name = tempfile.mkstemp(dir=self._root, prefix=".incoming-")
        tmp = Path(tmp_name)
        try:
            with open(path, "rb") as src, os.fdopen(fd, "wb") as dst:
                while chunk := src.read(_CHUNK):
                    hasher.update(chunk)
                    dst.write(chunk)
            digest = hasher.hexdigest()
            self._ingest(tmp, digest)
        finally:
            tmp.unlink(missing_ok=True)
        return digest

    def put_bytes(self, data: bytes) -> str:
        """Store *data*; return its SHA-256 hash. Idempotent.

        Examples:
            >>> import tempfile
            >>> cas = ArtifactStore(tempfile.mkdtemp())
            >>> cas.put_bytes(b"x") == cas.put_bytes(b"x")  # deduplicated
            True
        """
        digest = hashlib.sha256(data).hexdigest()
        if not self.has(digest):
            fd, tmp_name = tempfile.mkstemp(dir=self._root, prefix=".incoming-")
            tmp = Path(tmp_name)
            try:
                with os.fdopen(fd, "wb") as dst:
                    dst.write(data)
                self._ingest(tmp, digest)
            finally:
                tmp.unlink(missing_ok=True)
        return digest

    def discard(self, digest: str) -> bool:
        """Drop the bytes for *digest*; return True if bytes were present.

        This is the hash-and-discard operation: only bytes disappear — every
        :class:`~slab.models.ArtifactRef` (hash, recipe, provenance) in the run
        database survives, and identical content can be ``put`` again later.

        Examples:
            >>> import tempfile
            >>> cas = ArtifactStore(tempfile.mkdtemp())
            >>> digest = cas.put_bytes(b"wavecar")
            >>> cas.discard(digest), cas.discard(digest)
            (True, False)
        """
        path = self._path_for(_validate_digest(digest))
        if not path.exists():
            return False
        path.chmod(0o644)
        path.unlink()
        for parent in (path.parent, path.parent.parent):
            try:
                parent.rmdir()  # prune now-empty fan-out dirs
            except OSError:
                break
        return True

    # -- reads ------------------------------------------------------------------------

    def get(self, digest: str) -> Path:
        """Return the path to the stored bytes for a full hash or unique prefix.

        The returned file is read-only; callers must copy, not modify.

        Raises:
            ArtifactNotFoundError: No bytes stored under this hash.
            AmbiguousHashError: The prefix matches several artifacts.

        Examples:
            >>> import tempfile
            >>> cas = ArtifactStore(tempfile.mkdtemp())
            >>> digest = cas.put_bytes(b"bands")
            >>> cas.get(digest[:12]).read_bytes()
            b'bands'
        """
        return self._path_for(self.resolve(digest))

    def has(self, digest: str) -> bool:
        """Return True if bytes for the (full) hash are currently stored.

        Examples:
            >>> import tempfile
            >>> cas = ArtifactStore(tempfile.mkdtemp())
            >>> cas.has("0" * 64)
            False
        """
        return self._path_for(_validate_digest(digest)).exists()

    def size(self, digest: str) -> int:
        """Return the size in bytes of a stored artifact.

        Raises:
            ArtifactNotFoundError: No bytes stored under this hash.

        Examples:
            >>> import tempfile
            >>> cas = ArtifactStore(tempfile.mkdtemp())
            >>> cas.size(cas.put_bytes(b"12345"))
            5
        """
        normalized = _validate_digest(digest)
        path = self._path_for(normalized)
        if not path.exists():
            raise ArtifactNotFoundError.for_hash(normalized)
        return path.stat().st_size

    def resolve(self, digest: str) -> str:
        """Resolve a full hash or unique prefix (>= 6 hex chars) to the full hash.

        Only *stored* artifacts can be resolved — this consults the filesystem,
        so bytes that were discarded no longer resolve.

        Raises:
            ArtifactNotFoundError: Nothing stored matches.
            AmbiguousHashError: The prefix matches several stored artifacts.
            ValueError: The prefix is shorter than 6 chars or not hex.

        Examples:
            >>> import tempfile
            >>> cas = ArtifactStore(tempfile.mkdtemp())
            >>> digest = cas.put_bytes(b"forces")
            >>> cas.resolve(digest[:8]) == digest
            True
        """
        normalized = digest.lower()
        if _HEX64.fullmatch(normalized):
            if not self._path_for(normalized).exists():
                raise ArtifactNotFoundError.for_hash(normalized)
            return normalized
        if len(normalized) < 6 or not _HEX.fullmatch(normalized):
            raise ValueError(f"artifact hash prefix must be >= 6 hex characters, got {digest!r}")
        fan_out = self._root / normalized[:2] / normalized[2:4]
        if not fan_out.is_dir():
            raise ArtifactNotFoundError.for_hash(normalized)
        matches = sorted(
            entry.name
            for entry in fan_out.iterdir()
            if entry.name.startswith(normalized) and _HEX64.fullmatch(entry.name)
        )
        if not matches:
            raise ArtifactNotFoundError.for_hash(normalized)
        if len(matches) > 1:
            raise AmbiguousHashError(normalized, matches)
        return matches[0]

    def hashes(self) -> Iterator[str]:
        """Iterate over the hashes of all stored artifacts.

        Examples:
            >>> import tempfile
            >>> cas = ArtifactStore(tempfile.mkdtemp())
            >>> digest = cas.put_bytes(b"data")
            >>> list(cas.hashes()) == [digest]
            True
        """
        for ab in sorted(self._root.iterdir()):
            if not (ab.is_dir() and len(ab.name) == 2 and _HEX.fullmatch(ab.name)):
                continue
            for cd in sorted(ab.iterdir()):
                if not (cd.is_dir() and len(cd.name) == 2 and _HEX.fullmatch(cd.name)):
                    continue
                for entry in sorted(cd.iterdir()):
                    if _HEX64.fullmatch(entry.name):
                        yield entry.name

    # -- internals --------------------------------------------------------------------

    def _path_for(self, digest: str) -> Path:
        return self._root / digest[:2] / digest[2:4] / digest

    def _ingest(self, tmp: Path, digest: str) -> None:
        """Move a fully-written temp file into place, read-only, atomically."""
        dest = self._path_for(digest)
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp.chmod(0o444)
        os.replace(tmp, dest)
