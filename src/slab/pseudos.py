"""Pseudopotential families: aiida-pseudo's serving pattern, without AiiDA.

`aiida-pseudo <https://github.com/aiidateam/aiida-pseudo>`_ manages
pseudopotentials as installable *families* — a curated, versioned set of one
file per element, shipped with per-element recommended cutoffs. SLAB adopts
the pattern wholesale:

* **Install once, reference by name.** ``slab pseudos install sssp`` fetches
  an SSSP family from the official Materials Cloud archive, verifies every
  file against the published MD5 checksums, and lands it under a local root.
  From then on ``pseudo_family="SSSP/1.3.0/PBEsol/efficiency"`` works in the
  ``qe`` engine's ``calculator_options`` — no ``pseudo_dir``, no per-element
  file picking.
* **Cutoffs are family metadata, not folklore.** Each family records the
  recommended ``ecutwfc``/``ecutrho`` per element (in Ry, straight from the
  SSSP metadata); :func:`family_cutoffs` takes element-wise maxima over a
  structure — exactly the arithmetic AiiDA's protocols use.
* **Content-derived identity.** :func:`family_digest` hashes the per-element
  checksums, so a family's cache identity is its *contents*, portable across
  machines and roots — unlike a bare ``pseudo_dir`` path.

Layout: one directory per family under the root (``$SLAB_PSEUDOS``, else
``$XDG_DATA_HOME/slab/pseudos``, else ``~/.local/share/slab/pseudos``),
holding the ``.upf`` files plus a ``family.json`` manifest
(:class:`PseudoFamily`). The manifest carries a ``layout_version``; newer
layouts are refused loudly, mirroring the engine registry.

Only SSSP families install today. PseudoDojo is deliberately deferred: its
archives are served over plain HTTP (aiida-pseudo disables TLS verification
to fetch them), and slab will not silently download physics inputs over an
unverified channel. Point ``pseudo_dir=`` at your own PseudoDojo files
instead.

Family versions resolve git-style: ``SSSP/1.3/PBEsol/efficiency`` finds an
installed ``SSSP/1.3.0/PBEsol/efficiency`` as long as exactly one ``1.3.x``
is installed — ambiguity is refused, never guessed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from slab.errors import SlabError

LAYOUT_VERSION = 1
PSEUDOS_ENV_VAR = "SLAB_PSEUDOS"
_DEFAULT_SUBDIR = Path("slab") / "pseudos"

# The SSSP record on the Materials Cloud archive (doi:10.24435/materialscloud:rz-77).
# The 'latest' endpoint resolves to the newest record id, whose file listing
# names every SSSP_{version}_{functional}_{precision} archive + metadata json.
_SSSP_RECORD_LATEST = "https://archive.materialscloud.org/api/records/e67tc-6b197/versions/latest"
_HTTP_TIMEOUT_S = 120


class PseudoEntry(BaseModel):
    """One element's pseudopotential within a family.

    Cutoffs are the family's recommended values for this element, in the
    family's ``cutoff_unit`` (Ry for SSSP) — the numbers protocol expansion
    takes maxima over.
    """

    model_config = ConfigDict(frozen=True)

    filename: str = Field(min_length=1)
    md5: str = Field(min_length=32, max_length=32)
    cutoff_wfc: float = Field(gt=0)
    cutoff_rho: float = Field(gt=0)

    @model_validator(mode="after")
    def _filename_is_a_bare_name(self) -> PseudoEntry:
        # Filenames join onto directories all over this module (and end up in
        # pw.x inputs); a separator or '..' would escape the family directory.
        # Refused at validation, so neither a poisoned metadata JSON nor a
        # hand-copied manifest can traverse.
        if Path(self.filename).name != self.filename or self.filename in (".", ".."):
            raise ValueError(f"pseudopotential filename must be a bare name: {self.filename!r}")
        return self


class PseudoFamily(BaseModel):
    """A family's manifest (``family.json``) — its identity and metadata.

    Examples:
        >>> family = PseudoFamily.model_validate({
        ...     "name": "SSSP/1.3.0/PBEsol/efficiency",
        ...     "elements": {"Si": {"filename": "Si.upf", "md5": "0" * 32,
        ...                          "cutoff_wfc": 30.0, "cutoff_rho": 240.0}},
        ... })
        >>> family.elements["Si"].cutoff_wfc
        30.0
    """

    model_config = ConfigDict(frozen=True)

    layout_version: int = 1
    name: str = Field(min_length=1)
    kind: Literal["sssp"] = "sssp"
    cutoff_unit: Literal["Ry"] = "Ry"
    source: str = ""
    elements: dict[str, PseudoEntry] = Field(min_length=1)

    @model_validator(mode="after")
    def _layout_supported(self) -> PseudoFamily:
        if self.layout_version > LAYOUT_VERSION:
            raise ValueError(
                f"pseudopotential family {self.name!r} declares layout_version "
                f"{self.layout_version}, newer than this slab supports ({LAYOUT_VERSION}); "
                f"upgrade slab to read it"
            )
        return self


class PseudoFamilyError(SlabError):
    """A pseudopotential family is missing, ambiguous, corrupt, or uninstallable."""


def pseudos_root(root: str | os.PathLike[str] | None = None) -> Path:
    """The directory families install into.

    Resolution: explicit *root* > ``$SLAB_PSEUDOS`` > ``[paths] pseudos``
    from :mod:`slab.config` > ``$XDG_DATA_HOME/slab/pseudos`` >
    ``~/.local/share/slab/pseudos``. The directory need not exist yet
    (install creates it).
    """
    if root is not None:
        return Path(root).expanduser()
    from_env = os.environ.get(PSEUDOS_ENV_VAR)
    if from_env:
        return Path(from_env).expanduser()
    from slab.config import config_value

    configured = config_value("paths.pseudos")
    if configured:
        return Path(configured).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path("~/.local/share").expanduser()
    return base / _DEFAULT_SUBDIR


def family_dir_name(name: str) -> str:
    """Directory name for a family: its name with path separators flattened.

    Examples:
        >>> family_dir_name("SSSP/1.3.0/PBEsol/efficiency")
        'SSSP_1.3.0_PBEsol_efficiency'
    """
    return name.replace("/", "_")


def list_families(root: str | os.PathLike[str] | None = None) -> list[tuple[PseudoFamily, Path]]:
    """Every installed family under the root, as ``(manifest, directory)`` pairs.

    A directory with an unreadable or invalid ``family.json`` is an error,
    never silently skipped — a corrupt family must surface, not degrade to
    "family unknown".
    """
    base = pseudos_root(root)
    if not base.is_dir():
        return []
    found: list[tuple[PseudoFamily, Path]] = []
    for candidate in sorted(base.iterdir()):
        if candidate.name.startswith("."):  # .staging-*/.retired-* install debris
            continue
        manifest = candidate / "family.json"
        if not manifest.is_file():
            continue
        try:
            family = PseudoFamily.model_validate(json.loads(manifest.read_text()))
        except Exception as e:
            raise PseudoFamilyError(
                f"invalid family manifest {manifest}: {e} "
                f"(reinstall with 'slab pseudos install' or remove the directory)"
            ) from e
        found.append((family, candidate))
    return found


def find_family(
    name: str, root: str | os.PathLike[str] | None = None
) -> tuple[PseudoFamily, Path]:
    """Locate an installed family by name, git-style on the version segment.

    An exact name matches directly; otherwise the version segment is treated
    as a prefix (``SSSP/1.3/PBEsol/efficiency`` finds an installed
    ``SSSP/1.3.0/PBEsol/efficiency``). Zero or several matches raise with the
    installed inventory and the install command — never a guess.
    """
    installed = list_families(root)
    exact = [(f, p) for f, p in installed if f.name == name]
    if len(exact) == 1:
        return exact[0]
    matches = [(f, p) for f, p in installed if _version_prefix_match(name, f.name)]
    if len(matches) == 1:
        return matches[0]
    inventory = ", ".join(f.name for f, _ in installed) or "none"
    if len(matches) > 1:
        raise PseudoFamilyError(
            f"pseudopotential family {name!r} is ambiguous: matches "
            f"{[f.name for f, _ in matches]}; use the full version"
        )
    raise PseudoFamilyError(
        f"pseudopotential family {name!r} is not installed (installed: {inventory}). "
        f"Install SSSP families with e.g. "
        f"'slab pseudos install sssp --version 1.3 --functional PBEsol --precision efficiency'"
    )


def _version_prefix_match(requested: str, installed: str) -> bool:
    """True when the names differ only by a version-segment prefix (1.3 vs 1.3.0)."""
    req = requested.split("/")
    inst = installed.split("/")
    if len(req) != len(inst):
        return False
    for index, (r, i) in enumerate(zip(req, inst, strict=True)):
        if r == i:
            continue
        if index == 1 and i.startswith(r + "."):  # the version segment
            continue
        return False
    return True


def family_pseudos(family: PseudoFamily, symbols: Any = None) -> dict[str, str]:
    """The ``pseudopotentials=`` mapping a family provides.

    With *symbols* (an ``ase.Atoms`` or an iterable of element symbols), the
    mapping covers exactly those elements and a missing element is a loud
    error; without, the family's full mapping is returned.
    """
    if symbols is None:
        return {symbol: entry.filename for symbol, entry in sorted(family.elements.items())}
    return {
        symbol: _entry(family, symbol).filename for symbol in dict.fromkeys(_symbols_of(symbols))
    }


def family_cutoffs(family: PseudoFamily, symbols: Any) -> tuple[float, float]:
    """Recommended ``(ecutwfc, ecutrho)`` for a structure, in the family's unit.

    Element-wise maxima over the elements present — the same arithmetic
    AiiDA's ``get_recommended_cutoffs`` uses. There is no fixed dual: SSSP
    mixes norm-conserving/ultrasoft/PAW pseudos, so ``cutoff_rho`` is a
    stored per-element recommendation.

    Examples:
        >>> family = PseudoFamily.model_validate({
        ...     "name": "SSSP/1.3.0/PBEsol/efficiency",
        ...     "elements": {
        ...         "Si": {"filename": "Si.upf", "md5": "0" * 32,
        ...                "cutoff_wfc": 30.0, "cutoff_rho": 240.0},
        ...         "O": {"filename": "O.upf", "md5": "1" * 32,
        ...               "cutoff_wfc": 75.0, "cutoff_rho": 600.0},
        ...     },
        ... })
        >>> family_cutoffs(family, ["Si", "O", "Si"])
        (75.0, 600.0)
    """
    entries = [_entry(family, symbol) for symbol in dict.fromkeys(_symbols_of(symbols))]
    if not entries:
        raise PseudoFamilyError("family_cutoffs needs at least one element symbol")
    return (
        max(entry.cutoff_wfc for entry in entries),
        max(entry.cutoff_rho for entry in entries),
    )


def family_digest(family: PseudoFamily) -> str:
    """Short content-derived identity: a hash over the per-element checksums.

    Two installs of the same family bytes share a digest regardless of where
    they live — the portable cache identity ``describe_engine`` stamps.
    """
    payload = "\n".join(
        f"{symbol}:{entry.md5}" for symbol, entry in sorted(family.elements.items())
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def verify_family(family: PseudoFamily, directory: Path) -> list[str]:
    """Re-hash every file against the manifest; returns problems (empty = ok)."""
    problems: list[str] = []
    for symbol, entry in sorted(family.elements.items()):
        path = directory / entry.filename
        if not path.is_file():
            problems.append(f"{symbol}: missing file {entry.filename}")
            continue
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest != entry.md5:
            problems.append(f"{symbol}: checksum mismatch for {entry.filename}")
    return problems


def _symbols_of(symbols: Any) -> list[str]:
    get = getattr(symbols, "get_chemical_symbols", None)
    return list(get()) if callable(get) else list(symbols)


def _entry(family: PseudoFamily, symbol: str) -> PseudoEntry:
    entry = family.elements.get(symbol)
    if entry is None:
        raise PseudoFamilyError(
            f"family {family.name!r} has no pseudopotential for element {symbol!r} "
            f"({len(family.elements)} elements covered)"
        )
    return entry


# -- SSSP installation -------------------------------------------------------------------


def install_sssp(
    version: str = "1.3",
    functional: str = "PBEsol",
    precision: str = "efficiency",
    root: str | os.PathLike[str] | None = None,
    force: bool = False,
) -> tuple[PseudoFamily, Path]:
    """Install an SSSP family from the official Materials Cloud archive.

    Fetches the archive record's file listing, resolves *version* to the
    newest matching patch release (``1.3`` -> ``1.3.0``), downloads the
    metadata JSON and the ``.tar.gz`` of UPF files, and verifies **every**
    file against the published MD5 checksums — one mismatch aborts the whole
    install, nothing half-lands. The family is then addressable as
    ``SSSP/{patch}/{functional}/{precision}``.

    SSSP is distributed by Materials Cloud under CC-BY-4.0; individual UPF
    files carry their upstream libraries' licenses (recorded per element in
    the SSSP metadata).

    Args:
        version: Minor or full version (``"1.3"`` or ``"1.3.0"``).
        functional: ``"PBE"`` or ``"PBEsol"``.
        precision: ``"efficiency"`` or ``"precision"`` (SSSP's two variants;
            aiida-pseudo calls this the family's protocol).
        root: Install root override (default: :func:`pseudos_root`).
        force: Replace an already-installed family instead of refusing.

    Returns:
        The installed family's manifest and directory.
    """
    listing = _fetch_json(f"{_SSSP_RECORD_LATEST}")
    record_id = listing.get("id") if isinstance(listing, dict) else None
    if not record_id:
        raise PseudoFamilyError(
            f"unexpected Materials Cloud response from {_SSSP_RECORD_LATEST}: no record id"
        )
    files_url = f"https://archive.materialscloud.org/api/records/{record_id}/files"
    names = _record_file_names(_fetch_json(files_url))

    patch = _resolve_patch(names, version, functional, precision)
    stem = f"SSSP_{patch}_{functional}_{precision}"
    name = f"SSSP/{patch}/{functional}/{precision}"
    base = pseudos_root(root)
    destination = base / family_dir_name(name)
    if destination.exists() and not force:
        raise PseudoFamilyError(
            f"family {name!r} is already installed at {destination} (use force/--force to replace)"
        )

    metadata_url = f"{files_url}/{stem}.json/content"
    archive_url = f"{files_url}/{stem}.tar.gz/content"
    metadata = _fetch_json(metadata_url)
    if not isinstance(metadata, dict) or not metadata:
        raise PseudoFamilyError(
            f"unexpected SSSP metadata from {metadata_url}: expected a non-empty "
            f"element mapping, got {type(metadata).__name__}"
        )

    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="slab-pseudos-") as tmp:
        archive_path = Path(tmp) / f"{stem}.tar.gz"
        _download(archive_url, archive_path)
        unpacked = Path(tmp) / "unpacked"
        unpacked.mkdir()
        _extract_archive(archive_path, unpacked, archive_url)

        elements: dict[str, PseudoEntry] = {}
        # Verified files stage into a dot-directory on the SAME filesystem as
        # the destination, so the landing step below is an atomic rename —
        # never a cross-device file-by-file copy that could half-land.
        staged = Path(tempfile.mkdtemp(prefix=".staging-", dir=base))
        try:
            for symbol, info in sorted(metadata.items()):
                try:
                    entry = PseudoEntry(
                        filename=info["filename"],
                        md5=info["md5"],
                        cutoff_wfc=float(info["cutoff_wfc"]),
                        cutoff_rho=float(info["cutoff_rho"]),
                    )
                except Exception as e:
                    raise PseudoFamilyError(
                        f"SSSP metadata for element {symbol!r} is malformed: {e}"
                    ) from e
                source_file = _find_unpacked(unpacked, entry.filename)
                digest = hashlib.md5(source_file.read_bytes()).hexdigest()
                if digest != entry.md5:
                    raise PseudoFamilyError(
                        f"checksum mismatch for {entry.filename} ({symbol}): archive does not "
                        f"match the published metadata — refusing to install {name!r}"
                    )
                shutil.copyfile(source_file, staged / entry.filename)
                elements[symbol] = entry

            family = PseudoFamily(name=name, kind="sssp", source=archive_url, elements=elements)
            # The manifest lands last: a directory without family.json is
            # never mistaken for an installed family.
            (staged / "family.json").write_text(
                json.dumps(family.model_dump(mode="json"), indent=1, sort_keys=True)
            )
            _land_family(staged, destination)
        except BaseException:
            shutil.rmtree(staged, ignore_errors=True)
            raise
    return family, destination


def _land_family(staged: Path, destination: Path) -> None:
    """Swap the staged directory into place atomically.

    An existing install (``force=True``) is renamed aside first and removed
    only after the new one has landed; if landing fails, the old install is
    renamed back. Both renames are same-filesystem, so no interruption can
    leave the destination half-written or missing.
    """
    retired: Path | None = None
    if destination.exists():
        retired = Path(tempfile.mkdtemp(prefix=".retired-", dir=destination.parent))
        retired.rmdir()  # os.rename needs the target name free
        os.rename(destination, retired)
    try:
        os.rename(staged, destination)
    except OSError as e:
        if retired is not None and not destination.exists():
            os.rename(retired, destination)  # roll the old install back
        raise PseudoFamilyError(f"could not land family at {destination}: {e}") from e
    if retired is not None:
        shutil.rmtree(retired, ignore_errors=True)


def _extract_archive(archive_path: Path, unpacked: Path, source_url: str) -> None:
    """Unpack the downloaded archive, refusing garbage and unsafe members loudly."""
    try:
        with tarfile.open(archive_path) as archive:
            archive.extractall(unpacked, filter="data")
    except TypeError as e:  # pragma: no cover - Python < 3.11.4 lacks filter=
        raise PseudoFamilyError(
            "installing families needs Python >= 3.11.4 (tarfile extraction filters); "
            "upgrade Python"
        ) from e
    except (tarfile.TarError, EOFError, OSError, ValueError) as e:
        raise PseudoFamilyError(
            f"the archive downloaded from {source_url} is not a readable tar.gz ({e}); "
            f"a proxy or captive portal may have served an error page — retry, or check "
            f"connectivity to archive.materialscloud.org"
        ) from e


def _resolve_patch(names: list[str], version: str, functional: str, precision: str) -> str:
    """Newest patch release among the record's files matching the request."""
    import re

    pattern = re.compile(
        rf"^SSSP_({re.escape(version)}(?:\.\d+)*)_{re.escape(functional)}_{re.escape(precision)}\.tar\.gz$"
    )
    patches = sorted(
        {m.group(1) for name in names if (m := pattern.match(name))},
        key=lambda v: [int(part) for part in v.split(".")],
    )
    if not patches:
        available = sorted({n for n in names if n.startswith("SSSP_") and n.endswith(".tar.gz")})
        raise PseudoFamilyError(
            f"no SSSP archive matches version={version!r} functional={functional!r} "
            f"precision={precision!r}; the record offers: {available}"
        )
    return patches[-1]


def _record_file_names(files_response: dict[str, Any]) -> list[str]:
    entries = files_response.get("entries", [])
    names = [entry.get("key", "") for entry in entries if isinstance(entry, dict)]
    if not names:
        raise PseudoFamilyError("unexpected Materials Cloud response: no file entries")
    return names


def _find_unpacked(unpacked: Path, filename: str) -> Path:
    """The extracted file for *filename* (archives may nest one directory)."""
    direct = unpacked / filename
    if direct.is_file():
        return direct
    matches = list(unpacked.rglob(filename))
    if len(matches) == 1:
        return matches[0]
    raise PseudoFamilyError(
        f"archive is missing {filename!r} promised by the metadata"
        if not matches
        else f"archive contains {len(matches)} copies of {filename!r}"
    )


def _fetch_json(url: str) -> Any:
    try:
        with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT_S) as response:
            return json.load(response)
    except Exception as e:
        raise PseudoFamilyError(f"could not fetch {url}: {e}") from e


def _download(url: str, destination: Path) -> None:
    try:
        with (
            urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT_S) as response,
            open(destination, "wb") as out,
        ):
            shutil.copyfileobj(response, out)
    except Exception as e:
        raise PseudoFamilyError(f"could not download {url}: {e}") from e
