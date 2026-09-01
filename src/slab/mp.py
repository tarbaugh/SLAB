"""The offline Materials Project snapshot: local materials data, no network.

A snapshot is a read-only directory that a staging machine built from the
Materials Project and this machine merely mounts: ``metadata.sqlite`` (the
discovery and filtering interface), ``manifest.json`` (provenance),
``cifs/`` (interchange structure files, sharded, addressed by the relative
``cif_path`` each ``materials`` row stores), and further files this module
does not read (``metadata.parquet``, ``README_AGENT.md``, ``SHA256SUMS``).
It supplies structures and metadata, never energies, so it is a *builder*
in SLAB's vocabulary, not an engine: ``engine="mp"`` does not exist, and
nothing here builds a calculator. The snapshot root is configured once::

    [builders.mp]
    root = "/data/mp-snapshot"

Three rules from the snapshot's contract shape every function here. First,
there is no online fallback: a material absent from the snapshot is
reported as absent, never looked up. Second, a result's identity is the
pair ``(snapshot release, material_id)``, so :func:`describe_mp` stamps the
release into cache keys and deliberately drops the root path — the same
portability rule as pseudopotential families. Third, a stored ``cif_path``
must resolve below the snapshot root; one that escapes is refused loudly,
never resolved.

Everything opens the database read-only (a SQLite URI with ``mode=ro``,
plus ``PRAGMA query_only``), so no caller — including the agent's raw SQL
tool — can modify the distributed file. The traced task that turns a
material id into an ASE ``Atoms`` is ``foundation.tasks.fetch_structure``.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from slab.errors import BuilderError, BuilderNotAvailableError

#: Comparison suffixes the search filter DSL accepts, as ``column__<op>``.
_OPERATORS = {"lte": "<=", "gte": ">=", "lt": "<", "gt": ">", "ne": "!="}

_MATERIAL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ELEMENT_PATTERN = re.compile(r"^[A-Z][a-z]?$")
_MAX_ROWS = 500

#: Keys that may carry the snapshot's release string, in trust order, looked
#: for in the manifest first and the dataset_info table second.
_RELEASE_KEYS = (
    "database_release",
    "snapshot_release",
    "release",
    "database_version",
    "db_version",
    "version",
)


def mp_root(root: str | os.PathLike[str] | None = None) -> Path:
    """The snapshot root: per-call value, else ``[builders.mp] root``.

    Refuses when nothing is configured, and when the named directory does
    not hold a ``metadata.sqlite`` — a snapshot root without its database
    is a transfer or mount problem worth naming, not a query that finds
    nothing.
    """
    if root is None:
        configured = _mp_setting("root")
        if configured is None:
            raise BuilderNotAvailableError(
                "no Materials Project snapshot is configured on this machine. "
                "Set [builders.mp] root in slab.toml to the snapshot directory "
                "(the one holding metadata.sqlite, manifest.json, and cifs/)"
            )
        root = str(configured)
    resolved = Path(root).expanduser()
    if not (resolved / "metadata.sqlite").is_file():
        raise BuilderError(
            f"{resolved} is not a Materials Project snapshot root: "
            "metadata.sqlite is missing. A snapshot root holds "
            "metadata.sqlite, manifest.json, and the cifs/ tree — check the "
            "path, the mount, and whether the transfer completed"
        )
    return resolved


def connect(root: str | os.PathLike[str] | None = None) -> sqlite3.Connection:
    """A read-only connection to the snapshot's ``metadata.sqlite``.

    ``mode=ro`` refuses writes at the SQLite layer and ``PRAGMA query_only``
    refuses them again on the connection, so even raw SQL from the agent
    cannot modify the distributed file. The caller closes the connection
    (``contextlib.closing`` is the house idiom).
    """
    database = mp_root(root) / "metadata.sqlite"
    try:
        connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as e:
        raise BuilderError(f"cannot open {database} read-only: {e}") from e
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def snapshot_info(root: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Provenance and scale of the installed snapshot, from its own records.

    Merges ``manifest.json`` (kept whole under ``"manifest"``; a missing or
    malformed file is reported, not fatal — the database still answers),
    the ``dataset_info`` table, and a count of ``materials`` rows. The
    extracted ``"release"`` is the string workflows must report with every
    result, and it may be None when the snapshot records none.
    """
    resolved = mp_root(root)
    info: dict[str, Any] = {"root": str(resolved)}
    manifest_path = resolved / "manifest.json"
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                info["manifest"] = loaded
            else:
                info["manifest_error"] = "manifest.json is not a JSON object"
        except (OSError, ValueError) as e:
            info["manifest_error"] = f"cannot read manifest.json: {e}"
    with contextlib.closing(connect(resolved)) as connection:
        info["materials"] = _material_count(connection)
        info["dataset_info"] = _dataset_info(connection)
    info["release"] = _release(info)
    return info


def describe_mp(root: str | os.PathLike[str] | None = None) -> dict[str, object]:
    """Identity of the snapshot: provenance and cache identity in one.

    The release and the material count enter every ``fetch_structure``
    cache key, so installing a newer snapshot honestly invalidates cached
    structures, while the same release mounted at a different path on a
    different machine still hits — the root path is deliberately dropped,
    the same portability rule pseudopotential families follow. A snapshot
    that records no release degrades to the database file's size and mtime,
    the engines' honest fallback for an undetectable version.
    """
    info = snapshot_info(root)
    identity: dict[str, object] = {
        "builder": "mp",
        "release": info["release"],
        "materials": info["materials"],
    }
    if info["release"] is None:
        stat = (Path(str(info["root"])) / "metadata.sqlite").stat()
        identity["snapshot_fingerprint"] = [stat.st_size, stat.st_mtime_ns]
    return identity


def search_materials(
    filters: dict[str, Any] | None = None,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
    limit: int = 20,
    order_by: str | None = None,
    root: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Search the ``materials`` table with parameterized, schema-checked SQL.

    *filters* maps filter keys to values: ``"elements"`` (every listed
    element must be present) and ``"exclude_elements"`` reach the
    ``material_elements`` table; any other key is a ``materials`` column,
    bare for equality (``None`` matches SQL NULL) or suffixed
    ``__lte``/``__gte``/``__lt``/``__gt``/``__ne`` for comparisons. Unknown
    columns are refused with the real column list, so a wrong guess teaches
    the schema. *limit* is clamped to 1-500; *order_by* names a column,
    with a leading ``-`` for descending.
    """
    with contextlib.closing(connect(root)) as connection:
        known = _material_columns(connection)
        selected = _select_list(columns, known)
        where, parameters = _filter_clauses(filters or {}, known)
        order = _order_clause(order_by, known)
        clamped = max(1, min(int(limit), _MAX_ROWS))
        sql = f"SELECT {selected} FROM materials AS m{where}{order} LIMIT ?"
        try:
            rows = connection.execute(sql, [*parameters, clamped]).fetchall()
        except sqlite3.Error as e:
            raise BuilderError(f"snapshot query failed: {e}") from e
    return [dict(row) for row in rows]


def get_material(
    material_id: str, *, root: str | os.PathLike[str] | None = None
) -> dict[str, Any]:
    """One material's metadata record, its elements, and its resolved CIF path.

    The returned dict is the ``materials`` row plus ``"elements"`` (from
    ``material_elements``) and, when the row stores a ``cif_path`` that
    resolves below the snapshot root, ``"cif_file"`` with the absolute
    path. Absence raises: the snapshot is the only source, and there is no
    online fallback.
    """
    _guard_material_id(material_id)
    resolved_root = mp_root(root)
    with contextlib.closing(connect(resolved_root)) as connection:
        row = connection.execute(
            "SELECT * FROM materials WHERE material_id = ?", (material_id,)
        ).fetchone()
        if row is None:
            raise BuilderError(_absence_message(material_id, resolved_root))
        record = dict(row)
        record["elements"] = [
            element_row["element"]
            for element_row in connection.execute(
                "SELECT element FROM material_elements WHERE material_id = ? "
                "ORDER BY element",
                (material_id,),
            )
        ]
    cif_path = record.get("cif_path")
    if cif_path:
        record["cif_file"] = str(_resolve_cif(resolved_root, str(cif_path)))
    return record


def structure_path(
    material_id: str, *, root: str | os.PathLike[str] | None = None
) -> Path:
    """The validated absolute path of one material's CIF file.

    The stored ``cif_path`` must resolve below the snapshot root (a path
    that escapes is refused, never resolved) and the file must exist — a
    listed-but-missing CIF means the ``cifs/`` tree did not transfer
    completely, which is worth naming.
    """
    _guard_material_id(material_id)
    resolved_root = mp_root(root)
    with contextlib.closing(connect(resolved_root)) as connection:
        row = connection.execute(
            "SELECT cif_path FROM materials WHERE material_id = ?", (material_id,)
        ).fetchone()
    if row is None:
        raise BuilderError(_absence_message(material_id, resolved_root))
    cif_path = row["cif_path"]
    if not cif_path:
        raise BuilderError(
            f"the snapshot records no CIF for {material_id}; its metadata "
            "exists but no structure file was archived"
        )
    resolved = _resolve_cif(resolved_root, str(cif_path))
    if not resolved.is_file():
        raise BuilderError(
            f"the snapshot lists {cif_path!r} for {material_id}, but the file "
            f"is missing under {resolved_root} — the cifs/ tree may not have "
            "been transferred completely"
        )
    return resolved


def query_materials(
    sql: str, *, limit: int = 200, root: str | os.PathLike[str] | None = None
) -> dict[str, Any]:
    """Run one read-only SELECT against the snapshot database.

    Accepts a single statement whose first keyword is ``SELECT`` or
    ``WITH`` (the read-only connection is the hard backstop behind that
    check). Returns ``{"rows": [...], "row_count": N, "truncated": bool}``
    with at most *limit* rows (clamped to 1-500), so an unbounded query can
    never pour tens of thousands of records at the caller.
    """
    keyword = _leading_keyword(sql)
    if keyword not in {"SELECT", "WITH"}:
        raise BuilderError(
            f"only read-only queries run here (got {keyword or 'nothing'!r}); "
            "start the statement with SELECT or WITH. The snapshot database "
            "is immutable — derived results belong in the workspace, not here"
        )
    clamped = max(1, min(int(limit), _MAX_ROWS))
    with contextlib.closing(connect(root)) as connection:
        try:
            cursor = connection.execute(sql)
            rows = cursor.fetchmany(clamped + 1)
        except sqlite3.Error as e:
            raise BuilderError(f"snapshot query failed: {e}") from e
    kept = [dict(row) for row in rows[:clamped]]
    return {"rows": kept, "row_count": len(kept), "truncated": len(rows) > clamped}


def _leading_keyword(sql: str) -> str:
    """The first SQL keyword of *sql*, with comments stripped, uppercased.

    Examples:
        >>> _leading_keyword("  -- top materials\\n  select * from materials")
        'SELECT'
        >>> _leading_keyword("/* cte */ WITH t AS (SELECT 1) SELECT * FROM t")
        'WITH'
        >>> _leading_keyword("DROP TABLE materials")
        'DROP'
    """
    stripped = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    stripped = re.sub(r"--[^\n]*", " ", stripped)
    match = re.search(r"[A-Za-z]+", stripped)
    return match.group(0).upper() if match else ""


def _material_columns(connection: sqlite3.Connection) -> tuple[str, ...]:
    columns = tuple(
        row["name"] for row in connection.execute('PRAGMA table_info("materials")')
    )
    if not columns:
        raise BuilderError(
            "metadata.sqlite has no materials table — this database is not "
            "a Materials Project snapshot, or the build that produced it "
            "was incomplete"
        )
    return columns


def _select_list(
    columns: list[str] | tuple[str, ...] | None, known: tuple[str, ...]
) -> str:
    if not columns:
        return "m.*"
    unknown = [name for name in columns if name not in known]
    if unknown:
        raise BuilderError(_unknown_columns(unknown, known))
    return ", ".join(f'm."{name}"' for name in columns)


def _filter_clauses(
    filters: dict[str, Any], known: tuple[str, ...]
) -> tuple[str, list[Any]]:
    """The WHERE clause and its parameters for a filter mapping.

    Examples:
        >>> _filter_clauses({"band_gap__gte": 1.0}, ("material_id", "band_gap"))
        (' WHERE m."band_gap" >= ?', [1.0])
        >>> _filter_clauses({"band_gap": None}, ("material_id", "band_gap"))
        (' WHERE m."band_gap" IS NULL', [])
    """
    clauses: list[str] = []
    parameters: list[Any] = []
    for key, value in filters.items():
        if key in {"elements", "exclude_elements"}:
            prefix = "NOT " if key == "exclude_elements" else ""
            for element in _element_list(key, value):
                clauses.append(
                    f"{prefix}EXISTS (SELECT 1 FROM material_elements AS e "
                    "WHERE e.material_id = m.material_id AND e.element = ?)"
                )
                parameters.append(element)
            continue
        column, _, suffix = key.partition("__")
        if column not in known:
            raise BuilderError(_unknown_columns([column], known))
        if not suffix:
            if value is None:
                clauses.append(f'm."{column}" IS NULL')
            else:
                clauses.append(f'm."{column}" = ?')
                parameters.append(value)
            continue
        operator = _OPERATORS.get(suffix)
        if operator is None:
            raise BuilderError(
                f"unknown filter suffix {suffix!r} in {key!r}; the suffixes "
                f"are {', '.join(sorted(_OPERATORS))} (bare names test equality)"
            )
        if value is None:
            if suffix == "ne":
                clauses.append(f'm."{column}" IS NOT NULL')
                continue
            raise BuilderError(
                f"{key!r} compares against None; NULL orders with nothing — "
                "use a bare column name for IS NULL, or col__ne for IS NOT NULL"
            )
        clauses.append(f'm."{column}" {operator} ?')
        parameters.append(value)
    if not clauses:
        return "", []
    return " WHERE " + " AND ".join(clauses), parameters


def _element_list(key: str, value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    try:
        elements = [str(element) for element in value]
    except TypeError:
        raise BuilderError(
            f"{key!r} takes a list of element symbols, got {value!r}"
        ) from None
    for element in elements:
        if not _ELEMENT_PATTERN.match(element):
            raise BuilderError(
                f"{element!r} is not an element symbol (expected forms like "
                "'Fe' or 'O'; symbols are case-sensitive)"
            )
    return elements


def _order_clause(order_by: str | None, known: tuple[str, ...]) -> str:
    if order_by is None:
        return ""
    descending = order_by.startswith("-")
    column = order_by[1:] if descending else order_by
    if column not in known:
        raise BuilderError(_unknown_columns([column], known))
    return f' ORDER BY m."{column}"' + (" DESC" if descending else "")


def _unknown_columns(unknown: list[str], known: tuple[str, ...]) -> str:
    return (
        f"unknown column{'s' if len(unknown) > 1 else ''} "
        f"{', '.join(repr(name) for name in unknown)}; the materials table "
        f"has: {', '.join(known)}"
    )


def _guard_material_id(material_id: str) -> None:
    if not isinstance(material_id, str) or not _MATERIAL_ID_PATTERN.match(material_id):
        raise BuilderError(
            f"{material_id!r} does not look like a material id (expected "
            "forms like 'mp-149'); pass one material_id from a search result"
        )


def _resolve_cif(root: Path, cif_path: str) -> Path:
    resolved = (root / cif_path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise BuilderError(
            f"cif_path {cif_path!r} escapes the snapshot root {root}; "
            "refusing to resolve it — the snapshot stores relative paths "
            "below its own root, so this record is corrupt"
        )
    return resolved


def _absence_message(material_id: str, root: Path) -> str:
    release: str | None
    try:  # a second open, on the error path only
        release = snapshot_info(root)["release"]
    except BuilderError:
        release = None
    named = f"release {release}" if release else "release unknown"
    return (
        f"{material_id} is not in the installed snapshot ({named}). The "
        "snapshot is the only source here; there is no online fallback — "
        "report absence as absence"
    )


def _material_count(connection: sqlite3.Connection) -> int:
    try:
        row = connection.execute("SELECT count(*) AS n FROM materials").fetchone()
    except sqlite3.Error as e:
        raise BuilderError(
            f"metadata.sqlite has no readable materials table: {e}"
        ) from e
    return int(row["n"])


def _dataset_info(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """The ``dataset_info`` rows, or an empty list when the table is absent."""
    try:
        rows = connection.execute("SELECT * FROM dataset_info LIMIT 20").fetchall()
    except sqlite3.Error:
        return []
    return [dict(row) for row in rows]


def _release(info: dict[str, Any]) -> str | None:
    """The snapshot's release string, from the manifest or ``dataset_info``.

    Snapshot builds vary in where they record the release, so the search is
    by key name, manifest first, in a fixed trust order. A ``dataset_info``
    shaped as key/value rows is folded into one mapping so its entries are
    searched by their declared keys, not their column names.
    """
    candidates: list[dict[str, Any]] = []
    manifest = info.get("manifest")
    if isinstance(manifest, dict):
        candidates.append(manifest)
    folded: dict[str, Any] = {}
    for row in info.get("dataset_info", []):
        if not isinstance(row, dict):
            continue
        if set(row) in ({"key", "value"}, {"name", "value"}, {"field", "value"}):
            folded[str(row.get("key") or row.get("name") or row.get("field"))] = row[
                "value"
            ]
        else:
            candidates.append(row)
    if folded:
        candidates.append(folded)
    for key in _RELEASE_KEYS:
        for record in candidates:
            value = record.get(key)
            if isinstance(value, (str, int)) and str(value).strip():
                return str(value)
    return None


def _mp_setting(key: str) -> Any:
    from slab.config import config_value

    return config_value(f"builders.mp.{key}")
