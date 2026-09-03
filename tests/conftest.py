import hashlib
import json
import sqlite3
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from foundation import SQLiteRunStore
from slab.pseudos import PseudoFamily, family_dir_name


@pytest.fixture(autouse=True)
def _isolated_user_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """Every test sees an empty user config layer.

    Skill and roster discovery read ``~/.config/slab/`` through
    ``$XDG_CONFIG_HOME``, so without this a developer's real user files
    would leak into assertions. A test that wants a user layer sets
    ``XDG_CONFIG_HOME`` itself, which overrides this fixture's value.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("xdg")))


class LlmScript:
    """What the fake OpenAI-compatible server should answer, and what it saw."""

    def __init__(self) -> None:
        self.responses: list[tuple[int, dict[str, Any]]] = []
        self.requests: list[dict[str, Any]] = []
        self.get_response: tuple[int, dict[str, Any]] = (200, {"data": []})


class _LlmHandler(BaseHTTPRequestHandler):
    script: LlmScript

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length)) if length else {}
        body["_auth"] = self.headers.get("Authorization")
        body["_headers"] = {key.lower(): value for key, value in self.headers.items()}
        self.script.requests.append(body)
        status, payload = (
            self.script.responses.pop(0) if self.script.responses else (500, {"error": "empty"})
        )
        self._answer(status, payload)

    def do_GET(self) -> None:
        self._answer(*self.script.get_response)

    def _answer(self, status: int, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args: object) -> None:  # keep test output quiet
        pass


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries wait for real in life: about 80 s after five 5xx answers.

    The fake server answers 500 once a test's script runs dry, so without
    this every such test would wait the whole backoff. A test that asserts
    on the waits patches ``mason.client._sleep`` itself, and its patch wins.

    Patch the client's own attribute, never ``time.sleep``: ``mason.client.time``
    is the shared ``time`` module, so a patch there turns every sleep in the
    suite into a no-op, and a test that sleeps to let a clock tick or a child
    process finish then measures nothing.
    """
    monkeypatch.setattr("mason.client._sleep", lambda s: None)


@pytest.fixture()
def llm_server() -> Iterator[tuple[str, LlmScript]]:
    """A live local OpenAI-compatible server driven by a per-test script."""
    script = LlmScript()
    handler = type("Handler", (_LlmHandler,), {"script": script})
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1", script
    httpd.shutdown()
    httpd.server_close()  # shutdown stops serving; the listening socket needs closing too
    thread.join(timeout=5)


def _upf_bytes(symbol: str) -> bytes:
    return f'<UPF version="2.0.1"><!-- {symbol} test pseudo --></UPF>\n'.encode()


def make_family(
    root: Path, name: str = "SSSP/1.3.0/PBEsol/efficiency", symbols: tuple[str, ...] = ("Si", "O")
) -> tuple[PseudoFamily, Path]:
    """Write a small but structurally real pseudopotential family under *root*."""
    cutoffs = {"Si": (30.0, 240.0), "O": (75.0, 600.0), "Cu": (90.0, 720.0)}
    directory = root / family_dir_name(name)
    directory.mkdir(parents=True)
    elements = {}
    for symbol in symbols:
        filename = f"{symbol}.test.upf"
        payload = _upf_bytes(symbol)
        (directory / filename).write_bytes(payload)
        wfc, rho = cutoffs[symbol]
        elements[symbol] = {
            "filename": filename,
            "md5": hashlib.md5(payload).hexdigest(),
            "cutoff_wfc": wfc,
            "cutoff_rho": rho,
        }
    family = PseudoFamily.model_validate({"name": name, "elements": elements})
    (directory / "family.json").write_text(json.dumps(family.model_dump(mode="json")))
    return family, directory


#: The four bulk structures the miniature snapshot archives, with the metadata
#: a real snapshot's materials table would carry for them. mp-22862 carries
#: NULL band_gap / total_magnetization / ordering so tests can pin that NULL
#: means "not populated", never zero.
_MP_ROWS: tuple[tuple[Any, ...], ...] = (
    ("mp-149", "Si", 1, 2, 0.0, 0.61, 1, 0, 0.0, "NM", "CC-BY-4.0", "cifs/00/mp-149.cif"),
    ("mp-13", "Fe", 1, 1, 0.0, 0.0, 1, 1, 2.2, "FM", "CC-BY-4.0", "cifs/01/mp-13.cif"),
    (
        "mp-1271068",
        "Fe",
        1,
        1,
        0.081,
        0.0,
        0,
        1,
        2.6,
        "FM",
        "CC-BY-4.0",
        "cifs/01/mp-1271068.cif",
    ),
    (
        "mp-22862",
        "NaCl",
        2,
        2,
        0.0,
        None,
        1,
        0,
        None,
        None,
        "CC-BY-4.0",
        "cifs/02/mp-22862.cif",
    ),
)

_MP_COLUMNS = (
    "material_id",
    "formula_pretty",
    "nelements",
    "nsites",
    "energy_above_hull",
    "band_gap",
    "is_stable",
    "is_magnetic",
    "total_magnetization",
    "ordering",
    "source_license",
    "cif_path",
)


def _mp_atoms() -> dict[str, Any]:
    from ase.build import bulk

    return {
        "mp-149": bulk("Si", "diamond", a=5.43),
        "mp-13": bulk("Fe", "bcc", a=2.87),
        "mp-1271068": bulk("Fe", "fcc", a=3.45),
        "mp-22862": bulk("NaCl", "rocksalt", a=5.64),
    }


def build_mp_snapshot(
    dest: Path,
    *,
    release: str | None = "2025.11.1",
    manifest: bool = True,
    extra_materials: tuple[dict[str, Any], ...] = (),
) -> Path:
    """Build a miniature but structurally real Materials Project snapshot.

    Real CIFs written by ASE, the ``materials`` table with the columns
    workflows filter on, the ``material_elements`` membership table, a
    key/value ``dataset_info``, a ``units`` table, and ``manifest.json``.
    *release* ``None`` omits the release everywhere (the fingerprint
    fallback); *extra_materials* rows are inserted as given, with no CIF
    written — how tests plant corrupt or incomplete records.
    """
    from ase.io import write as ase_write

    dest.mkdir(parents=True, exist_ok=True)
    atoms_by_id = _mp_atoms()
    connection = sqlite3.connect(dest / "metadata.sqlite")
    try:
        connection.executescript(
            """
            CREATE TABLE materials (
                material_id TEXT PRIMARY KEY,
                formula_pretty TEXT,
                nelements INTEGER,
                nsites INTEGER,
                energy_above_hull REAL,
                band_gap REAL,
                is_stable INTEGER,
                is_magnetic INTEGER,
                total_magnetization REAL,
                ordering TEXT,
                source_license TEXT,
                cif_path TEXT
            );
            CREATE TABLE material_elements (material_id TEXT, element TEXT);
            CREATE INDEX ix_material_elements ON material_elements(element);
            CREATE TABLE dataset_info (key TEXT, value TEXT);
            CREATE TABLE units (field TEXT, unit TEXT, description TEXT);
            """
        )
        placeholders = ", ".join("?" for _ in _MP_COLUMNS)
        for row in _MP_ROWS:
            connection.execute(f"INSERT INTO materials VALUES ({placeholders})", row)
            record = dict(zip(_MP_COLUMNS, row, strict=True))
            atoms = atoms_by_id[str(record["material_id"])]
            cif = dest / str(record["cif_path"])
            cif.parent.mkdir(parents=True, exist_ok=True)
            ase_write(cif, atoms, format="cif")
            for element in sorted(set(atoms.get_chemical_symbols())):
                connection.execute(
                    "INSERT INTO material_elements VALUES (?, ?)",
                    (record["material_id"], element),
                )
        for extra in extra_materials:
            values = [extra.get(column) for column in _MP_COLUMNS]
            connection.execute(f"INSERT INTO materials VALUES ({placeholders})", values)
        if release is not None:
            connection.execute(
                "INSERT INTO dataset_info VALUES (?, ?)", ("database_release", release)
            )
        connection.execute(
            "INSERT INTO dataset_info VALUES (?, ?)",
            ("build_timestamp", "2026-08-30T12:00:00Z"),
        )
        connection.executemany(
            "INSERT INTO units VALUES (?, ?, ?)",
            [
                ("energy_above_hull", "eV/atom", "Energy above the convex hull"),
                ("band_gap", "eV", "Band gap; NULL means not populated"),
                ("total_magnetization", "muB", "Total magnetization per cell"),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    if manifest:
        payload: dict[str, Any] = {
            "material_count": len(_MP_ROWS) + len(extra_materials),
            "build_timestamp": "2026-08-30T12:00:00Z",
            "license_labels": ["CC-BY-4.0"],
        }
        if release is not None:
            payload["database_release"] = release
        (dest / "manifest.json").write_text(json.dumps(payload, indent=1))
    return dest


@pytest.fixture()
def mp_snapshot(tmp_path: Path) -> Path:
    """A miniature real snapshot at ``<tmp>/mp-snapshot``."""
    return build_mp_snapshot(tmp_path / "mp-snapshot")


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "runs.db"


@pytest.fixture()
def store(db_path: Path) -> Iterator[SQLiteRunStore]:
    s = SQLiteRunStore(db_path)
    yield s
    s.close()
