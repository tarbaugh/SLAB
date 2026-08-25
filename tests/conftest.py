import hashlib
import json
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


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "runs.db"


@pytest.fixture()
def store(db_path: Path) -> Iterator[SQLiteRunStore]:
    s = SQLiteRunStore(db_path)
    yield s
    s.close()
