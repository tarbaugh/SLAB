import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from slab import SQLiteRunStore
from slab.pseudos import PseudoFamily, family_dir_name


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
