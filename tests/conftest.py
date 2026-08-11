from collections.abc import Iterator
from pathlib import Path

import pytest

from slab import SQLiteRunStore


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "runs.db"


@pytest.fixture()
def store(db_path: Path) -> Iterator[SQLiteRunStore]:
    s = SQLiteRunStore(db_path)
    yield s
    s.close()
