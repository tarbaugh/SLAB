import time

import pytest

from foundation._ids import _CROCKFORD, new_run_id


def test_format() -> None:
    rid = new_run_id()
    assert len(rid) == 26
    assert set(rid) <= set(_CROCKFORD)
    assert rid == rid.lower()


def test_uniqueness() -> None:
    ids = {new_run_id() for _ in range(10_000)}
    assert len(ids) == 10_000


def test_time_ordering_explicit_timestamps() -> None:
    earlier = new_run_id(timestamp_ms=1_000)
    later = new_run_id(timestamp_ms=1_001)
    assert earlier < later


def _wait_for_next_millisecond() -> None:
    """Spin until the wall clock's millisecond changes.

    The ordering guarantee is per millisecond, so the test needs the
    embedded timestamps to differ. Reading the clock is what the generator
    itself does; ``time.sleep`` may be shortened or patched out.
    """
    start = time.time_ns() // 1_000_000
    while time.time_ns() // 1_000_000 == start:
        pass


def test_time_ordering_wall_clock() -> None:
    a = new_run_id()
    _wait_for_next_millisecond()
    b = new_run_id()
    assert a < b


@pytest.mark.parametrize("bad_ts", [-1, 1 << 48])
def test_timestamp_out_of_range(bad_ts: int) -> None:
    with pytest.raises(ValueError, match="out of range"):
        new_run_id(timestamp_ms=bad_ts)
