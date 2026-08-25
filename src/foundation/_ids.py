"""ULID-style run identifiers: time-ordered, compact, CLI- and token-friendly.

Ids sort lexicographically in creation order, so ``ORDER BY id`` is chronological
and short prefixes (git-style) usually resolve to the most convenient runs.
"""

from __future__ import annotations

import os
import time

_CROCKFORD = "0123456789abcdefghjkmnpqrstvwxyz"


def new_run_id(*, timestamp_ms: int | None = None) -> str:
    """Return a new 26-character lowercase ULID.

    Layout: 48-bit millisecond timestamp followed by 80 random bits, encoded in
    Crockford base32 (lowercased). Ids created later sort lexicographically after
    ids created earlier, to millisecond resolution.

    Args:
        timestamp_ms: Override the embedded timestamp (intended for tests).

    Examples:
        >>> a = new_run_id(timestamp_ms=1)
        >>> b = new_run_id(timestamp_ms=2)
        >>> len(a), a < b
        (26, True)
    """
    ts = time.time_ns() // 1_000_000 if timestamp_ms is None else timestamp_ms
    if not 0 <= ts < 1 << 48:
        raise ValueError(f"timestamp_ms out of range for a 48-bit ULID timestamp: {ts}")
    value = (ts << 80) | int.from_bytes(os.urandom(10), "big")
    return "".join(_CROCKFORD[(value >> shift) & 0x1F] for shift in range(125, -1, -5))
