"""Value serialization for tracing: canonical bytes, content hashes, round-trips.

Traced task inputs and outputs are Python values; to hash, cache, and store
them they must become bytes deterministically. Encoding rules:

* JSON-faithful values (``None``/``bool``/``int``/``float``/``str`` and
  ``dict``/``list`` compositions with string keys) -> canonical JSON with
  sorted keys, so two equal dicts hash equally regardless of insertion order.
* ``bytes`` -> stored as-is.
* Everything else (tuples, sets, numpy arrays, ASE ``Atoms``, ...) -> pickle.
  Pickle bytes are only *practically* stable — an unstable encoding can cause
  a spurious cache miss, never a wrong cache hit, so the failure direction is
  safe.

Each blob is prefixed with a 2-byte format tag (``J\\n``/``B\\n``/``P\\n``) so
it is self-describing; :func:`loads` needs no side-channel. Blobs are decoded
only from SLAB's own artifact store — the usual caveat about unpickling
untrusted data does not arise within a workspace you own.
"""

from __future__ import annotations

import hashlib
import json
import pickle

from slab.errors import SerializationError

_JSON = b"J\n"
_RAW = b"B\n"
_PICKLE = b"P\n"


def _json_faithful(obj: object) -> bool:
    """True if *obj* round-trips through JSON with identity of type and value."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return True
    if isinstance(obj, list):
        return all(_json_faithful(item) for item in obj)
    if isinstance(obj, dict):
        return all(isinstance(k, str) and _json_faithful(v) for k, v in obj.items())
    return False


def dumps(obj: object) -> bytes:
    """Encode *obj* to self-describing bytes (see module docstring for rules).

    Raises:
        SerializationError: The value cannot be pickled (e.g. a lambda, an open
            file handle, a live socket).

    Examples:
        >>> dumps({"b": 1, "a": 2}) == dumps({"a": 2, "b": 1})  # canonical
        True
        >>> dumps(b"raw")
        b'B\\nraw'
    """
    if isinstance(obj, bytes):
        return _RAW + obj
    if _json_faithful(obj):
        payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return _JSON + payload.encode()
    try:
        return _PICKLE + pickle.dumps(obj, protocol=5)
    except Exception as e:
        raise SerializationError(
            f"cannot serialize {type(obj).__name__!r} value for tracing: {e}"
        ) from e


def loads(data: bytes) -> object:
    """Decode bytes produced by :func:`dumps`.

    Raises:
        SerializationError: The payload has no recognizable format tag.

    Examples:
        >>> loads(dumps((1, 2, 3)))
        (1, 2, 3)
        >>> loads(dumps({"kpts": [4, 4, 4]}))
        {'kpts': [4, 4, 4]}
    """
    tag, payload = data[:2], data[2:]
    if tag == _RAW:
        return payload
    if tag == _JSON:
        return json.loads(payload)
    if tag == _PICKLE:
        return pickle.loads(payload)
    raise SerializationError(f"unrecognized serialization tag {tag!r}")


def fingerprint(obj: object) -> str:
    """SHA-256 content hash of *obj*'s canonical encoding.

    Equal values get equal fingerprints (within an encoding family); this is
    the key used for task caching and DAG edge derivation.

    Examples:
        >>> fingerprint({"a": 1}) == fingerprint({"a": 1})
        True
        >>> len(fingerprint("x"))
        64
    """
    return hashlib.sha256(dumps(obj)).hexdigest()
