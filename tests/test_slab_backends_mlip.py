"""MLIP backend seams that need no MLIP installed.

The checkpoint-fetch guard is pure harness: it exists because mace-torch
downloads named checkpoints with no socket timeout, which on a firewalled
compute node is a silent hang inside the batch job's time limit. The guard
is tested against a stand-in factory so the suite never needs torch.
"""

from __future__ import annotations

import socket
import urllib.error
from typing import Any

import pytest

from slab.backends import _CHECKPOINT_FETCH_TIMEOUT_S, _fetch_named_checkpoint
from slab.errors import EngineNotAvailableError


def test_fetch_guard_turns_a_blackhole_into_instructions() -> None:
    """URLError (what a firewalled download raises under a socket timeout)
    becomes a loud refusal that teaches the pre-warm and rootstock routes."""

    def firewalled(**options: Any) -> Any:
        raise urllib.error.URLError("connection timed out")

    with pytest.raises(EngineNotAvailableError) as excinfo:
        _fetch_named_checkpoint(firewalled, {"model": "small"}, engine="mace")
    message = str(excinfo.value)
    assert "mace_mp(model='small')" in message  # the pre-warm command
    assert "rootstock" in message
    assert "firewalled" in message


def test_fetch_guard_bounds_and_restores_the_socket_timeout() -> None:
    """Inside the construction window the default timeout is bounded (the
    hang becomes an error); afterwards the previous default is restored on
    success and failure alike."""
    previous = socket.getdefaulttimeout()
    observed: list[float | None] = []

    def watching(**options: Any) -> str:
        observed.append(socket.getdefaulttimeout())
        return "calculator"

    try:
        socket.setdefaulttimeout(None)
        assert _fetch_named_checkpoint(watching, {}, engine="mace") == "calculator"
        assert observed == [_CHECKPOINT_FETCH_TIMEOUT_S]
        assert socket.getdefaulttimeout() is None
        with pytest.raises(EngineNotAvailableError):
            _fetch_named_checkpoint(
                lambda **kw: (_ for _ in ()).throw(TimeoutError("stalled")), {}, engine="mace"
            )
        assert socket.getdefaulttimeout() is None
    finally:
        socket.setdefaulttimeout(previous)


def test_fetch_guard_lets_unrelated_errors_through_unwrapped() -> None:
    """Only network shapes get the firewall story: a ValueError from bad
    options must surface as itself, not as download advice."""
    with pytest.raises(ValueError, match="bad model"):
        _fetch_named_checkpoint(
            lambda **kw: (_ for _ in ()).throw(ValueError("bad model")), {}, engine="mace"
        )


def test_fetch_guard_sees_through_mace_runtime_error_wrapping() -> None:
    """mace-torch wraps every download failure in RuntimeError('Model
    download failed...'), so the network shapes arrive in disguise — by
    cause and by message, both must translate."""

    def wrapped_cause(**options: Any) -> Any:
        try:
            raise urllib.error.URLError("blackholed")
        except urllib.error.URLError as e:
            raise RuntimeError("Model download failed and no local model found") from e

    with pytest.raises(EngineNotAvailableError, match="rootstock"):
        _fetch_named_checkpoint(wrapped_cause, {"model": "small"}, engine="mace")

    def wrapped_message(**options: Any) -> Any:
        raise RuntimeError("Model download failed and no local model found")

    with pytest.raises(EngineNotAvailableError, match="firewalled"):
        _fetch_named_checkpoint(wrapped_message, {"model": "small"}, engine="mace")

    def unrelated(**options: Any) -> Any:
        raise RuntimeError("mismatched tensor shapes")

    with pytest.raises(RuntimeError, match="tensor"):
        _fetch_named_checkpoint(unrelated, {}, engine="mace")


def test_mace_checkpoint_file_identity_includes_freshness() -> None:
    """model= may be a checkpoint FILE; a path alone would let a
    retrain-in-place serve stale cached results forever."""
    import tempfile
    from pathlib import Path

    from slab.backends import describe_engine

    with tempfile.TemporaryDirectory() as root:
        checkpoint = Path(root) / "custom.model"
        checkpoint.write_bytes(b"weights-v1")
        a = describe_engine("mace", {"model": str(checkpoint)})
        assert a["model"] == str(checkpoint)
        assert a["model_size"] == len(b"weights-v1")
        import os

        os.utime(checkpoint, ns=(1, 1))
        checkpoint.write_bytes(b"weights-v2-retrained")
        b = describe_engine("mace", {"model": str(checkpoint)})
        assert (a["model_mtime_ns"], a["model_size"]) != (b["model_mtime_ns"], b["model_size"])
    # Named aliases stay name-identified — no file keys.
    alias = describe_engine("mace", {"model": "small"})
    assert "model_mtime_ns" not in alias
