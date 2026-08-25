"""Tests for structured failure evidence (foundation.errors.failure_record).

The record is what an LLM agent reads to decide a correction, so these tests
pin both directions of its contract: information-rich (traceback, chain,
notes all present) and token-bounded (elision and clipping actually engage).
"""

from __future__ import annotations

import pytest

from foundation.errors import (
    _MAX_MESSAGE_CHARS,
    _MAX_TRACEBACK_CHARS,
    _TRACEBACK_HEAD,
    _TRACEBACK_TAIL,
    failure_record,
)


def _caught(exc: BaseException) -> BaseException:
    """Raise and catch *exc* so it carries a real traceback."""
    try:
        raise exc
    except BaseException as e:  # deliberately generic test helper
        return e


def test_record_has_type_message_traceback() -> None:
    record = failure_record(_caught(RuntimeError("SCF diverged")))
    assert record["type"] == "RuntimeError"
    assert record["message"] == "SCF diverged"
    assert record["traceback"].startswith("Traceback (most recent call last)")
    assert "RuntimeError: SCF diverged" in record["traceback"]
    assert "notes" not in record  # absent, not empty: no key to mislead a reader


def test_notes_are_listed_and_inside_the_traceback() -> None:
    error = _caught(ValueError("no convergence"))
    error.add_note("relax failed after 143 completed step(s)")
    record = failure_record(error)
    assert record["notes"] == ["relax failed after 143 completed step(s)"]
    # format_exception appends notes after the exception line (PEP 678)
    assert "relax failed after 143 completed step(s)" in record["traceback"]


def test_cause_chain_is_preserved() -> None:
    try:
        try:
            raise OSError("pw.x exited with status 2")
        except OSError as root:
            raise RuntimeError("calculation failed") from root
    except RuntimeError as e:
        record = failure_record(e)
    assert record["type"] == "RuntimeError"  # the final exception names the record
    assert "OSError: pw.x exited with status 2" in record["traceback"]
    assert "direct cause" in record["traceback"]


def test_deep_stacks_elide_the_middle() -> None:
    # A deep stack of DISTINCT frames (recursion won't do: Python collapses
    # repeated identical frames on its own, e.g. "[Previous line repeated ...]").
    depth = 20
    source = "\n".join(
        [f"def f{i}():\n    return f{i + 1}()" for i in range(depth)]
        + [f"def f{depth}():\n    raise RuntimeError('bottom')"]
    )
    namespace: dict[str, object] = {}
    exec(source, namespace)  # deliberate frame factory
    with pytest.raises(RuntimeError) as excinfo:
        namespace["f0"]()  # type: ignore[operator]
    record = failure_record(excinfo.value)
    assert "frame(s) elided" in record["traceback"]
    # entry point and failure site both survive the elision
    assert "test_deep_stacks_elide_the_middle" in record["traceback"]
    assert f"in f{depth}" in record["traceback"]
    assert "RuntimeError: bottom" in record["traceback"]
    frame_lines = [
        line for line in record["traceback"].splitlines() if line.startswith('  File "')
    ]
    assert len(frame_lines) == _TRACEBACK_HEAD + _TRACEBACK_TAIL


def test_repeated_recursion_frames_stay_collapsed_not_elided() -> None:
    """Python's own repeat-collapse handles recursion; the record keeps it."""

    def descend(depth: int) -> None:
        if depth == 0:
            raise RuntimeError("bottom")
        descend(depth - 1)

    with pytest.raises(RuntimeError) as excinfo:
        descend(30)
    record = failure_record(excinfo.value)
    assert "Previous line repeated" in record["traceback"]
    assert 'raise RuntimeError("bottom")' in record["traceback"]


def test_shallow_stacks_are_untouched() -> None:
    record = failure_record(_caught(RuntimeError("x")))
    assert "elided" not in record["traceback"]


def test_long_messages_are_clipped() -> None:
    record = failure_record(_caught(ValueError("f" * (_MAX_MESSAGE_CHARS + 500))))
    assert len(record["message"]) < _MAX_MESSAGE_CHARS + 100
    assert "500 characters clipped" in record["message"]


def test_giant_message_cannot_wipe_the_frames() -> None:
    """A huge exception message is clipped as a piece, so the header, the
    frames, and the exception type all survive — the failure *site* is the
    field's whole purpose."""
    record = failure_record(_caught(ValueError("engine dump: " + "g" * (_MAX_TRACEBACK_CHARS * 2))))
    tb = record["traceback"]
    assert tb.startswith("Traceback (most recent call last)")
    assert tb.count('  File "') >= 1
    assert "ValueError" in tb
    assert "characters clipped" in tb
    assert len(tb) < _MAX_TRACEBACK_CHARS + 200


def test_very_long_chains_are_truncated_keeping_the_final_exceptions() -> None:
    """The 10k backstop engages on pathological cause chains; the end (the
    exceptions closest to the caller) survives."""
    error: BaseException = ValueError("link-0")
    for i in range(1, 120):
        try:
            raise ValueError(f"link-{i}") from error
        except ValueError as e:
            error = e
    record = failure_record(error)
    assert record["traceback"].startswith("[traceback truncated")
    assert "link-119" in record["traceback"]
    assert len(record["traceback"]) < _MAX_TRACEBACK_CHARS + 200


def test_notes_are_clipped_too() -> None:
    error = _caught(ValueError("x"))
    error.add_note("n" * (_MAX_MESSAGE_CHARS + 9))
    assert "9 characters clipped" in failure_record(error)["notes"][0]


def test_notes_count_is_capped() -> None:
    error = _caught(ValueError("x"))
    for i in range(15):
        error.add_note(f"note {i}")
    notes = failure_record(error)["notes"]
    assert len(notes) == 11
    assert notes[:2] == ["note 0", "note 1"]
    assert notes[-1] == "[... 5 more note(s)]"


# -- hostile input: the record builder runs inside exception handlers, where
#    raising would mask the very failure being recorded. It must never raise.


def test_non_sequence_notes_degrade_to_a_single_note() -> None:
    error = _caught(ValueError("bad"))
    error.__notes__ = 123  # type: ignore[attr-defined]
    assert failure_record(error)["notes"] == ["123"]


def test_string_notes_stay_one_note_not_characters() -> None:
    error = _caught(ValueError("bad"))
    error.__notes__ = "just a note"  # type: ignore[attr-defined]
    assert failure_record(error)["notes"] == ["just a note"]


def test_note_with_raising_str_gets_a_placeholder() -> None:
    class HostileNote:
        def __str__(self) -> str:
            raise RuntimeError("nope")

    error = _caught(ValueError("bad"))
    error.__notes__ = [HostileNote()]  # type: ignore[attr-defined]
    assert failure_record(error)["notes"] == ["<HostileNote str() failed>"]


def test_exception_with_raising_str_gets_a_placeholder_message() -> None:
    class HostileError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("nope")

    record = failure_record(_caught(HostileError()))
    assert record["type"] == "HostileError"
    assert record["message"] == "<HostileError str() failed>"


def test_unreadable_notes_attribute_gets_a_placeholder() -> None:
    class SneakyError(Exception):
        @property
        def __notes__(self) -> list[str]:  # getattr(default=) only covers AttributeError
            raise RuntimeError("no notes for you")

    record = failure_record(_caught(SneakyError("bad")))
    assert record["notes"] == ["<__notes__ unreadable>"]


def test_unformattable_traceback_degrades_to_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even the stdlib formatter blowing up must not raise out of the record
    builder — it runs inside exception handlers."""
    import traceback

    def broken_format(*args: object, **kwargs: object) -> list[str]:
        raise RuntimeError("formatter broke")

    monkeypatch.setattr(traceback, "format_exception", broken_format)
    record = failure_record(_caught(ValueError("bad")))
    assert record["traceback"] == "<traceback unavailable: RuntimeError: formatter broke>"
    assert record["message"] == "bad"


def test_record_is_json_serializable() -> None:
    import json

    error = _caught(RuntimeError("boom"))
    error.add_note("context")
    assert json.loads(json.dumps(failure_record(error)))["type"] == "RuntimeError"
