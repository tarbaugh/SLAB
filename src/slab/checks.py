"""The assertion vocabulary for verification hooks.

Each function evaluates one machine-checkable claim and returns an
:class:`Assertion` — data, not an exception — so results are recordable
whether they pass or fail. The vocabulary never raises on malformed *observed*
values (a check that crashes on a NaN would defeat its purpose); it returns a
failed assertion instead. Malformed *expectations* (e.g. a non-numeric bound)
are programming errors and do raise.

Vocabulary:

* :func:`converged` — an observed residual fell below a threshold.
* :func:`within_bounds` — a value lies inside [lo, hi].
* :func:`finite` — a scalar, or every element of a sequence, is finite.
* :func:`units` — a producer-declared unit string matches the expected one.
  This is annotation capture, not dimensional analysis: it catches the
  "engine returned kcal/mol, workflow assumed eV" class of error exactly when
  producers record their units.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict


class Assertion(BaseModel):
    """The outcome of one vocabulary assertion. Immutable.

    Examples:
        >>> a = converged(0.03, below=0.05)
        >>> (a.kind, a.passed)
        ('converged', True)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    passed: bool
    message: str
    observed: Any = None
    expected: Any = None


def _as_number(value: object) -> float | None:
    """Coerce to float for checking; bools and non-numbers are None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _require_number(value: object, what: str) -> float:
    number = _as_number(value)
    if number is None:
        raise TypeError(f"{what} must be a number, got {type(value).__name__!r}")
    return number


def converged(observed: object, *, below: float, label: str = "residual") -> Assertion:
    """Assert that an observed residual (fmax, energy change, ...) is `< below`.

    Examples:
        >>> converged(0.031, below=0.05, label="fmax").message
        'fmax=0.031 < 0.05'
        >>> converged(0.12, below=0.05, label="fmax").passed
        False
        >>> converged(float("nan"), below=0.05).passed
        False
    """
    threshold = _require_number(below, "below")
    value = _as_number(observed)
    if value is None or not math.isfinite(value):
        return Assertion(
            kind="converged",
            passed=False,
            message=f"{label} is not a finite number: {observed!r}",
            observed=repr(observed),
            expected={"below": threshold},
        )
    passed = value < threshold
    op = "<" if passed else "!<"
    return Assertion(
        kind="converged",
        passed=passed,
        message=f"{label}={value:g} {op} {threshold:g}",
        observed=value,
        expected={"below": threshold},
    )


def within_bounds(
    observed: object,
    *,
    lo: float | None = None,
    hi: float | None = None,
    label: str = "value",
) -> Assertion:
    """Assert that `lo <= observed <= hi` (bounds inclusive; either may be omitted).

    Raises:
        ValueError: Neither bound was given.

    Examples:
        >>> within_bounds(-10.8, lo=-12, hi=-9, label="energy").message
        'energy=-10.8 within [-12, -9]'
        >>> within_bounds(3.2, hi=2.0).passed
        False
    """
    if lo is None and hi is None:
        raise ValueError("within_bounds requires at least one of lo=, hi=")
    low = None if lo is None else _require_number(lo, "lo")
    high = None if hi is None else _require_number(hi, "hi")
    span = f"[{'-inf' if low is None else f'{low:g}'}, {'inf' if high is None else f'{high:g}'}]"
    expected = {"lo": low, "hi": high}
    value = _as_number(observed)
    if value is None or not math.isfinite(value):
        return Assertion(
            kind="within_bounds",
            passed=False,
            message=f"{label} is not a finite number: {observed!r}",
            observed=repr(observed),
            expected=expected,
        )
    passed = (low is None or low <= value) and (high is None or value <= high)
    verdict = "within" if passed else "outside"
    return Assertion(
        kind="within_bounds",
        passed=passed,
        message=f"{label}={value:g} {verdict} {span}",
        observed=value,
        expected=expected,
    )


def finite(observed: object, *, label: str = "value") -> Assertion:
    """Assert that a scalar — or every element of a sequence — is finite.

    Catches the NaN/inf class of silent corruption (exploded dynamics,
    diverged SCF) before it propagates.

    Examples:
        >>> finite([1.0, 2.5, -3.0], label="forces").message
        'all 3 forces values finite'
        >>> finite(float("inf")).passed
        False
    """
    if isinstance(observed, Iterable) and not isinstance(observed, (str, bytes, dict)):
        values = list(observed)
    else:
        values = [observed]
    if not values:
        return Assertion(
            kind="finite", passed=False, message=f"{label} is empty", observed=[], expected="finite"
        )
    bad: list[str] = []
    for item in values:
        number = _as_number(item)
        if number is None or not math.isfinite(number):
            bad.append(repr(item))
    if bad:
        shown = ", ".join(bad[:3]) + (", ..." if len(bad) > 3 else "")
        message = f"{len(bad)}/{len(values)} {label} values not finite: {shown}"
    elif len(values) == 1:
        message = f"{label}={values[0]:g} is finite"
    else:
        message = f"all {len(values)} {label} values finite"
    return Assertion(
        kind="finite",
        passed=not bad,
        message=message,
        observed=None if len(values) > 8 else [repr(v) for v in values],
        expected="finite",
    )


def units(observed: object, expected: str, *, label: str = "unit") -> Assertion:
    """Assert that a producer-declared unit string matches the expected one.

    Comparison is exact after whitespace stripping (case matters: ``eV`` and
    ``ev`` are different claims). This is unit *annotation* checking, not
    dimensional analysis — see the module docstring.

    Examples:
        >>> units("eV", "eV").passed
        True
        >>> units("kcal/mol", "eV").message
        "unit 'kcal/mol' != expected 'eV'"
    """
    if not isinstance(observed, str):
        return Assertion(
            kind="units",
            passed=False,
            message=f"{label} is not declared as a string: {observed!r}",
            observed=repr(observed),
            expected=expected,
        )
    declared = observed.strip()
    wanted = expected.strip()
    passed = declared == wanted
    op = "==" if passed else "!="
    return Assertion(
        kind="units",
        passed=passed,
        message=f"{label} {declared!r} {op} expected {wanted!r}",
        observed=declared,
        expected=wanted,
    )
