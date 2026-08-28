"""Arrhenius, Vogel-Fulcher-Tammann, and zero-crossing fits for rate tables.

    fit_rates.py rates.json --mode arrhenius
    fit_rates.py growth.json --mode vft --json
    fit_rates.py velocity.json --mode crossing

The input is a JSON list of ``{"T": kelvin, "value": rate}`` rows. The
value's unit is yours (a diffusivity, a growth velocity, an escape rate);
the fitted prefactor comes back in the same unit. ``arrhenius`` fits
ln(value) against 1/T. ``vft`` fits value = A exp(-B / (T - T0)) with T0 on
a grid. ``crossing`` locates where the value changes sign, which is where a
two-phase growth velocity crosses zero: the melting temperature.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_KB_EV = 8.617333262e-5  # eV/K
_EV_TO_J = 1.602176634e-19


def _load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Rows sorted by temperature, as (T, value) arrays."""
    if not path.is_file():
        raise SystemExit(f"error: no such file: {path}")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"error: {path} is not valid JSON: {e}") from e
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise SystemExit('error: expected a JSON list of {"T": ..., "value": ...} rows')
    try:
        pairs = sorted((float(r["T"]), float(r["value"])) for r in rows)
    except (KeyError, TypeError, ValueError) as e:
        raise SystemExit(f'error: every row needs numeric "T" and "value" fields: {e}') from e
    temperatures = np.array([t for t, _ in pairs])
    values = np.array([v for _, v in pairs])
    if len(np.unique(temperatures)) < len(temperatures):
        raise SystemExit("error: duplicate temperatures; average repeats before fitting")
    return temperatures, values


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(((observed - predicted) ** 2).sum())
    total = float(((observed - observed.mean()) ** 2).sum())
    return 1.0 if total == 0.0 else 1.0 - residual / total


def _fit_arrhenius(temperatures: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    if len(temperatures) < 3:
        raise SystemExit("error: an Arrhenius fit needs at least 3 temperatures")
    if (values <= 0).any():
        raise SystemExit(
            "error: Arrhenius fits ln(value); every value must be positive "
            "(fit a growth velocity's magnitude on one side of the crossing only)"
        )
    x = 1.0 / temperatures
    y = np.log(values)
    slope, intercept = np.polyfit(x, y, 1)
    ea_ev = -float(slope) * _KB_EV
    result: dict[str, Any] = {
        "mode": "arrhenius",
        "ea_eV": ea_ev,
        "ea_J": ea_ev * _EV_TO_J,
        "prefactor": float(math.exp(intercept)),
        "r2": _r_squared(y, slope * x + intercept),
        "points": len(temperatures),
        "warnings": [],
    }
    if ea_ev < 0:
        result["warnings"].append(
            "negative activation energy: the rate grows on cooling; "
            "an Arrhenius law is the wrong model for this data"
        )
    return result


def _fit_vft(temperatures: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    if len(temperatures) < 4:
        raise SystemExit("error: a VFT fit needs at least 4 temperatures")
    if (values <= 0).any():
        raise SystemExit("error: VFT fits ln(value); every value must be positive")
    y = np.log(values)
    t_min = float(temperatures.min())
    grid = np.linspace(0.0, 0.98 * t_min, 400)
    best: dict[str, Any] | None = None
    for t0 in grid:
        x = 1.0 / (temperatures - t0)
        slope, intercept = np.polyfit(x, y, 1)
        if slope >= 0:  # B must be positive for a slowing-down law
            continue
        sse = float(((y - (slope * x + intercept)) ** 2).sum())
        if best is None or sse < best["sse"]:
            best = {
                "t0": float(t0),
                "b": -float(slope),
                "prefactor": float(math.exp(intercept)),
                "sse": sse,
                "r2": _r_squared(y, slope * x + intercept),
            }
    if best is None:
        raise SystemExit(
            "error: no physical VFT fit (B > 0) exists for this data; "
            "the values do not decrease toward low temperature"
        )
    result: dict[str, Any] = {
        "mode": "vft",
        "prefactor": best["prefactor"],
        "b_K": best["b"],
        "t0_K": best["t0"],
        "r2": best["r2"],
        "points": len(temperatures),
        "warnings": [],
    }
    if best["t0"] >= grid[-2]:
        result["warnings"].append(
            "T0 sits at the edge of its search range: the data does not "
            "constrain the divergence temperature; add lower-temperature points"
        )
    if best["t0"] == 0.0:
        result["warnings"].append("best T0 is 0 K: the data is plain Arrhenius, not VFT")
    return result


def _find_crossings(temperatures: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    if len(temperatures) < 2:
        raise SystemExit("error: a crossing needs at least 2 temperatures")
    crossings: list[dict[str, float]] = []
    for i in range(len(temperatures) - 1):
        v1, v2 = float(values[i]), float(values[i + 1])
        t1, t2 = float(temperatures[i]), float(temperatures[i + 1])
        slope = (v2 - v1) / (t2 - t1)
        if v1 == 0.0:
            crossings.append({"T_K": t1, "slope_per_K": slope})
        elif v1 * v2 < 0:
            crossings.append({"T_K": t1 - v1 / slope, "slope_per_K": slope})
        if i == len(temperatures) - 2 and v2 == 0.0:
            crossings.append({"T_K": t2, "slope_per_K": slope})
    if not crossings:
        raise SystemExit(
            "error: the values never change sign; extend the temperature "
            "ladder across the transition before asking for a crossing"
        )
    result: dict[str, Any] = {
        "mode": "crossing",
        "crossings": crossings,
        "points": len(temperatures),
        "warnings": [],
    }
    if len(crossings) > 1:
        result["warnings"].append(
            f"{len(crossings)} sign changes: the data is not monotonic around "
            f"the transition; check the individual runs before quoting one"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, help='JSON list of {"T": K, "value": rate} rows')
    parser.add_argument(
        "--mode", required=True, choices=("arrhenius", "vft", "crossing"),
        help="the law to fit (crossing locates sign changes instead)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    temperatures, values = _load(args.data)
    if args.mode == "arrhenius":
        result = _fit_arrhenius(temperatures, values)
    elif args.mode == "vft":
        result = _fit_vft(temperatures, values)
    else:
        result = _find_crossings(temperatures, values)

    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    if result["mode"] == "arrhenius":
        print(
            f"Arrhenius over {result['points']} points: "
            f"Ea = {result['ea_eV']:.4f} eV = {result['ea_J']:.4e} J, "
            f"prefactor = {result['prefactor']:.4e} (input unit), R^2 = {result['r2']:.4f}"
        )
    elif result["mode"] == "vft":
        print(
            f"VFT over {result['points']} points: "
            f"prefactor = {result['prefactor']:.4e} (input unit), "
            f"B = {result['b_K']:.1f} K, T0 = {result['t0_K']:.1f} K, R^2 = {result['r2']:.4f}"
        )
    else:
        for crossing in result["crossings"]:
            print(
                f"zero crossing at T = {crossing['T_K']:.1f} K "
                f"(local slope {crossing['slope_per_K']:.4e} per K)"
            )
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
