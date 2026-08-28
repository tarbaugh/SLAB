"""Thermal conductivity and interface resistance from NEMD temperature profiles.

    fit_nemd.py kappa profile.json --flux 2.1e9
    fit_nemd.py tbr profile.dat --flux 2.1e9 --interface 42.5 --json

A profile is a JSON list of ``{"x": position, "T": kelvin}`` rows, or a
plain text file with two whitespace-separated columns (``#`` comments
allowed). Positions default to Angstrom; set ``--x-unit`` otherwise. The
heat flux is in W/m^2 and comes from the run itself: for Muller-Plathe,
the exchanged kinetic energy divided by (2 * cross-section area * time),
where the 2 counts the two half-profiles of the periodic cell.

``kappa`` fits one linear gradient and reports k = flux / |dT/dx| in
W/(m K). ``tbr`` fits a line on each side of ``--interface``, extrapolates
both to the interface, and reports R = |dT| / flux in m^2 K / W.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_X_TO_M = {"A": 1e-10, "nm": 1e-9, "um": 1e-6, "m": 1.0}


def _load_profile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """(x, T) arrays sorted by position, from JSON rows or two text columns."""
    if not path.is_file():
        raise SystemExit(f"error: no such file: {path}")
    text = path.read_text(encoding="utf-8")
    pairs: list[tuple[float, float]] = []
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        rows = None
    if rows is not None:
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            raise SystemExit('error: JSON profiles are a list of {"x": ..., "T": ...} rows')
        try:
            pairs = [(float(r["x"]), float(r["T"])) for r in rows]
        except (KeyError, TypeError, ValueError) as e:
            raise SystemExit(f'error: every row needs numeric "x" and "T" fields: {e}') from e
    else:
        for lineno, line in enumerate(text.splitlines(), start=1):
            bare = line.split("#", 1)[0].strip()
            if not bare:
                continue
            fields = bare.split()
            if len(fields) != 2:
                raise SystemExit(
                    f"error: line {lineno} has {len(fields)} column(s); "
                    f"text profiles are exactly two columns: position, temperature"
                )
            try:
                pairs.append((float(fields[0]), float(fields[1])))
            except ValueError as e:
                raise SystemExit(f"error: line {lineno} is not numeric: {e}") from e
    if not pairs:
        raise SystemExit(f"error: {path} holds no profile points")
    pairs.sort()
    return np.array([x for x, _ in pairs]), np.array([t for _, t in pairs])


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(((observed - predicted) ** 2).sum())
    total = float(((observed - observed.mean()) ** 2).sum())
    return 1.0 if total == 0.0 else 1.0 - residual / total


def _line(x_m: np.ndarray, temperatures: np.ndarray, side: str) -> tuple[float, float, float]:
    """(slope K/m, intercept K, r2) of a side's linear fit."""
    if len(x_m) < 3:
        raise SystemExit(
            f"error: only {len(x_m)} point(s) on the {side}; "
            f"each side of the interface needs at least 3 for a line"
        )
    slope, intercept = np.polyfit(x_m, temperatures, 1)
    return float(slope), float(intercept), _r_squared(temperatures, slope * x_m + intercept)


def _fit_kappa(x_m: np.ndarray, temperatures: np.ndarray, flux: float) -> dict[str, Any]:
    if len(x_m) < 4:
        raise SystemExit("error: a gradient fit needs at least 4 profile points")
    slope, _intercept, r2 = _line(x_m, temperatures, "profile")
    if abs(slope) < 1e-12:
        raise SystemExit(
            "error: the fitted gradient is zero; either the profile has not "
            "developed yet or the columns are swapped"
        )
    result: dict[str, Any] = {
        "mode": "kappa",
        "k_W_mK": flux / abs(slope),
        "gradient_K_m": slope,
        "r2": r2,
        "points": len(x_m),
        "warnings": [],
    }
    if r2 < 0.95:
        result["warnings"].append(
            f"the profile is not linear (R^2 = {r2:.3f}); trim to the linear "
            f"region between the exchange slabs and refit"
        )
    return result


def _fit_tbr(
    x_m: np.ndarray, temperatures: np.ndarray, flux: float, interface_m: float
) -> dict[str, Any]:
    if not (x_m.min() < interface_m < x_m.max()):
        raise SystemExit(
            "error: --interface lies outside the profile; it is the interface "
            "position in the same unit as the positions"
        )
    left = x_m < interface_m
    right = x_m > interface_m
    slope_l, intercept_l, r2_l = _line(x_m[left], temperatures[left], "left side")
    slope_r, intercept_r, r2_r = _line(x_m[right], temperatures[right], "right side")
    t_left = slope_l * interface_m + intercept_l
    t_right = slope_r * interface_m + intercept_r
    jump = t_left - t_right
    result: dict[str, Any] = {
        "mode": "tbr",
        "tbr_m2K_W": abs(jump) / flux,
        "dT_K": jump,
        "left": {"gradient_K_m": slope_l, "k_W_mK": flux / abs(slope_l) if slope_l else None,
                 "r2": r2_l, "points": int(left.sum())},
        "right": {"gradient_K_m": slope_r, "k_W_mK": flux / abs(slope_r) if slope_r else None,
                  "r2": r2_r, "points": int(right.sum())},
        "warnings": [],
    }
    for name, r2 in (("left", r2_l), ("right", r2_r)):
        if r2 < 0.95:
            result["warnings"].append(
                f"the {name} side is not linear (R^2 = {r2:.3f}); exclude the "
                f"points nearest the exchange slabs and the interface"
            )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("kappa", "tbr"), help="bulk gradient or interface jump")
    parser.add_argument("profile", type=Path, help="the temperature profile (JSON or 2 columns)")
    parser.add_argument("--flux", type=float, required=True, help="imposed heat flux in W/m^2")
    parser.add_argument(
        "--x-unit", default="A", choices=sorted(_X_TO_M),
        help="unit of the position column (default A)",
    )
    parser.add_argument(
        "--interface", type=float,
        help="interface position, in --x-unit (required for tbr)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if args.flux <= 0:
        raise SystemExit("error: --flux is the magnitude of the imposed flux; it must be positive")
    x, temperatures = _load_profile(args.profile)
    x_m = x * _X_TO_M[args.x_unit]
    if args.mode == "kappa":
        result = _fit_kappa(x_m, temperatures, args.flux)
    else:
        if args.interface is None:
            raise SystemExit("error: tbr needs --interface (the interface position)")
        result = _fit_tbr(x_m, temperatures, args.flux, args.interface * _X_TO_M[args.x_unit])

    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    if result["mode"] == "kappa":
        print(
            f"k = {result['k_W_mK']:.3f} W/(m K) from {result['points']} points "
            f"(gradient {result['gradient_K_m']:.4e} K/m, R^2 = {result['r2']:.4f})"
        )
    else:
        print(
            f"TBR = {result['tbr_m2K_W']:.4e} m^2 K / W "
            f"(temperature jump {result['dT_K']:+.2f} K at the interface)"
        )
        for side in ("left", "right"):
            info = result[side]
            print(
                f"  {side}: gradient {info['gradient_K_m']:.4e} K/m -> "
                f"k = {info['k_W_mK']:.3f} W/(m K), R^2 = {info['r2']:.4f}, "
                f"{info['points']} points"
            )
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
