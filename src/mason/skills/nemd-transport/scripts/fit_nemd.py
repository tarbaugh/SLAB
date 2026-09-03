"""Thermal conductivity and interface resistance from NEMD temperature profiles.

    fit_nemd.py kappa profile.json --flux 2.1e9 --drop-ends 3
    fit_nemd.py kappa profile.json --flux 2.1e9 --fold --drop-ends 3
    fit_nemd.py kappa block1.dat block2.dat block3.dat --flux 2.1e9 --xmin 20 --xmax 80
    fit_nemd.py tbr profile.dat --flux 2.1e9 --interface 42.5 --exclude-interface 8 --json

A profile is a JSON list of ``{"x": position, "T": kelvin}`` rows, or a
plain text file with two whitespace-separated columns (``#`` comments
allowed). Positions default to Angstrom; set ``--x-unit`` otherwise. The
heat flux is in W/m^2 and comes from the run itself: for Muller-Plathe in
a periodic cell, the exchanged kinetic energy divided by (2 * cross-section
area * time), where the 2 counts the two half-profiles; with fixed walls
and a source and sink there is one profile and no factor 2.

``kappa`` fits one linear gradient and reports k = flux / |dT/dx| in
W/(m K) with the standard error of the slope, the mean temperature of the
fitted window, and R^2. Choose the window with ``--xmin``/``--xmax`` (in
the position unit) or ``--drop-ends N`` (bins removed from each end, where
the exchange slabs sit). ``--fold`` treats the profile as the sawtooth of
a periodic Muller-Plathe cell: it splits it at the middle, fits each half
with its own ends dropped, and reports both gradients and their mismatch,
which must be small before the profile counts as converged. Several
profile files (time blocks) give a mean and a standard error over blocks.

``tbr`` fits a line on each side of ``--interface``, dropping the points
within ``--exclude-interface`` of it and ``--drop-ends`` from the outer
ends, extrapolates both to the interface, and reports R = |dT| / flux in
m^2 K / W and the conductance G = 1 / R in MW / (m^2 K), with each side's
gradient and implied conductivity.
"""

from __future__ import annotations

import argparse
import json
import math
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


def _line(
    x_m: np.ndarray, temperatures: np.ndarray, side: str
) -> tuple[float, float, float, float | None]:
    """(slope K/m, intercept K, r2, slope standard error) of a linear fit."""
    if len(x_m) < 3:
        raise SystemExit(
            f"error: only {len(x_m)} point(s) on the {side}; "
            f"each fitted region needs at least 3 for a line"
        )
    if len(x_m) >= 4:
        (slope, intercept), cov = np.polyfit(x_m, temperatures, 1, cov=True)
        se: float | None = float(math.sqrt(max(cov[0, 0], 0.0)))
    else:
        slope, intercept = np.polyfit(x_m, temperatures, 1)
        se = None
    predicted = slope * x_m + intercept
    return float(slope), float(intercept), _r_squared(temperatures, predicted), se


def _window(
    x: np.ndarray, temperatures: np.ndarray, xmin: float | None, xmax: float | None, drop: int
) -> tuple[np.ndarray, np.ndarray]:
    keep = np.ones(len(x), dtype=bool)
    if xmin is not None:
        keep &= x >= xmin
    if xmax is not None:
        keep &= x <= xmax
    x, temperatures = x[keep], temperatures[keep]
    if drop > 0:
        if len(x) <= 2 * drop:
            raise SystemExit(
                f"error: --drop-ends {drop} removes every point of a {len(x)}-point profile"
            )
        x, temperatures = x[drop:-drop], temperatures[drop:-drop]
    return x, temperatures


def _gradient(x_m: np.ndarray, temperatures: np.ndarray, flux: float, label: str) -> dict[str, Any]:
    if len(x_m) < 4:
        raise SystemExit(f"error: a gradient fit needs at least 4 profile points ({label})")
    slope, _intercept, r2, se = _line(x_m, temperatures, label)
    if abs(slope) < 1e-12:
        raise SystemExit(
            f"error: the fitted gradient is zero ({label}); either the profile has not "
            f"developed yet, the columns are swapped, or a sawtooth was fitted unfolded "
            f"(use --fold)"
        )
    k = flux / abs(slope)
    return {
        "k_W_mK": k,
        "k_se_W_mK": (k * se / abs(slope)) if se is not None else None,
        "gradient_K_m": slope,
        "r2": r2,
        "points": len(x_m),
        "mean_T_K": float(temperatures.mean()),
        "dT_across_window_K": float(abs(temperatures.max() - temperatures.min())),
    }


def _fit_kappa(x: np.ndarray, temperatures: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    scale = _X_TO_M[args.x_unit]
    warnings: list[str] = []
    if args.fold:
        middle = 0.5 * (x.min() + x.max())
        halves = []
        for name, keep in (("first half", x <= middle), ("second half", x > middle)):
            xs, ts = _window(x[keep], temperatures[keep], args.xmin, args.xmax, args.drop_ends)
            halves.append(_gradient(xs * scale, ts, args.flux, name))
        k1, k2 = halves[0]["k_W_mK"], halves[1]["k_W_mK"]
        mismatch = abs(k1 - k2) / (0.5 * (k1 + k2))
        result: dict[str, Any] = {
            "mode": "kappa",
            "folded": True,
            "k_W_mK": 0.5 * (k1 + k2),
            "k_se_W_mK": abs(k1 - k2) / 2.0,
            "halves": halves,
            "half_mismatch": mismatch,
            "gradient_K_m": 0.5 * (abs(halves[0]["gradient_K_m"]) + abs(halves[1]["gradient_K_m"])),
            "r2": min(h["r2"] for h in halves),
            "points": sum(h["points"] for h in halves),
            "mean_T_K": 0.5 * (halves[0]["mean_T_K"] + halves[1]["mean_T_K"]),
            "warnings": warnings,
        }
        if np.sign(halves[0]["gradient_K_m"]) == np.sign(halves[1]["gradient_K_m"]):
            warnings.append(
                "the two halves slope the same way; a periodic Muller-Plathe profile has "
                "opposite gradients, so this is not a sawtooth (drop --fold)"
            )
        if mismatch > 0.1:
            warnings.append(
                f"the two half-profiles give conductivities {mismatch * 100:.0f} % apart; "
                f"the profile has not converged, or the exchange slabs are inside the window"
            )
    else:
        xs, ts = _window(x, temperatures, args.xmin, args.xmax, args.drop_ends)
        fit = _gradient(xs * scale, ts, args.flux, "profile")
        result = {"mode": "kappa", "folded": False, **fit, "warnings": warnings}
    if result["r2"] < 0.95:
        warnings.append(
            f"the profile is not linear (R^2 = {result['r2']:.3f}); trim to the linear "
            f"region between the exchange slabs (--drop-ends or --xmin/--xmax) and refit"
        )
    if result.get("dT_across_window_K", 0.0) > 30.0 or any(
        h["dT_across_window_K"] > 30.0 for h in result.get("halves", [])
    ):
        warnings.append(
            "the fitted window spans more than 30 K; k varies across it, so report k at "
            "the window's mean temperature and keep the window narrow"
        )
    return result


def _fit_tbr(x: np.ndarray, temperatures: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    scale = _X_TO_M[args.x_unit]
    interface = args.interface
    if not (x.min() < interface < x.max()):
        raise SystemExit(
            "error: --interface lies outside the profile; it is the interface "
            "position in the same unit as the positions"
        )
    left = x < interface - args.exclude_interface
    right = x > interface + args.exclude_interface
    if args.drop_ends > 0:
        idx_left = np.nonzero(left)[0]
        idx_right = np.nonzero(right)[0]
        left[idx_left[: args.drop_ends]] = False
        right[idx_right[len(idx_right) - args.drop_ends :]] = False
    interface_m = interface * scale
    slope_l, intercept_l, r2_l, se_l = _line(x[left] * scale, temperatures[left], "left side")
    slope_r, intercept_r, r2_r, se_r = _line(x[right] * scale, temperatures[right], "right side")
    t_left = slope_l * interface_m + intercept_l
    t_right = slope_r * interface_m + intercept_r
    jump = t_left - t_right
    tbr = abs(jump) / args.flux
    result: dict[str, Any] = {
        "mode": "tbr",
        "tbr_m2K_W": tbr,
        "conductance_MW_m2K": (1.0 / tbr) * 1e-6 if tbr > 0 else None,
        "dT_K": jump,
        "excluded_within": args.exclude_interface,
        "left": {"gradient_K_m": slope_l, "k_W_mK": args.flux / abs(slope_l) if slope_l else None,
                 "gradient_se_K_m": se_l, "r2": r2_l, "points": int(left.sum())},
        "right": {"gradient_K_m": slope_r, "k_W_mK": args.flux / abs(slope_r) if slope_r else None,
                  "gradient_se_K_m": se_r, "r2": r2_r, "points": int(right.sum())},
        "warnings": [],
    }
    for name, r2 in (("left", r2_l), ("right", r2_r)):
        if r2 < 0.95:
            result["warnings"].append(
                f"the {name} side is not linear (R^2 = {r2:.3f}); exclude the "
                f"points nearest the exchange slabs (--drop-ends) and the interface "
                f"(--exclude-interface)"
            )
    if args.exclude_interface == 0.0:
        result["warnings"].append(
            "no points were excluded next to the interface; the bins adjacent to it are "
            "the least linear, so set --exclude-interface to a few bin widths"
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("kappa", "tbr"), help="bulk gradient or interface jump")
    parser.add_argument(
        "profile", type=Path, nargs="+",
        help="the temperature profile(s) (JSON or 2 columns); several files are time blocks",
    )
    parser.add_argument("--flux", type=float, required=True, help="imposed heat flux in W/m^2")
    parser.add_argument(
        "--x-unit", default="A", choices=sorted(_X_TO_M),
        help="unit of the position column (default A)",
    )
    parser.add_argument("--xmin", type=float, help="fit only positions at or above this")
    parser.add_argument("--xmax", type=float, help="fit only positions at or below this")
    parser.add_argument(
        "--drop-ends", type=int, default=0,
        help="bins to drop from each end of the fitted region (the exchange slabs)",
    )
    parser.add_argument(
        "--fold", action="store_true",
        help="kappa: split a periodic Muller-Plathe sawtooth at the middle and fit each half",
    )
    parser.add_argument(
        "--interface", type=float,
        help="interface position, in --x-unit (required for tbr)",
    )
    parser.add_argument(
        "--exclude-interface", type=float, default=0.0,
        help="tbr: drop points within this distance (in --x-unit) of the interface",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if args.flux <= 0:
        raise SystemExit("error: --flux is the magnitude of the imposed flux; it must be positive")
    if args.drop_ends < 0 or args.exclude_interface < 0:
        raise SystemExit("error: --drop-ends and --exclude-interface must be 0 or positive")
    if args.mode == "tbr" and args.interface is None:
        raise SystemExit("error: tbr needs --interface (the interface position)")
    if args.mode == "tbr" and args.fold:
        raise SystemExit("error: --fold applies to kappa only")

    results: list[dict[str, Any]] = []
    for path in args.profile:
        x, temperatures = _load_profile(path)
        if args.mode == "kappa":
            fit = _fit_kappa(x, temperatures, args)
        else:
            fit = _fit_tbr(x, temperatures, args)
        fit["file"] = str(path)
        results.append(fit)
    result = dict(results[0])
    if len(results) > 1:
        key = "k_W_mK" if args.mode == "kappa" else "tbr_m2K_W"
        values = np.array([r[key] for r in results])
        result["blocks"] = [{"file": r["file"], key: r[key]} for r in results]
        result[key] = float(values.mean())
        result[f"{key}_block_se"] = float(values.std(ddof=1) / math.sqrt(len(values)))
        if args.mode == "tbr":
            result["conductance_MW_m2K"] = 1e-6 / float(values.mean())
        result["warnings"] = sorted({w for r in results for w in r["warnings"]})

    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    if result["mode"] == "kappa":
        se = result.get("k_W_mK_block_se") or result.get("k_se_W_mK")
        se_text = f" +/- {se:.3f}" if se is not None else ""
        print(
            f"k = {result['k_W_mK']:.3f}{se_text} W/(m K) at <T> = {result['mean_T_K']:.1f} K "
            f"from {result['points']} points (gradient {result['gradient_K_m']:.4e} K/m, "
            f"R^2 = {result['r2']:.4f})"
        )
        if result.get("folded"):
            for name, half in zip(("first half", "second half"), result["halves"], strict=True):
                print(
                    f"  {name}: k = {half['k_W_mK']:.3f} W/(m K), gradient "
                    f"{half['gradient_K_m']:+.4e} K/m, {half['points']} points"
                )
            print(f"  half-profile mismatch {result['half_mismatch'] * 100:.1f} %")
        if "blocks" in result:
            print(f"  ({len(result['blocks'])} time blocks; the error is their standard error)")
    else:
        se = result.get("tbr_m2K_W_block_se")
        se_text = f" +/- {se:.2e}" if se is not None else ""
        print(
            f"TBR = {result['tbr_m2K_W']:.4e}{se_text} m^2 K / W = G "
            f"{result['conductance_MW_m2K']:.1f} MW / (m^2 K) "
            f"(temperature jump {result['dT_K']:+.2f} K at the interface, points within "
            f"{result['excluded_within']:g} excluded)"
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
