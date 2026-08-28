"""Heat capacity, thermal expansion, and latent heat from NPT ramps.

    fit_thermal_ramp.py ramp.json
    fit_thermal_ramp.py ramp.json --window 400 700 --json
    fit_thermal_ramp.py crystal.json --other liquid.json --at 823

A ramp is a JSON list of ``{"T": kelvin, "E": eV, "V": A^3}`` rows, one per
temperature rung (total energy and mean volume of the whole cell). The
single-file mode fits linear slopes over ``--window`` and reports the
volumetric heat capacity c_p = (dE/dT)/V in J/(m^3 K) and the linear
thermal expansion coefficient (dV/dT)/(3V) in 1/K. With ``--other`` and
``--at T``, it instead reports the enthalpy difference between the two
ramps at T: the latent heat, in eV per cell and J/m^3 of the first
(reference) file's phase.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_EV_TO_J = 1.602176634e-19
_A3_TO_M3 = 1e-30


def _load(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(T, E, V) arrays sorted by temperature."""
    if not path.is_file():
        raise SystemExit(f"error: no such file: {path}")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"error: {path} is not valid JSON: {e}") from e
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise SystemExit('error: expected a JSON list of {"T", "E", "V"} rows')
    try:
        triples = sorted((float(r["T"]), float(r["E"]), float(r["V"])) for r in rows)
    except (KeyError, TypeError, ValueError) as e:
        raise SystemExit(f'error: every row needs numeric "T", "E", "V" fields: {e}') from e
    if any(v <= 0 for _, _, v in triples):
        raise SystemExit("error: every volume must be positive")
    return (
        np.array([t for t, _, _ in triples]),
        np.array([e for _, e, _ in triples]),
        np.array([v for _, _, v in triples]),
    )


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(((observed - predicted) ** 2).sum())
    total = float(((observed - observed.mean()) ** 2).sum())
    return 1.0 if total == 0.0 else 1.0 - residual / total


def _interpolate(path: Path, at: float) -> tuple[float, float]:
    """(E, V) of a ramp linearly interpolated at temperature *at*."""
    temperatures, energies, volumes = _load(path)
    if not (temperatures.min() <= at <= temperatures.max()):
        raise SystemExit(
            f"error: T = {at:g} K is outside {path}'s ladder "
            f"({temperatures.min():g}-{temperatures.max():g} K); "
            f"extend the ladder instead of extrapolating a latent heat"
        )
    return (
        float(np.interp(at, temperatures, energies)),
        float(np.interp(at, temperatures, volumes)),
    )


def _slopes(
    temperatures: np.ndarray, energies: np.ndarray, volumes: np.ndarray,
    window: tuple[float, float] | None,
) -> dict[str, Any]:
    if window is not None:
        low, high = sorted(window)
        keep = (temperatures >= low) & (temperatures <= high)
        temperatures, energies, volumes = temperatures[keep], energies[keep], volumes[keep]
    if len(temperatures) < 3:
        raise SystemExit(
            f"error: {len(temperatures)} rung(s) in the window; "
            f"a slope worth trusting needs at least 3"
        )
    de_dt, e_intercept = np.polyfit(temperatures, energies, 1)
    dv_dt, v_intercept = np.polyfit(temperatures, volumes, 1)
    mean_volume = float(volumes.mean())
    result: dict[str, Any] = {
        "mode": "slopes",
        "window_K": [float(temperatures.min()), float(temperatures.max())],
        "rungs": len(temperatures),
        "cp_total_eV_K": float(de_dt),
        "cp_vol_J_m3K": float(de_dt) * _EV_TO_J / (mean_volume * _A3_TO_M3),
        "cte_per_K": float(dv_dt) / (3.0 * mean_volume),
        "r2_E": _r_squared(energies, de_dt * temperatures + e_intercept),
        "r2_V": _r_squared(volumes, dv_dt * temperatures + v_intercept),
        "warnings": [],
    }
    for name, r2 in (("E(T)", result["r2_E"]), ("V(T)", result["r2_V"])):
        if r2 < 0.9:
            result["warnings"].append(
                f"{name} is not linear over the window (R^2 = {r2:.3f}); narrow "
                f"the window, or check whether a phase change sits inside it"
            )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, help="ramp.json for one phase")
    parser.add_argument(
        "--window", type=float, nargs=2, metavar=("TLOW", "THIGH"),
        help="fit slopes over this temperature range only",
    )
    parser.add_argument(
        "--other", type=Path,
        help="a second phase's ramp; switches to the latent-heat mode",
    )
    parser.add_argument(
        "--at", type=float,
        help="temperature (K) where the two ramps are compared (with --other)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if (args.other is None) != (args.at is None):
        raise SystemExit("error: --other and --at come together: two ramps, one temperature")

    if args.other is not None and args.at is not None:
        e_ref, v_ref = _interpolate(args.data, args.at)
        e_other, _ = _interpolate(args.other, args.at)
        result: dict[str, Any] = {
            "mode": "latent",
            "at_K": args.at,
            "dH_eV": e_other - e_ref,
            "dH_J_m3": (e_other - e_ref) * _EV_TO_J / (v_ref * _A3_TO_M3),
            "reference_volume_A3": v_ref,
            "warnings": [],
        }
    else:
        window = (args.window[0], args.window[1]) if args.window else None
        result = _slopes(*_load(args.data), window)

    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    if result["mode"] == "latent":
        print(
            f"dH at {result['at_K']:g} K = {result['dH_eV']:+.4f} eV per cell "
            f"= {result['dH_J_m3']:+.4e} J/m^3 "
            f"(per the reference file's volume, {result['reference_volume_A3']:.2f} A^3)"
        )
    else:
        low, high = result["window_K"]
        print(
            f"slopes over {low:g}-{high:g} K ({result['rungs']} rungs): "
            f"c_p = {result['cp_vol_J_m3K']:.4e} J/(m^3 K) "
            f"[dE/dT = {result['cp_total_eV_K']:.4e} eV/K, R^2 = {result['r2_E']:.4f}], "
            f"CTE = {result['cte_per_K']:.4e} 1/K [R^2 = {result['r2_V']:.4f}]"
        )
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
