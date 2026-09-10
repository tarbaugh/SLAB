"""Heat capacity, thermal expansion, and latent heat from NPT ramps.

    fit_thermal_ramp.py ramp.json
    fit_thermal_ramp.py ramp.json --window 400 700 --json
    fit_thermal_ramp.py crystal.json --other liquid.json --at 823

A ramp is a JSON list of rows, one per temperature rung, with ``"T"``,
``"V"`` (A^3, the whole cell), and ``"H"`` (eV, the enthalpy E + PV) or
``"E"`` (total energy; the PV term is added when ``"P_bar"`` is present).
Rows may also carry ``"T_measured"`` (used for the fit when present),
``"N"`` (atoms), ``"mass_amu"``, ``"H_se"`` and ``"V_se"`` (block standard
errors), ``"L"`` (mean cell lengths), and ``"direction"`` (up or down).

The single-file mode fits linear slopes over ``--window`` and reports the
heat capacity c_p = dH/dT per atom in k_B, in J/(mol K), in J/(kg K) when
the mass is known, and per volume in J/(m^3 K); the linear expansion
coefficient (dV/dT)/(3V), and one per axis from the cell lengths when
present. Slopes carry standard errors from the fit's covariance (four or
more rungs). An upward and a downward branch are compared at shared
temperatures for hysteresis. With ``--other`` and ``--at T``, it instead
reports the enthalpy difference between the two ramps at T: the latent
heat, per atom (eV and kJ/mol), per cell, and per volume of the first
(reference) file's phase. The two ramps must hold the same atom count.
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
_KB_EV = 8.617333262e-5
_AVOGADRO = 6.02214076e23
_AMU_KG = 1.66053906660e-27
_BAR_TO_EV_A3 = 6.241509074e-7
MIN_RUNGS = 5


def _load(path: Path) -> list[dict[str, Any]]:
    """Rows sorted by temperature, with H derived when only E is recorded."""
    if not path.is_file():
        raise SystemExit(f"error: no such file: {path}")
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"error: {path} is not valid JSON: {e}") from e
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise SystemExit('error: expected a JSON list of {"T", "H" or "E", "V"} rows')
    clean: list[dict[str, Any]] = []
    for r in rows:
        try:
            volume = float(r["V"])
            row: dict[str, Any] = {
                "T": float(r["T"]),
                "T_fit": float(r.get("T_measured", r["T"])),
                "measured": "T_measured" in r,
                "V": volume,
                "direction": str(r.get("direction", "up")),
                "N": int(r["N"]) if "N" in r else None,
                "mass_amu": float(r["mass_amu"]) if "mass_amu" in r else None,
                "H_se": float(r["H_se"]) if "H_se" in r else None,
                "V_se": float(r["V_se"]) if "V_se" in r else None,
                "L": [float(x) for x in r["L"]] if "L" in r else None,
            }
            if "H" in r:
                row["H"] = float(r["H"])
            elif "P_bar" in r:
                row["H"] = float(r["E"]) + float(r["P_bar"]) * _BAR_TO_EV_A3 * volume
            else:
                row["H"] = float(r["E"])
        except (KeyError, TypeError, ValueError) as e:
            raise SystemExit(
                f'error: every row needs numeric "T", "V", and "H" or "E" fields: {e}'
            ) from e
        if volume <= 0:
            raise SystemExit("error: every volume must be positive")
        clean.append(row)
    return sorted(clean, key=lambda row: (row["direction"] != "up", row["T"]))


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual = float(((observed - predicted) ** 2).sum())
    total = float(((observed - observed.mean()) ** 2).sum())
    return 1.0 if total == 0.0 else 1.0 - residual / total


def _line(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float | None]:
    """Slope, intercept, and the slope's standard error (None below 4 points)."""
    if len(x) >= 4:
        (slope, intercept), cov = np.polyfit(x, y, 1, cov=True)
        return float(slope), float(intercept), float(np.sqrt(cov[0, 0]))
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept), None


def _interpolate(rows: list[dict[str, Any]], at: float, path: Path) -> tuple[float, float]:
    """(H, V) of the upward branch linearly interpolated at temperature *at*."""
    up = [r for r in rows if r["direction"] == "up"]
    temperatures = np.array([r["T"] for r in up])
    if not (temperatures.min() <= at <= temperatures.max()):
        raise SystemExit(
            f"error: T = {at:g} K is outside {path}'s ladder "
            f"({temperatures.min():g}-{temperatures.max():g} K); "
            f"extend the ladder instead of extrapolating a latent heat"
        )
    return (
        float(np.interp(at, temperatures, [r["H"] for r in up])),
        float(np.interp(at, temperatures, [r["V"] for r in up])),
    )


def _slopes(rows: list[dict[str, Any]], window: tuple[float, float] | None) -> dict[str, Any]:
    up = [r for r in rows if r["direction"] == "up"]
    down = [r for r in rows if r["direction"] == "down"]
    if window is not None:
        low, high = sorted(window)
        up = [r for r in up if low <= r["T"] <= high]
        down = [r for r in down if low <= r["T"] <= high]
    if len(up) < 3:
        raise SystemExit(
            f"error: {len(up)} rung(s) in the window; a slope worth trusting needs at least 3"
        )
    temperatures = np.array([r["T_fit"] for r in up])
    enthalpies = np.array([r["H"] for r in up])
    volumes = np.array([r["V"] for r in up])
    dh_dt, h_intercept, dh_se = _line(temperatures, enthalpies)
    dv_dt, v_intercept, dv_se = _line(temperatures, volumes)
    mean_volume = float(volumes.mean())
    n_atoms = up[0]["N"]
    mass_amu = up[0]["mass_amu"]
    result: dict[str, Any] = {
        "mode": "slopes",
        "window_K": [float(min(r["T"] for r in up)), float(max(r["T"] for r in up))],
        "rungs": len(up),
        "fitted_temperature": "measured" if all(r["measured"] for r in up) else "target",
        "cp_total_eV_K": dh_dt,
        "cp_total_se_eV_K": dh_se,
        "cp_vol_J_m3K": dh_dt * _EV_TO_J / (mean_volume * _A3_TO_M3),
        "cte_per_K": dv_dt / (3.0 * mean_volume),
        "cte_se_per_K": (dv_se / (3.0 * mean_volume)) if dv_se is not None else None,
        "r2_H": _r_squared(enthalpies, dh_dt * temperatures + h_intercept),
        "r2_V": _r_squared(volumes, dv_dt * temperatures + v_intercept),
        "warnings": [],
    }
    if n_atoms:
        result["n_atoms"] = n_atoms
        result["cp_kB_per_atom"] = dh_dt / (n_atoms * _KB_EV)
        result["cp_J_molK"] = dh_dt / n_atoms * _EV_TO_J * _AVOGADRO
        if dh_se is not None:
            result["cp_kB_per_atom_se"] = dh_se / (n_atoms * _KB_EV)
        if mass_amu:
            result["cp_J_kgK"] = dh_dt * _EV_TO_J / (mass_amu * _AMU_KG)
        if result["cp_kB_per_atom"] > 4.5 or result["cp_kB_per_atom"] < 2.0:
            result["warnings"].append(
                f"c_p = {result['cp_kB_per_atom']:.2f} k_B per atom is far from the "
                f"classical 3 k_B (plus a modest anharmonic and c_p - c_v excess); a phase "
                f"change or an unequilibrated rung is likely inside the window"
            )
    else:
        result["warnings"].append(
            "rows carry no atom count (N), so c_p is per cell and per volume only"
        )
    if all(r["L"] is not None for r in up):
        lengths = np.array([r["L"] for r in up])
        per_axis = []
        for k in range(3):
            slope, _, _ = _line(temperatures, lengths[:, k])
            per_axis.append(slope / float(lengths[:, k].mean()))
        result["cte_per_axis_per_K"] = per_axis
        spread = max(per_axis) - min(per_axis)
        if spread > 0.2 * abs(result["cte_per_K"]) and spread > 1e-7:
            result["warnings"].append(
                "the expansion differs between axes by more than 20 %: report the "
                "per-axis coefficients, not the isotropic average"
            )
    for name, r2 in (("H(T)", result["r2_H"]), ("V(T)", result["r2_V"])):
        if r2 < 0.9:
            result["warnings"].append(
                f"{name} is not linear over the window (R^2 = {r2:.3f}); narrow "
                f"the window, or check whether a phase change sits inside it"
            )
    if len(up) < MIN_RUNGS:
        result["warnings"].append(
            f"{len(up)} rungs cannot show a kink; use at least {MIN_RUNGS} per window for "
            f"a reportable slope"
        )
    if down:
        differences = []
        for r in down:
            match = next((u for u in up if abs(u["T"] - r["T"]) < 1e-9), None)
            if match is None:
                continue
            gap = r["H"] - match["H"]
            scale = 3.0 * max(
                (match["H_se"] or 0.0) + (r["H_se"] or 0.0), 1e-3 * abs(dh_dt) * 1.0
            )
            differences.append({"T": r["T"], "dH_eV": gap, "tolerance_eV": scale})
        result["hysteresis"] = differences
        if any(abs(d["dH_eV"]) > d["tolerance_eV"] for d in differences):
            result["warnings"].append(
                "the downward branch does not retrace the upward one: hysteresis "
                "(superheating or supercooling) inside the ladder; the slopes across "
                "that range are not equilibrium values"
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
        reference = _load(args.data)
        other = _load(args.other)
        n_ref, n_other = reference[0]["N"], other[0]["N"]
        if n_ref is None or n_other is None:
            raise SystemExit(
                "error: a latent heat needs the atom count (N) in both ramps; per-cell "
                "differences between unequal cells mean nothing"
            )
        if n_ref != n_other:
            raise SystemExit(
                f"error: {args.data} holds {n_ref} atoms and {args.other} holds {n_other}; "
                f"a latent heat compares equal cells, or per atom from equal cells"
            )
        h_ref, v_ref = _interpolate(reference, args.at, args.data)
        h_other, _ = _interpolate(other, args.at, args.other)
        d_h = h_other - h_ref
        result: dict[str, Any] = {
            "mode": "latent",
            "at_K": args.at,
            "n_atoms": n_ref,
            "dH_eV_per_atom": d_h / n_ref,
            "dH_kJ_mol": d_h / n_ref * _EV_TO_J * _AVOGADRO / 1000.0,
            "dH_eV": d_h,
            "dH_J_m3": d_h * _EV_TO_J / (v_ref * _A3_TO_M3),
            "reference_volume_A3": v_ref,
            "warnings": [],
        }
    else:
        window = (args.window[0], args.window[1]) if args.window else None
        result = _slopes(_load(args.data), window)

    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    if result["mode"] == "latent":
        print(
            f"dH at {result['at_K']:g} K = {result['dH_eV_per_atom']:+.4f} eV/atom "
            f"= {result['dH_kJ_mol']:+.3f} kJ/mol = {result['dH_eV']:+.4f} eV per "
            f"{result['n_atoms']}-atom cell = {result['dH_J_m3']:+.4e} J/m^3 "
            f"(per the reference file's volume, {result['reference_volume_A3']:.2f} A^3)"
        )
    else:
        low, high = result["window_K"]
        se = f" +/- {result['cp_total_se_eV_K']:.2e}" if result["cp_total_se_eV_K"] else ""
        print(
            f"slopes over {low:g}-{high:g} K ({result['rungs']} rungs, "
            f"{result['fitted_temperature']} temperatures): "
            f"dH/dT = {result['cp_total_eV_K']:.4e}{se} eV/K [R^2 = {result['r2_H']:.4f}]"
        )
        if "cp_kB_per_atom" in result:
            se_kb = (
                f" +/- {result['cp_kB_per_atom_se']:.2f}" if "cp_kB_per_atom_se" in result else ""
            )
            kg = f", {result['cp_J_kgK']:.1f} J/(kg K)" if "cp_J_kgK" in result else ""
            print(
                f"c_p = {result['cp_kB_per_atom']:.3f}{se_kb} k_B/atom = "
                f"{result['cp_J_molK']:.2f} J/(mol K){kg}, "
                f"{result['cp_vol_J_m3K']:.4e} J/(m^3 K)"
            )
        else:
            print(f"c_p = {result['cp_vol_J_m3K']:.4e} J/(m^3 K)")
        cte_se = f" +/- {result['cte_se_per_K']:.1e}" if result["cte_se_per_K"] else ""
        print(f"linear CTE = {result['cte_per_K']:.4e}{cte_se} 1/K [R^2 = {result['r2_V']:.4f}]")
        if "cte_per_axis_per_K" in result:
            axes = ", ".join(f"{v:.3e}" for v in result["cte_per_axis_per_K"])
            print(f"CTE per axis (a, b, c): {axes} 1/K")
        if result.get("hysteresis"):
            for d in result["hysteresis"]:
                print(f"hysteresis at {d['T']:g} K: dH(down - up) = {d['dH_eV']:+.4f} eV")
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
