"""Classical nucleation theory: interfacial energy and nucleation barriers.

    cnt.py gamma rstar.json --tm 823 --dhf 2.4e8
    cnt.py barrier --gamma 0.062 --tm 823 --dhf 2.4e8 --temps 500 600 700 \
        --f-het 0.32 --omega 30.5 --json

``gamma`` inverts the Gibbs-Thomson critical radius: given seeded-crystal
grow/shrink brackets as a JSON list of ``{"T": kelvin, "r_star_nm": ...}``
rows, plus the melting temperature ``--tm`` (K) and the volumetric latent
heat ``--dhf`` (J/m^3), each row yields

    gamma = r* dHf (Tm - T) / (2 Tm)     [J/m^2]

``barrier`` runs the algebra forward: with ``--gamma`` it reports, at each
requested undercooled temperature, the driving force, the critical radius,
the barrier dG* = 16 pi gamma^3 / (3 dGv^2) in eV and in kT, and (with
``--omega``, the atomic volume in A^3) the atom count of the critical
nucleus. ``--f-het`` scales the barrier for heterogeneous nucleation.
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


def _gamma(args: argparse.Namespace) -> dict[str, Any]:
    if not args.data.is_file():
        raise SystemExit(f"error: no such file: {args.data}")
    try:
        rows = json.loads(args.data.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"error: {args.data} is not valid JSON: {e}") from e
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise SystemExit('error: expected a JSON list of {"T": ..., "r_star_nm": ...} rows')
    try:
        pairs = sorted((float(r["T"]), float(r["r_star_nm"])) for r in rows)
    except (KeyError, TypeError, ValueError) as e:
        raise SystemExit(f'error: every row needs numeric "T" and "r_star_nm": {e}') from e
    if not pairs:
        raise SystemExit("error: the file holds no rows")
    points: list[dict[str, float]] = []
    for temperature, r_star_nm in pairs:
        if temperature >= args.tm:
            raise SystemExit(
                f"error: T = {temperature:g} K is not below Tm = {args.tm:g} K; "
                f"a critical radius exists only in the undercooled regime"
            )
        if r_star_nm <= 0:
            raise SystemExit("error: every r_star_nm must be positive")
        gamma = (r_star_nm * 1e-9) * args.dhf * (args.tm - temperature) / (2.0 * args.tm)
        points.append({"T_K": temperature, "r_star_nm": r_star_nm, "gamma_J_m2": gamma})
    gammas = np.array([p["gamma_J_m2"] for p in points])
    result: dict[str, Any] = {
        "mode": "gamma",
        "points": points,
        "gamma_J_m2": float(gammas.mean()),
        "gamma_std_J_m2": float(gammas.std()),
        "warnings": [],
    }
    if len(points) >= 3:
        temperatures = np.array([p["T_K"] for p in points])
        correlation = float(np.corrcoef(temperatures, gammas)[0, 1])
        if abs(correlation) > 0.8:
            result["warnings"].append(
                f"gamma trends with temperature (correlation {correlation:+.2f}): "
                f"the linear driving-force approximation dGv = dHf dT/Tm is "
                f"straining; quote gamma(T), not one number"
            )
    return result


def _barrier(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    for temperature in args.temps:
        if temperature >= args.tm:
            raise SystemExit(
                f"error: T = {temperature:g} K is not below Tm = {args.tm:g} K; "
                f"there is no barrier without undercooling"
            )
        dgv = args.dhf * (args.tm - temperature) / args.tm  # J/m^3
        r_star_m = 2.0 * args.gamma / dgv
        barrier_j = 16.0 * math.pi * args.gamma**3 / (3.0 * dgv**2) * args.f_het
        barrier_ev = barrier_j / _EV_TO_J
        row: dict[str, float] = {
            "T_K": temperature,
            "undercooling_K": args.tm - temperature,
            "r_star_nm": r_star_m * 1e9,
            "dG_star_eV": barrier_ev,
            "dG_star_over_kT": barrier_ev / (_KB_EV * temperature),
        }
        if args.omega is not None:
            volume_m3 = (4.0 / 3.0) * math.pi * r_star_m**3
            row["n_star_atoms"] = volume_m3 / (args.omega * 1e-30)
        rows.append(row)
    return {"mode": "barrier", "f_het": args.f_het, "rows": rows, "warnings": []}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="mode", required=True)

    gamma_parser = subparsers.add_parser("gamma", help="gamma from r*(T) brackets")
    gamma_parser.add_argument("data", type=Path, help='JSON list of {"T", "r_star_nm"} rows')

    barrier_parser = subparsers.add_parser("barrier", help="barriers from gamma")
    barrier_parser.add_argument("--gamma", type=float, required=True,
                                help="interfacial free energy (J/m^2)")
    barrier_parser.add_argument("--temps", type=float, nargs="+", required=True,
                                help="temperatures to evaluate (K)")
    barrier_parser.add_argument("--f-het", type=float, default=1.0,
                                help="heterogeneous potency factor in (0, 1] (default 1)")
    barrier_parser.add_argument("--omega", type=float,
                                help="atomic volume (A^3), to report the critical atom count")

    for sub in (gamma_parser, barrier_parser):
        sub.add_argument("--tm", type=float, required=True, help="melting temperature (K)")
        sub.add_argument("--dhf", type=float, required=True,
                         help="volumetric latent heat of fusion (J/m^3)")
        sub.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if args.tm <= 0 or args.dhf <= 0:
        raise SystemExit("error: --tm and --dhf must be positive")
    if args.mode == "barrier":
        if args.gamma <= 0:
            raise SystemExit("error: --gamma must be positive")
        if not 0.0 < args.f_het <= 1.0:
            raise SystemExit("error: --f-het is the CNT potency factor; it lies in (0, 1]")
        if args.omega is not None and args.omega <= 0:
            raise SystemExit("error: --omega is an atomic volume in A^3; it must be positive")
        result = _barrier(args)
    else:
        result = _gamma(args)

    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    if result["mode"] == "gamma":
        for point in result["points"]:
            print(
                f"T = {point['T_K']:7.1f} K, r* = {point['r_star_nm']:6.2f} nm "
                f"-> gamma = {point['gamma_J_m2']:.4f} J/m^2"
            )
        print(
            f"gamma = {result['gamma_J_m2']:.4f} +/- {result['gamma_std_J_m2']:.4f} J/m^2 "
            f"({len(result['points'])} bracket(s))"
        )
    else:
        for row in result["rows"]:
            line = (
                f"T = {row['T_K']:7.1f} K (dT = {row['undercooling_K']:5.1f}): "
                f"r* = {row['r_star_nm']:6.2f} nm, dG* = {row['dG_star_eV']:8.3f} eV "
                f"= {row['dG_star_over_kT']:8.1f} kT"
            )
            if "n_star_atoms" in row:
                line += f", n* = {row['n_star_atoms']:.0f} atoms"
            print(line)
        if result["f_het"] < 1.0:
            print(f"(barriers already scaled by f_het = {result['f_het']:g})")
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
