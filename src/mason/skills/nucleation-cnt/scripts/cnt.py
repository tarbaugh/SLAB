"""Classical nucleation theory: interfacial energy, barriers, and rates.

    cnt.py gamma seeds.json --tm 823 --dhf 2.4e8 --omega 30.5
    cnt.py barrier --gamma 0.062 --gamma-slope 1.5e-4 --tm 823 --dhf 2.4e8 \
        --temps 700 750 --theta 60 --omega 30.5 --json
    cnt.py rate --gamma 0.062 --tm 823 --dhf 2.4e8 --temps 700 --omega 30.5 \
        --attachment 2e11

``gamma`` reads seeded-crystal brackets as a JSON list of rows with
``"T"`` (K) and either ``"n_star"`` (the critical cluster's atom count
by the order parameter that defined it; ``"n_low"``/``"n_high"`` give the
grow/shrink bracket) or ``"r_star_nm"`` (``"r_low_nm"``/``"r_high_nm"``
for the bracket). With the melting temperature ``--tm`` (K), the
volumetric latent heat ``--dhf`` (J/m^3), and the atomic volume of the
solid ``--omega`` (A^3), each row yields

    gamma = (3 N* rho_s^2 dmu^3 / (32 pi))^(1/3)      [J/m^2]

with rho_s = 1/omega and dmu = dGv omega, or gamma = r* dGv / 2 for a
radius. gamma from seeding rises toward Tm, so the rows are fitted as
gamma(T) = gamma_m + slope (T - Tm) and gamma at Tm is reported with the
slope; one mean number is not the quantity to carry forward.

``barrier`` runs the algebra forward at each requested temperature with
gamma(T) from ``--gamma`` at ``--gamma-ref-t`` (default Tm) and
``--gamma-slope``: the driving force, the critical radius, the barrier
dG* = 16 pi gamma^3 / (3 dGv^2) in eV and in kT, and with ``--omega`` the
atom count of the critical nucleus. ``--theta`` (degrees, from the
interface-adhesion skill) or ``--f-het`` scales the barrier and the atom
count for heterogeneous nucleation. ``rate`` adds the Zeldovich factor,
and with ``--attachment`` (f+, 1/s) and the liquid density the rate
J = rho_l Z f+ exp(-dG*/kT) in 1/(m^3 s).

The driving force is dGv = dHf (Tm - T)/Tm unless ``--dmu-table`` gives
``{"T", "dgv_J_m3"}`` rows from a Gibbs-Helmholtz integration of the two
phases' enthalpies; the linear form is warned about past 20 %
undercooling.
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
_KB_J = 1.380649e-23
_EV_TO_J = 1.602176634e-19
LINEAR_LIMIT = 0.2


def _driving_force(args: argparse.Namespace, temperature: float, warnings: list[str]) -> float:
    """dGv in J/m^3 at *temperature*, from the table when given."""
    if temperature >= args.tm:
        raise SystemExit(
            f"error: T = {temperature:g} K is not below Tm = {args.tm:g} K; "
            f"a critical nucleus exists only in the undercooled regime"
        )
    table = getattr(args, "_dmu", None)
    if table is not None:
        temperatures, values = table
        if not (temperatures.min() <= temperature <= temperatures.max()):
            raise SystemExit(
                f"error: T = {temperature:g} K is outside the --dmu-table range "
                f"({temperatures.min():g}-{temperatures.max():g} K)"
            )
        return float(np.interp(temperature, temperatures, values))
    undercooling = (args.tm - temperature) / args.tm
    if undercooling > LINEAR_LIMIT:
        note = (
            f"dGv = dHf (Tm - T)/Tm at {undercooling * 100:.0f} % undercooling overestimates "
            f"the driving force (dHf falls with dCp dT); pass --dmu-table from the two "
            f"phases' enthalpies"
        )
        if note not in warnings:
            warnings.append(note)
    return float(args.dhf * (args.tm - temperature) / args.tm)


def _load_dmu(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"error: cannot read --dmu-table {path}: {e}") from e
    try:
        pairs = sorted((float(r["T"]), float(r["dgv_J_m3"])) for r in rows)
    except (KeyError, TypeError, ValueError) as e:
        raise SystemExit(f'error: --dmu-table rows need numeric "T" and "dgv_J_m3": {e}') from e
    if len(pairs) < 2:
        raise SystemExit("error: --dmu-table needs at least 2 rows")
    return np.array([t for t, _ in pairs]), np.array([v for _, v in pairs])


def _f_het(args: argparse.Namespace) -> float:
    if args.theta is not None:
        cos = math.cos(math.radians(args.theta))
        return (2.0 + cos) * (1.0 - cos) ** 2 / 4.0
    return float(args.f_het)


def _gamma(args: argparse.Namespace) -> dict[str, Any]:
    if not args.data.is_file():
        raise SystemExit(f"error: no such file: {args.data}")
    try:
        rows = json.loads(args.data.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"error: {args.data} is not valid JSON: {e}") from e
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows) or not rows:
        raise SystemExit(
            'error: expected a JSON list of {"T", "n_star" or "r_star_nm"} rows'
        )
    warnings: list[str] = []
    points: list[dict[str, Any]] = []
    rho_s = 1.0 / (args.omega * 1e-30) if args.omega else None
    for r in sorted(rows, key=lambda r: float(r.get("T", 0.0))):
        try:
            temperature = float(r["T"])
        except (KeyError, TypeError, ValueError) as e:
            raise SystemExit(f'error: every row needs a numeric "T": {e}') from e
        dgv = _driving_force(args, temperature, warnings)
        point: dict[str, Any] = {"T_K": temperature, "dGv_J_m3": dgv}
        if "n_star" in r:
            if rho_s is None:
                raise SystemExit("error: rows with n_star need --omega (the atomic volume, A^3)")
            n_star = float(r["n_star"])
            if n_star <= 0:
                raise SystemExit("error: every n_star must be positive")
            dmu = dgv / rho_s  # J per atom
            gamma = (3.0 * n_star * rho_s**2 * dmu**3 / (32.0 * math.pi)) ** (1.0 / 3.0)
            point["n_star"] = n_star
            point["r_star_nm"] = (3.0 * n_star / (4.0 * math.pi * rho_s)) ** (1.0 / 3.0) * 1e9
            if "n_low" in r and "n_high" in r:
                half = 0.5 * abs(float(r["n_high"]) - float(r["n_low"]))
                point["gamma_se_J_m2"] = gamma * half / (3.0 * n_star)
            if n_star < 50:
                warnings.append(
                    f"N* = {n_star:.0f} atoms at {temperature:g} K is a nucleus of a few "
                    f"shells; the sharp-interface picture is strained there"
                )
        elif "r_star_nm" in r:
            r_star = float(r["r_star_nm"])
            if r_star <= 0:
                raise SystemExit("error: every r_star_nm must be positive")
            gamma = (r_star * 1e-9) * dgv / 2.0
            point["r_star_nm"] = r_star
            if rho_s is not None:
                point["n_star"] = (4.0 / 3.0) * math.pi * (r_star * 1e-9) ** 3 * rho_s
            if "r_low_nm" in r and "r_high_nm" in r:
                half = 0.5 * abs(float(r["r_high_nm"]) - float(r["r_low_nm"]))
                point["gamma_se_J_m2"] = gamma * half / r_star
        else:
            raise SystemExit('error: every row needs "n_star" or "r_star_nm"')
        point["gamma_J_m2"] = gamma
        points.append(point)
    order_parameter = next(
        (str(r["order_parameter"]) for r in rows if "order_parameter" in r), None
    )
    gammas = np.array([p["gamma_J_m2"] for p in points])
    temperatures = np.array([p["T_K"] for p in points])
    result: dict[str, Any] = {
        "mode": "gamma",
        "points": points,
        "order_parameter": order_parameter,
        "gamma_J_m2": float(gammas.mean()),
        "gamma_std_J_m2": float(gammas.std(ddof=1)) if len(gammas) > 1 else 0.0,
        "warnings": warnings,
    }
    if any("n_star" in p for p in points) and order_parameter is None:
        warnings.append(
            'no "order_parameter" field: N* depends on the cluster criterion (bond-order '
            "threshold, cutoff, mislabeling correction); record it with the brackets"
        )
    if len(points) >= 2 and len(np.unique(temperatures)) >= 2:
        slope, intercept = np.polyfit(temperatures, gammas, 1)
        gamma_tm = float(slope * args.tm + intercept)
        result["gamma_at_tm_J_m2"] = gamma_tm
        result["dgamma_dT_J_m2K"] = float(slope)
        result["gamma_fit_note"] = (
            "gamma(T) = gamma_at_tm + dgamma_dT (T - Tm); pass both to barrier as "
            "--gamma/--gamma-slope"
        )
        if len(points) >= 4:
            _, cov = np.polyfit(temperatures, gammas, 1, cov=True)
            result["gamma_at_tm_se_J_m2"] = float(
                math.sqrt(max(cov[1, 1] + args.tm**2 * cov[0, 0] + 2 * args.tm * cov[0, 1], 0.0))
            )
        if slope < 0:
            warnings.append(
                "gamma falls toward Tm; seeding usually gives a gamma that rises toward Tm "
                "(a Tolman-like trend), so check the brackets and the driving force"
            )
    else:
        warnings.append(
            "one undercooling only: gamma(T) cannot be fitted; run two or three more "
            "undercoolings before carrying gamma to a barrier"
        )
    return result


def _forward(args: argparse.Namespace, with_rate: bool) -> dict[str, Any]:
    warnings: list[str] = []
    f_het = _f_het(args)
    rows: list[dict[str, float]] = []
    reference = args.gamma_ref_t if args.gamma_ref_t is not None else args.tm
    for temperature in args.temps:
        dgv = _driving_force(args, temperature, warnings)
        gamma = args.gamma + args.gamma_slope * (temperature - reference)
        if gamma <= 0:
            raise SystemExit(
                f"error: gamma(T) = {gamma:.4f} J/m^2 at {temperature:g} K; the slope "
                f"extrapolates gamma below zero"
            )
        r_star_m = 2.0 * gamma / dgv
        barrier_j = 16.0 * math.pi * gamma**3 / (3.0 * dgv**2) * f_het
        barrier_ev = barrier_j / _EV_TO_J
        row: dict[str, float] = {
            "T_K": temperature,
            "undercooling_K": args.tm - temperature,
            "gamma_J_m2": gamma,
            "dGv_J_m3": dgv,
            "r_star_nm": r_star_m * 1e9,
            "dG_star_eV": barrier_ev,
            "dG_star_over_kT": barrier_ev / (_KB_EV * temperature),
        }
        if args.omega is not None:
            rho_s = 1.0 / (args.omega * 1e-30)
            n_full = (4.0 / 3.0) * math.pi * r_star_m**3 * rho_s
            row["n_star_atoms"] = n_full * f_het
            if with_rate:
                dmu_j = dgv / rho_s
                zeldovich = math.sqrt(dmu_j / (6.0 * math.pi * _KB_J * temperature * n_full))
                row["zeldovich"] = zeldovich
                if args.attachment is not None:
                    rho_l = 1.0 / (args.omega_liquid * 1e-30) if args.omega_liquid else rho_s
                    exponent = -barrier_j / (_KB_J * temperature)
                    row["rate_per_m3_s"] = rho_l * zeldovich * args.attachment * math.exp(exponent)
        rows.append(row)
    if with_rate and args.omega is None:
        raise SystemExit("error: rate needs --omega (the atomic volume) for N* and Z")
    return {
        "mode": "rate" if with_rate else "barrier",
        "f_het": f_het,
        "gamma_law": {"gamma_J_m2": args.gamma, "slope_J_m2K": args.gamma_slope,
                      "reference_T_K": reference},
        "rows": rows,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="mode", required=True)

    gamma_parser = subparsers.add_parser("gamma", help="gamma(T) from seeded brackets")
    gamma_parser.add_argument("data", type=Path, help='JSON list of {"T", "n_star"|"r_star_nm"}')

    forward: list[argparse.ArgumentParser] = []
    for name, help_text in (("barrier", "barriers from gamma(T)"), ("rate", "barriers and rates")):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--gamma", type=float, required=True,
                         help="interfacial free energy (J/m^2) at --gamma-ref-t")
        sub.add_argument("--gamma-slope", type=float, default=0.0,
                         help="dgamma/dT (J/m^2/K) from the gamma mode's fit (default 0)")
        sub.add_argument("--gamma-ref-t", type=float,
                         help="temperature where --gamma applies (default Tm)")
        sub.add_argument("--temps", type=float, nargs="+", required=True,
                         help="temperatures to evaluate (K)")
        sub.add_argument("--f-het", type=float, default=1.0,
                         help="heterogeneous potency factor in (0, 1] (default 1)")
        sub.add_argument("--theta", type=float,
                         help="contact angle in degrees; sets f_het = (2+cos)(1-cos)^2/4")
        sub.add_argument("--attachment", type=float,
                         help="rate: attachment frequency f+ at the critical nucleus (1/s)")
        sub.add_argument("--omega-liquid", type=float,
                         help="rate: atomic volume of the liquid (A^3; default --omega)")
        forward.append(sub)

    for sub in (gamma_parser, *forward):
        sub.add_argument("--tm", type=float, required=True, help="melting temperature (K)")
        sub.add_argument("--dhf", type=float, required=True,
                         help="volumetric latent heat of fusion (J/m^3)")
        sub.add_argument("--omega", type=float,
                         help="atomic volume of the solid (A^3); needed for N*")
        sub.add_argument("--dmu-table", type=Path,
                         help='JSON {"T", "dgv_J_m3"} rows from a Gibbs-Helmholtz integration')
        sub.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if args.tm <= 0 or args.dhf <= 0:
        raise SystemExit("error: --tm and --dhf must be positive")
    if args.omega is not None and args.omega <= 0:
        raise SystemExit("error: --omega is an atomic volume in A^3; it must be positive")
    args._dmu = _load_dmu(args.dmu_table) if args.dmu_table is not None else None
    if args.mode in ("barrier", "rate"):
        if args.gamma <= 0:
            raise SystemExit("error: --gamma must be positive")
        if not 0.0 < args.f_het <= 1.0:
            raise SystemExit("error: --f-het is the CNT potency factor; it lies in (0, 1]")
        if args.theta is not None and not 0.0 < args.theta <= 180.0:
            raise SystemExit("error: --theta is a contact angle in (0, 180] degrees")
        if args.attachment is not None and args.attachment <= 0:
            raise SystemExit("error: --attachment is a frequency in 1/s; it must be positive")
        result = _forward(args, with_rate=args.mode == "rate")
    else:
        result = _gamma(args)

    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    if result["mode"] == "gamma":
        for point in result["points"]:
            se = f" +/- {point['gamma_se_J_m2']:.4f}" if "gamma_se_J_m2" in point else ""
            size = f"N* = {point['n_star']:.0f}, " if "n_star" in point else ""
            print(
                f"T = {point['T_K']:7.1f} K: {size}r* = {point['r_star_nm']:6.2f} nm "
                f"-> gamma = {point['gamma_J_m2']:.4f}{se} J/m^2"
            )
        if "gamma_at_tm_J_m2" in result:
            se = (
                f" +/- {result['gamma_at_tm_se_J_m2']:.4f}" if "gamma_at_tm_se_J_m2" in result
                else ""
            )
            print(
                f"gamma(T) fit: gamma(Tm) = {result['gamma_at_tm_J_m2']:.4f}{se} J/m^2, "
                f"dgamma/dT = {result['dgamma_dT_J_m2K']:+.3e} J/m^2/K "
                f"({len(result['points'])} bracket(s))"
            )
        else:
            print(f"gamma = {result['gamma_J_m2']:.4f} J/m^2 ({len(result['points'])} bracket)")
    else:
        for row in result["rows"]:
            line = (
                f"T = {row['T_K']:7.1f} K (dT = {row['undercooling_K']:5.1f}): gamma = "
                f"{row['gamma_J_m2']:.4f}, r* = {row['r_star_nm']:6.2f} nm, dG* = "
                f"{row['dG_star_eV']:8.3f} eV = {row['dG_star_over_kT']:8.1f} kT"
            )
            if "n_star_atoms" in row:
                line += f", n* = {row['n_star_atoms']:.0f} atoms"
            if "zeldovich" in row:
                line += f", Z = {row['zeldovich']:.3f}"
            if "rate_per_m3_s" in row:
                line += f", J = {row['rate_per_m3_s']:.3e} /(m^3 s)"
            print(line)
        if result["f_het"] < 1.0:
            print(f"(barriers and atom counts already scaled by f_het = {result['f_het']:.4f})")
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
