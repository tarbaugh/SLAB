"""Hold-averaged density of quenched trajectories, the densification gap,
the replica spread per rate, and the density-versus-log-rate law.

    quench_report.py quench-100Kps-r1.traj quench-10Kps-r1.traj --hold-frames 50
    quench_report.py quench-*.traj --hold-frames 50 --rho-c 4.63 --json

Reads ASE-readable trajectories and averages the mass density over the
isothermal hold at the end of each (``--hold-frames``, the number of
frames the workflow recorded at the final temperature; ``--tail`` gives a
fraction instead). The average carries a block standard error, and a
drift between the two halves of the hold larger than three standard
errors is a warning: the glass has not settled. Files named
``quench-<rate>Kps[-r<k>].traj`` are grouped by rate, replicas are
averaged with their spread, and with two or more rates the law
rho = rho_0 + a log10(rate) is fitted. With ``--rho-c`` (the crystal
density, g/cm^3, at the same temperature and pressure) it also reports
delta_v = 1 - rho_a / rho_c, the densification on crystallization.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

_AMU_TO_G = 1.66053906660e-24
_A3_TO_CM3 = 1e-24
_NAME = re.compile(r"quench-(?P<rate>[0-9.eE+-]+)Kps(?:-r(?P<replica>\d+))?")
BLOCKS = 5


def _densities(path: Path) -> np.ndarray:
    """The density of every frame in g/cm^3."""
    from ase.io import read

    try:
        result = read(path, index=":")
    except Exception as e:
        raise SystemExit(f"error: cannot read {path} with ASE: {e}") from e
    frames = list(result) if isinstance(result, list) else [result]
    if len(frames) < 4:
        raise SystemExit(
            f"error: {path} has {len(frames)} frame(s); a hold average needs at least 4"
        )
    densities = []
    for i, atoms in enumerate(frames):
        try:
            volume = float(atoms.get_volume())
        except ValueError:  # no cell at all
            volume = 0.0
        if not np.isfinite(volume) or volume <= 0:
            raise SystemExit(f"error: frame {i} of {path} has no positive cell volume")
        mass_g = float(atoms.get_masses().sum()) * _AMU_TO_G
        densities.append(mass_g / (volume * _A3_TO_CM3))
    return np.array(densities)


def _rate_of(path: Path) -> tuple[float | None, int | None]:
    match = _NAME.search(path.name)
    if match is None:
        return None, None
    try:
        rate = float(match.group("rate"))
    except ValueError:
        return None, None
    replica = int(match.group("replica")) if match.group("replica") else None
    return rate, replica


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trajectories", type=Path, nargs="+", help="ASE-readable trajectories")
    parser.add_argument(
        "--hold-frames", type=int,
        help="frames at the end recorded during the isothermal hold (the workflow's "
        "HOLD_STEPS / SEGMENT_STEPS)",
    )
    parser.add_argument(
        "--tail", type=float, default=0.25,
        help="fraction of frames averaged from the end when --hold-frames is not "
        "given (default 0.25; a ramp's tail is not a hold, and the report says so)",
    )
    parser.add_argument(
        "--rho-c", type=float,
        help="crystal density in g/cm^3 at the same T and P, to also report "
        "delta_v = 1 - rho_a/rho_c",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if not 0.0 < args.tail <= 1.0:
        raise SystemExit("error: --tail is a fraction of the frames; it must be in (0, 1]")
    if args.hold_frames is not None and args.hold_frames < 4:
        raise SystemExit("error: --hold-frames must be at least 4")
    if args.rho_c is not None and args.rho_c <= 0:
        raise SystemExit("error: --rho-c is a density in g/cm^3; it must be positive")

    reports: list[dict[str, Any]] = []
    for path in args.trajectories:
        if not path.is_file():
            raise SystemExit(f"error: no such file: {path}")
        densities = _densities(path)
        if args.hold_frames is not None:
            if args.hold_frames > len(densities):
                raise SystemExit(
                    f"error: {path} has {len(densities)} frames, fewer than the "
                    f"{args.hold_frames} hold frames asked for"
                )
            hold = densities[-args.hold_frames :]
        else:
            hold = densities[-max(2, round(len(densities) * args.tail)) :]
        n_blocks = min(BLOCKS, len(hold) // 2)
        blocks = [block.mean() for block in np.array_split(hold, n_blocks)] if n_blocks >= 2 else []
        se = float(np.std(blocks, ddof=1) / np.sqrt(len(blocks))) if len(blocks) >= 2 else None
        half = len(hold) // 2
        drift = float(hold[half:].mean() - hold[:half].mean())
        rate, replica = _rate_of(path)
        report: dict[str, Any] = {
            "file": str(path),
            "rate_K_per_ps": rate,
            "replica": replica,
            "frames": len(densities),
            "hold_frames": len(hold),
            "rho_g_cm3": float(hold.mean()),
            "rho_se_g_cm3": se,
            "rho_std_g_cm3": float(hold.std()),
            "drift_g_cm3": drift,
            "warnings": [],
        }
        if se is not None and abs(drift) > 3.0 * se and abs(drift) > 1e-4 * hold.mean():
            report["warnings"].append(
                f"the density drifts by {drift:+.4f} g/cm^3 across the hold, more than "
                f"three standard errors: the glass has not settled; hold longer"
            )
        if args.hold_frames is None:
            report["warnings"].append(
                "averaged over the last fraction of the trajectory, not a declared hold; "
                "a ramp's tail is a range of temperatures, not one density"
            )
        if args.rho_c is not None:
            report["delta_v"] = 1.0 - float(hold.mean()) / args.rho_c
        reports.append(report)

    by_rate: dict[float, list[dict[str, Any]]] = {}
    for report in reports:
        if report["rate_K_per_ps"] is not None:
            by_rate.setdefault(report["rate_K_per_ps"], []).append(report)
    rates: list[dict[str, Any]] = []
    for rate in sorted(by_rate, reverse=True):
        values = np.array([r["rho_g_cm3"] for r in by_rate[rate]])
        entry: dict[str, Any] = {
            "rate_K_per_ps": rate,
            "replicas": len(values),
            "rho_g_cm3": float(values.mean()),
            "rho_spread_g_cm3": float(values.std(ddof=1)) if len(values) > 1 else None,
        }
        if args.rho_c is not None:
            entry["delta_v"] = 1.0 - float(values.mean()) / args.rho_c
        rates.append(entry)
    law: dict[str, Any] | None = None
    if len(rates) >= 2:
        x = np.log10([r["rate_K_per_ps"] for r in rates])
        y = np.array([r["rho_g_cm3"] for r in rates])
        slope, intercept = np.polyfit(x, y, 1)
        law = {
            "rho_0_g_cm3": float(intercept),
            "slope_per_decade_g_cm3": float(slope),
            "note": "rho = rho_0 + slope * log10(rate in K/ps); the sign is the system's, "
            "not a rule (silica loosens on slower cooling, metals densify)",
        }

    if args.json:
        payload = {
            "reports": reports,
            "rates": rates,
            "log_rate_law": law,
            "rho_c_g_cm3": args.rho_c,
        }
        print(json.dumps(payload, indent=1))
        return 0
    for report in reports:
        se_text = f" +/- {report['rho_se_g_cm3']:.4f}" if report["rho_se_g_cm3"] else ""
        line = (
            f"{report['file']}: rho = {report['rho_g_cm3']:.4f}{se_text} g/cm^3 "
            f"({report['hold_frames']} of {report['frames']} frames averaged, "
            f"drift {report['drift_g_cm3']:+.4f})"
        )
        if "delta_v" in report:
            line += f", delta_v = {report['delta_v']:.4f}"
        print(line)
        for warning in report["warnings"]:
            print(f"warning: {warning}")
    for entry in rates:
        spread = f" +/- {entry['rho_spread_g_cm3']:.4f}" if entry["rho_spread_g_cm3"] else ""
        print(
            f"rate {entry['rate_K_per_ps']:g} K/ps: rho = {entry['rho_g_cm3']:.4f}{spread} "
            f"g/cm^3 over {entry['replicas']} replica(s)"
        )
    if law is not None:
        print(
            f"rho = {law['rho_0_g_cm3']:.4f} + {law['slope_per_decade_g_cm3']:+.4f} * "
            f"log10(rate / (K/ps)) g/cm^3; extrapolate to an experimental rate with the "
            f"rate stated, never as an experimental glass"
        )
    print("(state the quench rate, the hold length, and the replica count next to each value)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
