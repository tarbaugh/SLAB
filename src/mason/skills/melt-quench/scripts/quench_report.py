"""Tail-averaged density of quenched trajectories, and the densification gap.

    quench_report.py quench-100Kps.traj quench-10Kps.traj
    quench_report.py quench-*.traj --rho-c 4.63 --json

Reads ASE-readable trajectories, averages the mass density over the last
``--tail`` fraction of the frames, and reports one density per file in
g/cm^3. With ``--rho-c`` (the crystal density, g/cm^3) it also reports
delta_v = 1 - rho_a / rho_c, the densification on crystallization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_AMU_TO_G = 1.66053906660e-24
_A3_TO_CM3 = 1e-24


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
            f"error: {path} has {len(frames)} frame(s); a tail average needs at least 4"
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trajectories", type=Path, nargs="+", help="ASE-readable trajectories")
    parser.add_argument(
        "--tail", type=float, default=0.25,
        help="fraction of frames averaged, from the end (default 0.25)",
    )
    parser.add_argument(
        "--rho-c", type=float,
        help="crystal density in g/cm^3, to also report delta_v = 1 - rho_a/rho_c",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if not 0.0 < args.tail <= 1.0:
        raise SystemExit("error: --tail is a fraction of the frames; it must be in (0, 1]")
    if args.rho_c is not None and args.rho_c <= 0:
        raise SystemExit("error: --rho-c is a density in g/cm^3; it must be positive")

    reports: list[dict[str, Any]] = []
    for path in args.trajectories:
        if not path.is_file():
            raise SystemExit(f"error: no such file: {path}")
        densities = _densities(path)
        tail = densities[-max(2, round(len(densities) * args.tail)) :]
        report: dict[str, Any] = {
            "file": str(path),
            "frames": len(densities),
            "tail_frames": len(tail),
            "rho_g_cm3": float(tail.mean()),
            "rho_std_g_cm3": float(tail.std()),
        }
        if args.rho_c is not None:
            report["delta_v"] = 1.0 - float(tail.mean()) / args.rho_c
        reports.append(report)

    if args.json:
        print(json.dumps({"reports": reports, "rho_c_g_cm3": args.rho_c}, indent=1))
        return 0
    for report in reports:
        line = (
            f"{report['file']}: rho = {report['rho_g_cm3']:.4f} "
            f"+/- {report['rho_std_g_cm3']:.4f} g/cm^3 "
            f"({report['tail_frames']} of {report['frames']} frames averaged)"
        )
        if "delta_v" in report:
            line += f", delta_v = {report['delta_v']:.4f}"
        print(line)
    print(
        "(densities are tail averages; check the series reached a plateau "
        "before quoting one, and state the quench rate next to each value)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
