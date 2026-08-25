"""Render a convergence ladder and name the cheapest converged rung.

Input: a JSON file holding an ordered list of ``{"value", "energy"}``
records (cheapest rung first, most accurate last), energies in eV.

    convergence_table.py conv.json --natoms 2
    convergence_table.py conv.json --threshold-mev 0.5 --json

Each rung is compared to the final rung. The verdict names the first rung
whose difference - and every later rung's - stays within the threshold
(default 1 meV, per atom when ``--natoms`` is given).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_ladder(path: Path) -> list[tuple[str, float]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"error: {path} is not valid JSON: {e}") from e
    if not isinstance(data, list) or not all(
        isinstance(r, dict) and "value" in r and "energy" in r for r in data
    ):
        raise SystemExit(f'error: {path} must hold a list of {{"value", "energy"}} records')
    return [(str(r["value"]), float(r["energy"])) for r in data]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, help="conv.json, cheapest rung first")
    parser.add_argument(
        "--threshold-mev", type=float, default=1.0,
        help="convergence threshold in meV (per atom with --natoms; default 1.0)",
    )
    parser.add_argument("--natoms", type=int, help="atoms per cell, to report meV/atom")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if not args.data.is_file():
        raise SystemExit(f"error: no such file: {args.data}")
    ladder = load_ladder(args.data)
    if len(ladder) < 2:
        raise SystemExit("error: a ladder needs at least 2 rungs (4 or more is typical)")
    if args.natoms is not None and args.natoms < 1:
        raise SystemExit("error: --natoms must be a positive atom count")

    unit = "meV/atom" if args.natoms else "meV"
    scale = 1000.0 / (args.natoms or 1)
    reference = ladder[-1][1]
    diffs = [abs(energy - reference) * scale for _, energy in ladder]
    converged_at: str | None = None
    for i in range(len(ladder) - 1):
        if all(diff <= args.threshold_mev for diff in diffs[i:-1]):
            converged_at = ladder[i][0]
            break
    settled = converged_at is not None

    if args.json:
        print(
            json.dumps(
                {
                    "rungs": [
                        {"value": value, "energy_eV": energy, f"diff_{unit}": diff}
                        for (value, energy), diff in zip(ladder, diffs, strict=True)
                    ],
                    "threshold": args.threshold_mev,
                    "unit": unit,
                    "converged_at": converged_at,
                },
                indent=1,
            )
        )
        return 0
    width = max(len(value) for value, _ in ladder)
    print(f"{'value':>{width}}  {'energy (eV)':>16}  {f'vs final ({unit})':>18}")
    for (value, energy), diff in zip(ladder, diffs, strict=True):
        marker = "  <- reference" if energy == reference and value == ladder[-1][0] else ""
        print(f"{value:>{width}}  {energy:>16.6f}  {diff:>18.3f}{marker}")
    if settled:
        print(
            f"converged at {converged_at}: stays within {args.threshold_mev} {unit} "
            f"of the final rung"
        )
    else:
        print(
            f"not converged: even the second-to-last rung differs by more than "
            f"{args.threshold_mev} {unit}; extend the ladder"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
