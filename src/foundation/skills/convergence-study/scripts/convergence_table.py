"""Render a convergence ladder and name the cheapest converged rung.

Input: a JSON file holding an ordered list of records (cheapest rung
first, most accurate last). Every record carries ``"value"`` and
``"energy"`` (eV); a record may also carry ``"fmax"`` (eV/Å, the largest
force) and ``"pressure"`` (kbar), so one ladder can be read for the
quantity the production run will use.

    convergence_table.py conv.json --natoms 2
    convergence_table.py conv.json --quantity force
    convergence_table.py conv.json --quantity pressure --threshold 0.3 --json

Each rung is compared to the final rung. The verdict names the first rung
that stays within the threshold and is followed by at least two more
rungs that also stay within it (the final rung counts as one). Default
thresholds: 1 meV (per atom with ``--natoms``) for energy, 5 meV/Å for
force, 0.5 kbar for pressure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

QUANTITIES: dict[str, tuple[str, float, float]] = {
    # name: (record key, default threshold, scale from the record's unit to the report unit)
    "energy": ("energy", 1.0, 1000.0),  # eV -> meV
    "force": ("fmax", 5.0, 1000.0),  # eV/A -> meV/A
    "pressure": ("pressure", 0.5, 1.0),  # kbar
}
CONFIRMING_RUNGS = 2


def load_ladder(path: Path, key: str) -> list[tuple[str, float]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"error: {path} is not valid JSON: {e}") from e
    if not isinstance(data, list) or not all(
        isinstance(r, dict) and "value" in r and "energy" in r for r in data
    ):
        raise SystemExit(f'error: {path} must hold a list of {{"value", "energy"}} records')
    missing = [str(r["value"]) for r in data if key not in r]
    if missing:
        raise SystemExit(
            f"error: {path} has no {key!r} for rung(s) {', '.join(missing)}; "
            f"record it in the workflow to read the ladder for that quantity"
        )
    return [(str(r["value"]), float(r[key])) for r in data]


def converged_rung(diffs: list[float], threshold: float) -> int | None:
    """Index of the first rung within *threshold* that at least
    ``CONFIRMING_RUNGS`` later rungs confirm; None when no rung qualifies.

    The final rung is the reference and confirms trivially, so a ladder
    of n rungs can name index n - 1 - CONFIRMING_RUNGS at the latest.
    """
    for i in range(len(diffs) - CONFIRMING_RUNGS):
        if all(diff <= threshold for diff in diffs[i:-1]):
            return i
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, help="conv.json, cheapest rung first")
    parser.add_argument(
        "--quantity",
        choices=sorted(QUANTITIES),
        default="energy",
        help="which recorded quantity to read (default energy)",
    )
    parser.add_argument(
        "--threshold",
        "--threshold-mev",
        dest="threshold",
        type=float,
        default=None,
        help="convergence threshold in the report unit: meV (per atom with --natoms), "
        "meV/A, or kbar (defaults 1.0, 5.0, 0.5)",
    )
    parser.add_argument("--natoms", type=int, help="atoms per cell, to report meV/atom")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if not args.data.is_file():
        raise SystemExit(f"error: no such file: {args.data}")
    key, default_threshold, scale = QUANTITIES[args.quantity]
    threshold = default_threshold if args.threshold is None else args.threshold
    if threshold <= 0:
        raise SystemExit("error: --threshold must be positive")
    ladder = load_ladder(args.data, key)
    if len(ladder) < 2:
        raise SystemExit("error: a ladder needs at least 2 rungs (4 or more is typical)")
    if args.natoms is not None and args.natoms < 1:
        raise SystemExit("error: --natoms must be a positive atom count")

    if args.quantity == "energy":
        unit = "meV/atom" if args.natoms else "meV"
        scale = scale / (args.natoms or 1)
    elif args.quantity == "force":
        unit = "meV/A"
    else:
        unit = "kbar"
    reference = ladder[-1][1]
    diffs = [abs(quantity - reference) * scale for _, quantity in ladder]
    index = converged_rung(diffs, threshold)
    converged_at = ladder[index][0] if index is not None else None

    if args.json:
        report: dict[str, Any] = {
            "quantity": args.quantity,
            "rungs": [
                {"value": value, key: quantity, f"diff_{unit}": diff}
                for (value, quantity), diff in zip(ladder, diffs, strict=True)
            ],
            "threshold": threshold,
            "unit": unit,
            "converged_at": converged_at,
            "confirming_rungs": CONFIRMING_RUNGS,
        }
        print(json.dumps(report, indent=1))
        return 0
    width = max(len(value) for value, _ in ladder)
    column = {"energy": "energy (eV)", "force": "fmax (eV/A)", "pressure": "pressure (kbar)"}
    print(f"{'value':>{width}}  {column[args.quantity]:>16}  {f'vs final ({unit})':>18}")
    for i, ((value, quantity), diff) in enumerate(zip(ladder, diffs, strict=True)):
        marker = "  <- reference" if i == len(ladder) - 1 else ""
        print(f"{value:>{width}}  {quantity:>16.6f}  {diff:>18.3f}{marker}")
    if index is not None:
        print(
            f"converged at {ladder[index][0]}: stays within {threshold} {unit} of the "
            f"final rung, confirmed by {len(ladder) - 1 - index} later rung(s)"
        )
    else:
        print(
            f"not converged: no rung stays within {threshold} {unit} of the final "
            f"rung with {CONFIRMING_RUNGS} later rungs confirming it; extend the ladder"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
