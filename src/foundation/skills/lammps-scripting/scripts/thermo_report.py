"""Equilibration statistics for the thermo tables of a LAMMPS run.

    thermo_report.py ar-nvt-thermo.json
    thermo_report.py log.lammps --table 0 --tail 0.5 --columns Temp Press
    thermo_report.py ar-nvt-thermo.json --json

Reads either the ``-thermo.json`` artifact ``run_lammps`` keeps or a
LAMMPS log, takes one table (the last by default), and reports each
column over the tail of the table: the mean, the standard deviation, the
block standard error (the tail cut into equal blocks, so correlated rows
do not pretend to be independent samples), and the drift, the change of a
straight-line fit across the tail in units of that error. A drift above
three errors is a warning: the quantity is still moving, so hold longer or
discard more. Exit code 2 means no table could be read.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

_THERMO_HEAD = re.compile(r"^\s*Step\s+\S")
_LOOP = re.compile(r"^Loop time of (\S+) on (\d+) procs for (\d+) steps with (\d+) atoms")
_DRIFT_LIMIT = 3.0


def _numeric(tokens: list[str]) -> bool:
    try:
        for token in tokens:
            float(token)
    except ValueError:
        return False
    return True


def tables_from_log(text: str) -> list[dict[str, Any]]:
    """The thermo tables of a log: columns and rows, with the loop line when present."""
    tables: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if current is not None:
            tokens = stripped.split()
            if tokens and _numeric(tokens):
                current["rows"].append([float(t) for t in tokens])
                continue
            if stripped.startswith("WARNING"):
                continue
            tables.append(current)
            current = None
        if _THERMO_HEAD.match(line):
            current = {"columns": stripped.split(), "rows": [], "loop": None}
        elif (m := _LOOP.match(line)) and tables and tables[-1]["loop"] is None:
            tables[-1]["loop"] = {
                "seconds": float(m.group(1)),
                "procs": int(m.group(2)),
                "steps": int(m.group(3)),
                "atoms": int(m.group(4)),
            }
    if current is not None:
        tables.append(current)
    return tables


def load_tables(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json":
        loaded = json.loads(text)
        if not isinstance(loaded, list):
            raise ValueError("the JSON file is not a list of thermo tables")
        return [
            {
                "columns": list(table["columns"]),
                "rows": [[float(v) for v in row] for row in table["rows"]],
                "loop": table.get("loop"),
            }
            for table in loaded
        ]
    return tables_from_log(text)


def block_error(values: np.ndarray, blocks: int) -> float:
    """Standard error of the mean from *blocks* equal blocks of *values*."""
    n = len(values)
    blocks = max(1, min(blocks, n))
    size = n // blocks
    if blocks < 2 or size < 1:
        return float("nan")
    means = np.array([values[i * size : (i + 1) * size].mean() for i in range(blocks)])
    return float(means.std(ddof=1) / np.sqrt(blocks))


def column_stats(steps: np.ndarray, values: np.ndarray, blocks: int) -> dict[str, Any]:
    """Mean, std, block error, and drift (in errors) of one column over its tail."""
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    error = block_error(values, blocks)
    if len(values) > 2 and steps[-1] != steps[0]:
        slope, _ = np.polyfit(steps, values, 1)
        change = float(slope * (steps[-1] - steps[0]))
    else:
        change = 0.0
    drift_in_errors = change / error if error and np.isfinite(error) and error > 0 else float("nan")
    return {
        "mean": mean,
        "std": std,
        "block_error": error,
        "change_over_tail": change,
        "drift_in_errors": drift_in_errors,
        "drifting": bool(np.isfinite(drift_in_errors) and abs(drift_in_errors) > _DRIFT_LIMIT),
    }


def report(
    table: dict[str, Any], *, tail: float, blocks: int, columns: list[str] | None
) -> dict[str, Any]:
    rows = np.asarray(table["rows"], dtype=float)
    if rows.size == 0:
        raise ValueError("the table has no rows")
    names = table["columns"]
    n_tail = max(2, round(len(rows) * tail)) if len(rows) > 1 else 1
    n_tail = min(n_tail, len(rows))
    block = rows[len(rows) - n_tail :]
    step_index = names.index("Step") if "Step" in names else 0
    steps = block[:, step_index]
    wanted = columns or [name for name in names if name != "Step"]
    unknown = [name for name in wanted if name not in names]
    if unknown:
        raise ValueError(f"no column {', '.join(unknown)}; the table has {', '.join(names)}")
    out: dict[str, Any] = {
        "rows": len(rows),
        "tail_rows": n_tail,
        "tail_steps": [float(steps[0]), float(steps[-1])],
        "blocks": max(1, min(blocks, n_tail)),
        "loop": table.get("loop"),
        "columns": {},
    }
    for name in wanted:
        out["columns"][name] = column_stats(steps, block[:, names.index(name)], blocks)
    out["drifting"] = sorted(name for name, stats in out["columns"].items() if stats["drifting"])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", type=Path, help="a -thermo.json artifact or a LAMMPS log")
    parser.add_argument("--table", type=int, default=-1, help="which table (default: the last)")
    parser.add_argument(
        "--tail", type=float, default=0.5, help="fraction of rows to average (default 0.5)"
    )
    parser.add_argument("--blocks", type=int, default=5, help="blocks for the error (default 5)")
    parser.add_argument("--columns", nargs="+", help="columns to report (default: all but Step)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    if not 0.0 < args.tail <= 1.0:
        print("error: --tail must be in (0, 1]", file=sys.stderr)
        return 2
    try:
        tables = load_tables(args.file)
        if not tables:
            raise ValueError("no thermo table found")
        table = tables[args.table]
        result = report(table, tail=args.tail, blocks=args.blocks, columns=args.columns)
    except (OSError, ValueError, IndexError, KeyError) as e:
        print(f"error: cannot read a thermo table from {args.file}: {e}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    loop = result["loop"]
    where = (
        f"{loop['steps']} steps, {loop['atoms']} atoms, {loop['seconds']:.2f} s"
        if loop
        else "no loop line (the run did not finish this table)"
    )
    print(
        f"table: {result['rows']} rows, tail {result['tail_rows']} rows "
        f"(steps {result['tail_steps'][0]:.0f} to {result['tail_steps'][1]:.0f}), "
        f"{result['blocks']} blocks; {where}"
    )
    print(f"{'column':<12} {'mean':>14} {'std':>12} {'block err':>12} {'drift/err':>10}")
    for name, stats in result["columns"].items():
        drift = stats["drift_in_errors"]
        flag = "  DRIFTING" if stats["drifting"] else ""
        print(
            f"{name:<12} {stats['mean']:>14.6g} {stats['std']:>12.4g} "
            f"{stats['block_error']:>12.4g} {drift:>10.2f}{flag}"
        )
    if result["drifting"]:
        print(
            f"warning: {', '.join(result['drifting'])} still moving over the tail; "
            "hold longer or discard more before averaging"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
