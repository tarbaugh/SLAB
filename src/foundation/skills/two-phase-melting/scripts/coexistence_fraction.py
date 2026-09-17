"""Cell-size gate, calibrated crystalline fraction, and the NPH plateau.

    coexistence_fraction.py nph-r1.log --cells 10
    coexistence_fraction.py nph-r1.log --cells 6 --small-cell --plateau
    coexistence_fraction.py nph-r1.log --cells 10 --plateau --fraction fraction.dat \
        --natoms 4000 --crystal-baseline 0.62 --liquid-baseline 0.05

Reads a LAMMPS log (YAML thermo documents, or the text tables as a
fallback) or the ``-thermo.json`` artifact ``run_lammps`` keeps, and
checks the two things a coexistence run is judged on.

The cell first. A cross section under eight unit cells carries a
finite-size shift on T_m of tens of kelvin, so ``--cells`` under eight is
refused; ``--small-cell`` accepts it and prints the caveat line the
report must carry.

Then the run. ``--plateau`` treats the log as the NPH direct route: the
latent heat drives the cell to T_m, so the temperature settles on a
plateau while both phases survive. It reports the mean temperature over
the primary window (the tail set by ``--window``) and over the secondary
window (the second half of the primary), each with its block standard
error, the drift across the primary window with the error of the fitted
slope, and the verdict. The verdict is a plateau only when the drift
stays inside three block errors, the two windows agree, and the
calibrated crystalline fraction from ``--fraction`` stays off both
baselines. A hot crystal reads well below one under instantaneous CNA, so
``--crystal-baseline`` and ``--liquid-baseline`` come from the pure-phase
legs at the same temperature. Exit code 2 is a refusal.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

#: The cross section, in unit cells, below which T_m is not reportable.
_MIN_CROSS_SECTION_CELLS = 8.0
#: A drift wider than this many block errors is not a plateau.
_DRIFT_LIMIT = 3.0
#: The calibrated fraction must stay this far off both baselines.
_PHASE_MARGIN = 0.05

_THERMO_HEAD = re.compile(r"^\s*Step\s+\S")
_YAML_OPEN = "---"
_YAML_CLOSE = "..."
_YAML_ROW = "  - ["


def _number(token: str) -> float:
    try:
        return float(token)
    except ValueError:
        return float("nan")


def _numeric_row(stripped: str) -> bool:
    try:
        for token in stripped.split():
            float(token)
    except ValueError:
        return False
    return True


def _yaml_table(lines: list[str]) -> dict[str, Any] | None:
    """One YAML thermo document as columns and rows, or None when it has none."""
    columns: list[str] = []
    rows: list[list[float]] = []
    for line in lines:
        if line.startswith("keywords:"):
            inside = line.split(":", 1)[1].strip().strip("[]")
            columns = [name.strip().strip("'\"") for name in inside.split(",") if name.strip()]
        elif line.startswith(_YAML_ROW):
            inside = line.strip().lstrip("-").strip().strip("[]")
            rows.append([_number(token) for token in inside.split(",") if token.strip()])
    if not columns or not rows:
        return None
    return {"columns": columns, "rows": rows}


def thermo_tables(text: str) -> list[dict[str, Any]]:
    """Every thermo table of a log, YAML documents and text tables, in order."""
    tables: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    document: list[str] | None = None
    for line in text.splitlines():
        if document is not None:
            if line.strip() == _YAML_CLOSE:
                table = _yaml_table(document)
                if table is not None:
                    tables.append(table)
                document = None
            elif line.startswith(("keywords:", "data:", _YAML_ROW)):
                document.append(line)
            continue
        stripped = line.strip()
        if current is not None:
            if stripped and _numeric_row(stripped):
                current["rows"].append([_number(token) for token in stripped.split()])
                continue
            if stripped.startswith("WARNING"):
                continue
            tables.append(current)
            current = None
        if stripped == _YAML_OPEN:
            document = []
        elif _THERMO_HEAD.match(line):
            current = {"columns": stripped.split(), "rows": []}
    if document is not None:
        table = _yaml_table(document)
        if table is not None:
            tables.append(table)
    if current is not None:
        tables.append(current)
    return [table for table in tables if table["rows"]]


def load_tables(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json":
        loaded = json.loads(text)
        if not isinstance(loaded, list):
            raise ValueError("the JSON file is not a list of thermo tables")
        return [
            {"columns": list(t["columns"]), "rows": [[float(v) for v in r] for r in t["rows"]]}
            for t in loaded
            if t.get("rows")
        ]
    return thermo_tables(text)


def read_series(path: Path, column: int) -> tuple[np.ndarray, np.ndarray]:
    """(step, value) of an ``ave/time`` file, its comment lines skipped."""
    steps: list[float] = []
    values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if not _numeric_row(stripped) or len(tokens) <= column:
            continue
        steps.append(float(tokens[0]))
        values.append(float(tokens[column]))
    if len(values) < 2:
        raise ValueError(f"fewer than two rows with column {column} in {path}")
    return np.array(steps), np.array(values)


def block_error(values: np.ndarray, blocks: int) -> float:
    """Standard error of the mean from *blocks* equal blocks of *values*."""
    n = len(values)
    blocks = max(1, min(blocks, n))
    size = n // blocks
    if blocks < 2 or size < 2:
        return float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    means = np.array([values[i * size : (i + 1) * size].mean() for i in range(blocks)])
    return float(means.std(ddof=1) / np.sqrt(blocks))


def window_stats(steps: np.ndarray, values: np.ndarray, blocks: int) -> dict[str, Any]:
    """Mean, block error, and the drift across the window with the slope's error."""
    mean = float(values.mean())
    error = block_error(values, blocks)
    span = float(steps[-1] - steps[0])
    if len(values) > 3 and span != 0.0:
        (slope, _), cov = np.polyfit(steps, values, 1, cov=True)
        change = float(slope) * span
        change_error = float(np.sqrt(max(cov[0, 0], 0.0))) * span
    else:
        change, change_error = 0.0, float("nan")
    return {
        "rows": len(values),
        "steps": [float(steps[0]), float(steps[-1])],
        "mean": mean,
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "block_error": error,
        "change": change,
        "change_error": change_error,
        "drift_in_errors": (
            change / error if error and np.isfinite(error) and error > 0 else float("nan")
        ),
    }


def _tail(steps: np.ndarray, values: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    keep = max(2, round(len(values) * fraction))
    keep = min(keep, len(values))
    return steps[len(values) - keep :], values[len(values) - keep :]


def plateau(
    steps: np.ndarray,
    temperatures: np.ndarray,
    *,
    window: float,
    blocks: int,
) -> dict[str, Any]:
    """The primary and secondary window statistics of an NPH temperature series."""
    primary_steps, primary = _tail(steps, temperatures, window)
    secondary_steps, secondary = _tail(primary_steps, primary, 0.5)
    first = window_stats(primary_steps, primary, blocks)
    second = window_stats(secondary_steps, secondary, blocks)
    gap = abs(first["mean"] - second["mean"])
    combined = float(np.sqrt(first["block_error"] ** 2 + second["block_error"] ** 2))
    return {
        "primary": first,
        "secondary": second,
        "window_gap": gap,
        "window_gap_error": combined,
        "windows_agree": bool(np.isfinite(combined) and gap <= 2.0 * combined),
        "settled": bool(
            np.isfinite(first["drift_in_errors"]) and abs(first["drift_in_errors"]) <= _DRIFT_LIMIT
        ),
        "t_plateau": first["mean"],
        "t_plateau_error": float(max(first["block_error"], gap / 2.0)),
    }


def calibrated_fraction(
    values: np.ndarray, *, natoms: int | None, crystal: float, liquid: float
) -> tuple[np.ndarray, list[str]]:
    """The series mapped onto the calibrated baselines, with what to warn about."""
    warnings: list[str] = []
    series = values / natoms if natoms else values
    if natoms is None and float(series.max()) > 1.0:
        raise ValueError(
            "the fraction series holds counts, not fractions; pass --natoms to divide by it"
        )
    if crystal <= liquid:
        raise ValueError("--crystal-baseline must sit above --liquid-baseline")
    if crystal >= 0.999 and liquid <= 0.001:
        warnings.append(
            "the baselines are 1 and 0, so the fraction is uncalibrated; a hot crystal reads "
            "far below 1 under instantaneous CNA, and every gate against 1 fails"
        )
    return (series - liquid) / (crystal - liquid), warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("log", type=Path, help="a LAMMPS log or a -thermo.json artifact")
    parser.add_argument(
        "--cells", type=float, help="unit cells across the cross section (or give the two lengths)"
    )
    parser.add_argument("--cross-section-A", type=float, help="cross-section edge in A")
    parser.add_argument("--lattice-A", type=float, help="a(T) of the crystal in A")
    parser.add_argument(
        "--small-cell", action="store_true",
        help="accept a cross section under eight cells and print the caveat",
    )
    parser.add_argument("--plateau", action="store_true", help="NPH plateau statistics")
    parser.add_argument("--table", type=int, default=-1, help="which table (default: the last)")
    parser.add_argument("--temp-column", default="Temp", help="temperature column (default Temp)")
    parser.add_argument(
        "--window", type=float, default=0.5, help="primary window as a tail fraction (default 0.5)"
    )
    parser.add_argument("--blocks", type=int, default=5, help="blocks for the error (default 5)")
    parser.add_argument("--timestep-fs", type=float, help="timestep in fs, to print ps windows")
    parser.add_argument("--fraction", type=Path, help="an ave/time file of the fraction series")
    parser.add_argument(
        "--fraction-column", type=int, default=1, help="column of the fraction (default 1)"
    )
    parser.add_argument("--natoms", type=int, help="atoms, when the series holds counts")
    parser.add_argument(
        "--crystal-baseline", type=float, default=1.0,
        help="the crystal-only leg's fraction at this temperature",
    )
    parser.add_argument(
        "--liquid-baseline", type=float, default=0.0,
        help="the liquid-only leg's fraction at this temperature",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    cells = args.cells
    if cells is None:
        if args.cross_section_A is None or args.lattice_A is None:
            print(
                "error: pass --cells, or --cross-section-A with --lattice-A", file=sys.stderr
            )
            return 2
        if args.lattice_A <= 0.0:
            print("error: --lattice-A must be a positive length", file=sys.stderr)
            return 2
        cells = args.cross_section_A / args.lattice_A
    if cells <= 0.0:
        print("error: --cells must be a positive count of unit cells", file=sys.stderr)
        return 2
    if not 0.0 < args.window <= 1.0:
        print("error: --window must be in (0, 1]", file=sys.stderr)
        return 2

    caveats: list[str] = []
    if cells < _MIN_CROSS_SECTION_CELLS:
        if not args.small_cell:
            print(
                f"error: the cross section is {cells:.1f} unit cells, under the "
                f"{_MIN_CROSS_SECTION_CELLS:.0f} this method needs; widen the cell, or pass "
                "--small-cell to accept the finite-size shift and carry its caveat",
                file=sys.stderr,
            )
            return 2
        caveats.append(
            f"finite-size caveat: the cross section is {cells:.1f} unit cells, under the "
            f"{_MIN_CROSS_SECTION_CELLS:.0f} a reportable T_m needs. T_m from this cell carries "
            "a finite-size shift of tens of kelvin, the interface cannot roughen across the "
            "cell, and the report must carry this line next to the number."
        )

    try:
        tables = load_tables(args.log)
        if not tables:
            raise ValueError("no thermo table found")
        table = tables[args.table]
    except (OSError, ValueError, IndexError, KeyError, json.JSONDecodeError) as e:
        print(f"error: cannot read a thermo table from {args.log}: {e}", file=sys.stderr)
        return 2

    result: dict[str, Any] = {
        "log": str(args.log),
        "cells": cells,
        "small_cell": bool(cells < _MIN_CROSS_SECTION_CELLS),
        "tables": len(tables),
        "rows": len(table["rows"]),
        "caveats": caveats,
        "warnings": [],
    }

    rows = np.asarray(table["rows"], dtype=float)
    names = list(table["columns"])
    if args.plateau:
        if args.temp_column not in names:
            print(
                f"error: no {args.temp_column} column; the table has {', '.join(names)}",
                file=sys.stderr,
            )
            return 2
        step_index = names.index("Step") if "Step" in names else 0
        temperatures = rows[:, names.index(args.temp_column)]
        if len(temperatures) < 8:
            print(
                f"error: {len(temperatures)} thermo rows is too few for two windows; "
                "print thermo more often, or run longer",
                file=sys.stderr,
            )
            return 2
        result["plateau"] = plateau(
            rows[:, step_index], temperatures, window=args.window, blocks=args.blocks
        )
        if args.timestep_fs:
            for key in ("primary", "secondary"):
                stats = result["plateau"][key]
                stats["ps"] = [s * args.timestep_fs / 1000.0 for s in stats["steps"]]

    if args.fraction is not None:
        try:
            fraction_steps, raw = read_series(args.fraction, args.fraction_column)
            series, warnings = calibrated_fraction(
                raw,
                natoms=args.natoms,
                crystal=args.crystal_baseline,
                liquid=args.liquid_baseline,
            )
        except (OSError, ValueError) as e:
            print(f"error: cannot read the fraction series: {e}", file=sys.stderr)
            return 2
        result["warnings"].extend(warnings)
        tail_steps, tail = _tail(fraction_steps, series, args.window)
        result["fraction"] = {
            "file": str(args.fraction),
            "crystal_baseline": args.crystal_baseline,
            "liquid_baseline": args.liquid_baseline,
            "rows": len(series),
            "calibrated_first": float(series[0]),
            "calibrated_last": float(series[-1]),
            "window_mean": float(tail.mean()),
            "window_min": float(tail.min()),
            "window_max": float(tail.max()),
            "steps": [float(tail_steps[0]), float(tail_steps[-1])],
            "both_phases": bool(
                float(tail.min()) > _PHASE_MARGIN and float(tail.max()) < 1.0 - _PHASE_MARGIN
            ),
        }
        if args.plateau:
            window = result["plateau"]["primary"]["steps"]
            # One sampling interval of slack: an ave/time row is stamped at the
            # end of its own averaging window.
            slack = float(np.diff(fraction_steps).max()) if len(fraction_steps) > 1 else 0.0
            covered = (
                fraction_steps[0] <= window[0] + slack
                and fraction_steps[-1] >= window[1] - slack
            )
            if not covered:
                result["warnings"].append(
                    f"the fraction series covers steps {fraction_steps[0]:.0f} to "
                    f"{fraction_steps[-1]:.0f}, which does not cover the plateau window "
                    f"{window[0]:.0f} to {window[1]:.0f}; a second fix ave/time wrote over "
                    "this file, or the series is from another stage"
                )
        if not result["fraction"]["both_phases"]:
            result["warnings"].append(
                f"the calibrated fraction spans {tail.min():.2f} to {tail.max():.2f} over the "
                "window, so a phase was consumed and the cell is no longer coexistence"
            )

    if args.plateau:
        stats = result["plateau"]
        reasons = []
        if not stats["settled"]:
            reasons.append(
                f"the temperature drifts {stats['primary']['change']:+.1f} over the window, "
                f"{abs(stats['primary']['drift_in_errors']):.1f} block errors"
            )
        if not stats["windows_agree"]:
            reasons.append(
                f"the two windows differ by {stats['window_gap']:.1f} against a combined error "
                f"of {stats['window_gap_error']:.1f}"
            )
        if "fraction" in result and not result["fraction"]["both_phases"]:
            reasons.append("one phase was consumed, so the plateau is not a coexistence plateau")
        if "fraction" not in result:
            reasons.append("no fraction series, so coexistence is unproven")
        result["verdict"] = "two-phase plateau" if not reasons else "not a plateau"
        result["verdict_reasons"] = reasons

    if args.json:
        print(json.dumps(result, indent=1))
        return 0

    print(
        f"{args.log.name}: {result['tables']} thermo table(s), {result['rows']} rows in the one "
        f"read; cross section {cells:.1f} unit cells"
    )
    for caveat in caveats:
        print(caveat)
    if args.plateau:
        stats = result["plateau"]
        for label, key in (("primary", "primary"), ("secondary", "secondary")):
            one = stats[key]
            span = (
                f"{one['ps'][0]:.1f} to {one['ps'][1]:.1f} ps"
                if "ps" in one
                else f"steps {one['steps'][0]:.0f} to {one['steps'][1]:.0f}"
            )
            print(
                f"{label:<10} {one['mean']:>9.2f} +/- {one['block_error']:.2f} "
                f"({one['rows']} rows, {span})"
            )
        primary = stats["primary"]
        print(
            f"drift      {primary['change']:>+9.2f} +/- {primary['change_error']:.2f} across the "
            f"primary window ({primary['drift_in_errors']:+.1f} block errors)"
        )
        if "fraction" in result:
            frac = result["fraction"]
            print(
                f"fraction   {frac['window_mean']:>9.2f} calibrated over the window "
                f"({frac['window_min']:.2f} to {frac['window_max']:.2f}, baselines "
                f"{frac['liquid_baseline']:.2f} and {frac['crystal_baseline']:.2f})"
            )
        print(
            f"verdict: {result['verdict']}; T_m = {stats['t_plateau']:.1f} +/- "
            f"{stats['t_plateau_error']:.1f}"
        )
        for reason in result["verdict_reasons"]:
            print(f"  because {reason}")
    elif "fraction" in result:
        frac = result["fraction"]
        print(
            f"fraction   {frac['window_mean']:>9.2f} calibrated over the window "
            f"({frac['window_min']:.2f} to {frac['window_max']:.2f})"
        )
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
