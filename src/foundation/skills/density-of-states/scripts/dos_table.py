"""Read a density_of_states result file: the verdict, a table, a plot.

Input: the ``-dos.json`` artifact that ``density_of_states`` keeps (read
it with ``read_artifact`` and save it, or pass the path of a kept copy).
It holds the energy grid, the total density of states, the integrated
density of states, the Fermi level, the summary, and, for a projected
run, one curve per element and angular momentum.

    dos_table.py si-dos.json
    dos_table.py si-dos.json --dat si-dos.dat
    dos_table.py si-dos.json --png si-dos.png
    dos_table.py si-dos.json --png si-dos.png --window -4 6

With no option the script prints the summary as JSON. ``--dat`` writes a
whitespace table: the energy in eV relative to the valence band maximum
(or to the Fermi level for a metal), the total density of states, and one
column per projected group. ``--png`` draws the same curves when
matplotlib is installed and exits 2 when it is not.

``--png`` draws an energy window, not the whole grid. The default window
runs from 8 eV below the reference to 8 eV above the conduction band
minimum (above the Fermi level for a metal), cut to the grid the run
has. ``--window LO HI`` sets the window in eV relative to the reference,
and ``--whole-grid`` draws the whole grid. The vertical axis fits the
curves inside the window. The report names the window. ``--dat`` always
holds the whole grid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"error: cannot read {path} as JSON: {e}") from e
    if not isinstance(data, dict) or "summary" not in data:
        raise SystemExit(f"error: {path} is not a density_of_states result file (no 'summary')")
    return data


def reference(data: dict[str, Any]) -> tuple[float, str]:
    """The zero of the energy axis and its name."""
    summary = data["summary"]
    if summary.get("vbm") is not None and not summary.get("is_metal"):
        return float(summary["vbm"]), "the valence band maximum"
    return float(data["fermi"]), "the Fermi level"


def curves(data: dict[str, Any]) -> tuple[list[float], list[float], dict[str, list[float]]]:
    """The grid, the total density of states, and the group curves."""
    grid = data.get("energies")
    if grid is None:
        where = data.get("dos_in", "the -dos.json artifact")
        raise SystemExit(
            f"error: this file carries no energy grid; it is in {where}, so read that "
            f"artifact and pass it instead"
        )
    groups = data.get("projected_dos") or {}
    return (
        [float(e) for e in grid],
        [float(v) for v in data["dos"]],
        {name: [float(v) for v in values] for name, values in sorted(groups.items())},
    )


def write_dat(data: dict[str, Any], out: Path) -> None:
    zero, name = reference(data)
    grid, total, groups = curves(data)
    names = list(groups)
    lines = [
        f"# density of states, {len(grid)} rows, step {data['delta_e']} eV, "
        f"Gaussian broadening {data['degauss_ry']} Ry",
        f"# energies in eV relative to {name} ({zero:.4f} eV)",
        f"# dos in states/eV/cell; Fermi level at {data['fermi'] - zero:.4f} eV",
        "# columns: E, total dos" + "".join(f", {group}" for group in names),
    ]
    for row in range(len(grid)):
        values = [grid[row] - zero, total[row], *(groups[group][row] for group in names)]
        lines.append(" ".join(f"{value:.6f}" for value in values))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


#: The default plot window reaches this far below the reference and this
#: far above the conduction band minimum (the Fermi level for a metal).
_WINDOW_MARGIN_EV = 8.0


def plot_window(
    data: dict[str, Any], window: tuple[float, float] | None, whole_grid: bool
) -> tuple[float, float]:
    """The energy window of the plot, in eV relative to the reference."""
    zero, _ = reference(data)
    grid, _, _ = curves(data)
    lowest, highest = grid[0] - zero, grid[-1] - zero
    if whole_grid:
        return lowest, highest
    if window is not None:
        low, high = window
        if low >= high:
            print(f"error: --window needs LO below HI, got {low:g} {high:g}", file=sys.stderr)
            raise SystemExit(2)
        if high <= lowest or low >= highest:
            print(
                f"error: --window {low:g} {high:g} lies off the grid, which spans "
                f"{lowest:.2f} to {highest:.2f} eV relative to the reference",
                file=sys.stderr,
            )
            raise SystemExit(2)
        return low, high
    gap = data["summary"].get("gap") or 0.0
    return max(lowest, -_WINDOW_MARGIN_EV), min(highest, gap + _WINDOW_MARGIN_EV)


def write_png(data: dict[str, Any], out: Path, window: tuple[float, float] | None = None) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "error: --png needs the matplotlib package, which is not installed; "
            "write --dat and plot that table instead",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    zero, name = reference(data)
    grid, total, groups = curves(data)
    shifted = [e - zero for e in grid]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(shifted, total, color="black", linewidth=1, label="total")
    for group, values in groups.items():
        ax.plot(shifted, values, linewidth=1, label=group)
    ax.axvline(data["fermi"] - zero, color="grey", linestyle="--", linewidth=0.5)
    ax.set_xlabel(f"E - E({'VBM' if name.startswith('the valence') else 'Fermi'}) (eV)")
    ax.set_ylabel("DOS (states/eV/cell)")
    top = None
    if window is not None:
        ax.set_xlim(*window)
        # Fit the vertical axis to the window, or a semicore peak outside
        # it flattens the curves inside it.
        inside = [v for e, v in zip(shifted, total, strict=True) if window[0] <= e <= window[1]]
        top = 1.05 * max(inside) if inside and max(inside) > 0 else None
    ax.set_ylim(bottom=0.0, top=top)
    if groups:
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dos", type=Path, help="a -dos.json file from density_of_states")
    parser.add_argument("--dat", type=Path, help="write the dos table here")
    parser.add_argument("--png", type=Path, help="draw the dos curves here (needs matplotlib)")
    span = parser.add_mutually_exclusive_group()
    span.add_argument(
        "--window",
        nargs=2,
        type=float,
        metavar=("LO", "HI"),
        help="the energy window of --png in eV relative to the reference "
        "(default: -8 to 8 above the conduction band minimum)",
    )
    span.add_argument("--whole-grid", action="store_true", help="draw the whole grid in --png")
    args = parser.parse_args(argv)
    if (args.window is not None or args.whole_grid) and args.png is None:
        parser.error("--window and --whole-grid set the window of --png; give --png")

    data = load(args.dos)
    report = {
        "fermi": data["fermi"],
        "delta_e": data["delta_e"],
        "degauss_ry": data["degauss_ry"],
        "n_bands": data["n_bands"],
        "projection_groups": data.get("projection_groups", []),
        **dict(data["summary"]),
    }
    if args.dat is not None:
        write_dat(data, args.dat)
        report["dat"] = str(args.dat)
    if args.png is not None:
        window = plot_window(data, tuple(args.window) if args.window else None, args.whole_grid)
        write_png(data, args.png, window)
        report["png"] = str(args.png)
        report["png_window_ev"] = [round(window[0], 3), round(window[1], 3)]
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
