"""Read a band_structure result file: the gap verdict, a table, a plot.

Input: the ``-bands.json`` artifact that ``band_structure`` keeps (read it
with ``read_artifact`` and save it, or pass the path of a kept copy). It
holds the path, the k-point distances, the eigenvalues in eV, the SCF
Fermi level, and the summary.

    bands_table.py si-bands.json
    bands_table.py si-bands.json --dat si-bands.dat
    bands_table.py si-bands.json --png si-bands.png
    bands_table.py si-bands.json --png si-bands.png --window -4 6
    bands_table.py si-bands.json --projection Si-p --dat si-fat.dat

With no option the script prints the summary as JSON. ``--projection``
names a group of a projected run, such as ``Si-p``. In ``--dat`` it adds
one weight column per band after the energy columns, so a row holds x,
then n energies, then the n weights in the same band order. In ``--png``
it sizes each marker by the weight, which is the fat-band diagram.

``--dat`` writes a whitespace table: the distance along the path in 1/Å,
then one column per band, in eV relative to the valence band maximum (or
to the Fermi level for a metal). Comment lines at the top give the
reference energy and the x position of each special point. ``--png``
draws the same diagram when matplotlib is installed and exits 2 when it
is not.

``--png`` draws an energy window, not every band. The default window
runs from 8 eV below the reference to 8 eV above the conduction band
minimum (above the Fermi level for a metal), cut to the energies the run
has. ``--window LO HI`` sets the window in eV relative to the reference,
and ``--all-bands`` draws every band. The report names the window and
counts the bands that lie outside it. ``--dat`` always holds every band.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

#: The labels ASE uses, as a band diagram prints them.
_GREEK = {"G": "Γ"}

#: The default plot window reaches this far below the reference and this
#: far above the conduction band minimum (the Fermi level for a metal).
_WINDOW_MARGIN_EV = 8.0
#: Blank space between the outermost drawn energy and the frame.
_WINDOW_PAD_EV = 0.5


def load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"error: cannot read {path} as JSON: {e}") from e
    if not isinstance(data, dict) or "summary" not in data:
        raise SystemExit(f"error: {path} is not a band_structure result file (no 'summary')")
    return data


def reference(data: dict[str, Any]) -> tuple[float, str]:
    """The zero of the energy axis and its name."""
    summary = data["summary"]
    if summary.get("vbm") is not None and not summary.get("is_metal"):
        return float(summary["vbm"]), "the valence band maximum"
    return float(data["fermi"]), "the Fermi level"


def energies(data: dict[str, Any]) -> list[list[float]]:
    rows = data.get("energies")
    if rows is None:
        where = data.get("energies_in", "the -bands.json artifact")
        raise SystemExit(
            f"error: this file carries no energies; they are in {where}, so read that "
            f"artifact and pass it instead"
        )
    return [[float(e) for e in row] for row in rows]


def projection(data: dict[str, Any], group: str | None) -> list[list[float]] | None:
    """The weight of *group* per k-point and band, or None without a group."""
    if group is None:
        return None
    groups = data.get("projections")
    if not groups:
        names = ", ".join(data.get("projection_groups", [])) or "none"
        raise SystemExit(
            f"error: this file carries no projections (groups: {names}); run "
            f"band_structure with projected=True, and read the -bands.json it keeps"
        )
    if group not in groups:
        raise SystemExit(
            f"error: {group!r} is not a group of this run; it has "
            f"{', '.join(sorted(groups))}"
        )
    return [[float(w) for w in row] for row in groups[group]]


def write_dat(data: dict[str, Any], out: Path, group: str | None = None) -> None:
    zero, name = reference(data)
    rows = energies(data)
    weights = projection(data, group)
    ticks = " ".join(f"{t['label']} {t['x']:.6f}" for t in data["ticks"])
    lines = [
        f"# band structure along {data['path']} ({data['lattice']}), "
        f"{len(rows)} k-points, {len(rows[0])} bands",
        f"# energies in eV relative to {name} ({zero:.4f} eV)",
        f"# special points (label x): {ticks}",
        "# columns: x (1/A), then band 1 to band " + str(len(rows[0])),
    ]
    if weights is not None:
        lines.append(
            f"# then the weight of {group} in band 1 to band {len(rows[0])}, "
            f"one column each"
        )
    for index, (x, row) in enumerate(zip(data["distances"], rows, strict=True)):
        columns = [f"{x:.6f}", *(f"{e - zero:.4f}" for e in row)]
        if weights is not None:
            columns += [f"{w:.4f}" for w in weights[index]]
        lines.append(" ".join(columns))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_window(
    data: dict[str, Any], window: tuple[float, float] | None, all_bands: bool
) -> tuple[float, float]:
    """The energy window of the diagram, in eV relative to the reference."""
    zero, _ = reference(data)
    rows = energies(data)
    lowest = min(min(row) for row in rows) - zero - _WINDOW_PAD_EV
    highest = max(max(row) for row in rows) - zero + _WINDOW_PAD_EV
    if all_bands:
        return lowest, highest
    if window is not None:
        low, high = window
        if low >= high:
            print(f"error: --window needs LO below HI, got {low:g} {high:g}", file=sys.stderr)
            raise SystemExit(2)
        if high < lowest or low > highest:
            print(
                f"error: --window {low:g} {high:g} holds no band; the bands span "
                f"{lowest:.2f} to {highest:.2f} eV relative to the reference",
                file=sys.stderr,
            )
            raise SystemExit(2)
        return low, high
    gap = data["summary"].get("gap") or 0.0
    return max(lowest, -_WINDOW_MARGIN_EV), min(highest, gap + _WINDOW_MARGIN_EV)


def bands_outside(data: dict[str, Any], window: tuple[float, float]) -> dict[str, int]:
    """How many bands lie wholly below and wholly above the window."""
    zero, _ = reference(data)
    rows = energies(data)
    below = above = 0
    for band in range(len(rows[0])):
        values = [row[band] - zero for row in rows]
        below += max(values) < window[0]
        above += min(values) > window[1]
    return {"below": below, "above": above}


def write_png(
    data: dict[str, Any],
    out: Path,
    group: str | None = None,
    window: tuple[float, float] | None = None,
) -> None:
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
    rows = energies(data)
    weights = projection(data, group)
    x = data["distances"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for band in range(len(rows[0])):
        values = [row[band] - zero for row in rows]
        ax.plot(x, values, color="black", linewidth=1)
        if weights is not None:
            # The marker area is the weight, which is what a fat-band
            # diagram draws.
            ax.scatter(x, values, s=[80 * w[band] for w in weights], color="tab:red", alpha=0.6)
    for tick in data["ticks"]:
        ax.axvline(tick["x"], color="grey", linewidth=0.5)
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_xticks([t["x"] for t in data["ticks"]])
    ax.set_xticklabels([_GREEK.get(t["label"], t["label"]) for t in data["ticks"]])
    ax.set_xlim(x[0], x[-1])
    if window is not None:
        ax.set_ylim(*window)
    ax.set_ylabel(f"E - E({'VBM' if name.startswith('the valence') else 'Fermi'}) (eV)")
    title = f"{data['lattice']} {data['path']}"
    ax.set_title(title if group is None else f"{title}, {group}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bands", type=Path, help="a -bands.json file from band_structure")
    parser.add_argument("--dat", type=Path, help="write the band table here")
    parser.add_argument("--png", type=Path, help="draw the band diagram here (needs matplotlib)")
    parser.add_argument(
        "--projection",
        metavar="GROUP",
        help="add the weight of this group, such as Si-p, from a projected run",
    )
    span = parser.add_mutually_exclusive_group()
    span.add_argument(
        "--window",
        nargs=2,
        type=float,
        metavar=("LO", "HI"),
        help="the energy window of --png in eV relative to the reference "
        "(default: -8 to 8 above the conduction band minimum)",
    )
    span.add_argument("--all-bands", action="store_true", help="draw every band in --png")
    args = parser.parse_args(argv)
    if (args.window is not None or args.all_bands) and args.png is None:
        parser.error("--window and --all-bands set the window of --png; give --png")

    data = load(args.bands)
    summary = dict(data["summary"])
    report = {
        "path": data["path"],
        "lattice": data["lattice"],
        "n_kpoints": len(data["kpoints"]),
        "n_bands": data["n_bands"],
        "projection_groups": data.get("projection_groups", []),
        **summary,
    }
    if args.projection is not None:
        projection(data, args.projection)  # refuse an unknown group before any file is written
        report["projection"] = args.projection
    if args.dat is not None:
        write_dat(data, args.dat, args.projection)
        report["dat"] = str(args.dat)
    if args.png is not None:
        window = plot_window(data, tuple(args.window) if args.window else None, args.all_bands)
        write_png(data, args.png, args.projection, window)
        report["png"] = str(args.png)
        report["png_window_ev"] = [round(window[0], 3), round(window[1], 3)]
        report["bands_outside_window"] = bands_outside(data, window)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
