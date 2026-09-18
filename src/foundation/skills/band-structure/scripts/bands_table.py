"""Read a band_structure result file: the gap verdict, a table, a plot.

Input: the ``-bands.json`` artifact that ``band_structure`` keeps (read it
with ``read_artifact`` and save it, or pass the path of a kept copy). It
holds the path, the k-point distances, the eigenvalues in eV, the SCF
Fermi level, and the summary.

    bands_table.py si-bands.json
    bands_table.py si-bands.json --dat si-bands.dat
    bands_table.py si-bands.json --png si-bands.png

With no option the script prints the summary as JSON. ``--dat`` writes a
whitespace table: the distance along the path in 1/Å, then one column
per band, in eV relative to the valence band maximum (or to the Fermi
level for a metal). Comment lines at the top give the reference energy
and the x position of each special point. ``--png`` draws the same
diagram when matplotlib is installed and exits 2 when it is not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

#: The labels ASE uses, as a band diagram prints them.
_GREEK = {"G": "Γ"}


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


def write_dat(data: dict[str, Any], out: Path) -> None:
    zero, name = reference(data)
    rows = energies(data)
    ticks = " ".join(f"{t['label']} {t['x']:.6f}" for t in data["ticks"])
    lines = [
        f"# band structure along {data['path']} ({data['lattice']}), "
        f"{len(rows)} k-points, {len(rows[0])} bands",
        f"# energies in eV relative to {name} ({zero:.4f} eV)",
        f"# special points (label x): {ticks}",
        "# columns: x (1/A), then band 1 to band " + str(len(rows[0])),
    ]
    for x, row in zip(data["distances"], rows, strict=True):
        lines.append(" ".join([f"{x:.6f}", *(f"{e - zero:.4f}" for e in row)]))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_png(data: dict[str, Any], out: Path) -> None:
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
    x = data["distances"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for band in range(len(rows[0])):
        ax.plot(x, [row[band] - zero for row in rows], color="black", linewidth=1)
    for tick in data["ticks"]:
        ax.axvline(tick["x"], color="grey", linewidth=0.5)
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_xticks([t["x"] for t in data["ticks"]])
    ax.set_xticklabels([_GREEK.get(t["label"], t["label"]) for t in data["ticks"]])
    ax.set_xlim(x[0], x[-1])
    ax.set_ylabel(f"E - E({'VBM' if name.startswith('the valence') else 'Fermi'}) (eV)")
    ax.set_title(f"{data['lattice']} {data['path']}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bands", type=Path, help="a -bands.json file from band_structure")
    parser.add_argument("--dat", type=Path, help="write the band table here")
    parser.add_argument("--png", type=Path, help="draw the band diagram here (needs matplotlib)")
    args = parser.parse_args(argv)

    data = load(args.bands)
    summary = dict(data["summary"])
    report = {
        "path": data["path"],
        "lattice": data["lattice"],
        "n_kpoints": len(data["kpoints"]),
        "n_bands": data["n_bands"],
        **summary,
    }
    if args.dat is not None:
        write_dat(data, args.dat)
        report["dat"] = str(args.dat)
    if args.png is not None:
        write_png(data, args.png)
        report["png"] = str(args.png)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
