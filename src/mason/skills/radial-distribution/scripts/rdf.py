"""Radial distribution function g(r) from an ASE-readable file.

    rdf.py md.traj
    rdf.py md.traj --skip 200 --bins 200 --species Cu Cu --json

Averages over frames (after ``--skip``), uses the minimum-image
convention, and caps ``--rmax`` at half the smallest perpendicular cell
width, the largest radius the convention supports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _read_frames(path: Path) -> list[Any]:
    """Every frame in the file, as a list whatever ASE returns."""
    from ase.io import read

    try:
        result = read(path, index=":")
    except Exception as e:
        raise SystemExit(f"error: cannot read {path} with ASE: {e}") from e
    return list(result) if isinstance(result, list) else [result]


def _half_min_width(cell: np.ndarray) -> float:
    """Half the smallest perpendicular width of the cell."""
    volume = abs(float(np.linalg.det(cell)))
    if volume < 1e-9:
        raise SystemExit("error: the cell has no volume; g(r) needs a periodic cell")
    widths = []
    for i in range(3):
        cross = np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])
        widths.append(volume / float(np.linalg.norm(cross)))
    return 0.5 * min(widths)


def _pair_distances(
    positions: np.ndarray, cell: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray
) -> np.ndarray:
    """Minimum-image A-B distances (self-pairs excluded)."""
    fractional = positions @ np.linalg.inv(cell)
    delta = fractional[mask_a, None, :] - fractional[None, mask_b, :]
    delta -= np.round(delta)
    vectors = delta @ cell
    distances = np.sqrt((vectors**2).sum(axis=-1))
    same = np.equal(np.nonzero(mask_a)[0][:, None], np.nonzero(mask_b)[0][None, :])
    return np.asarray(distances[~same])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, help="an ASE-readable trajectory or structure")
    parser.add_argument("--skip", type=int, default=0, help="frames to drop from the start")
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--rmax", type=float, help="max radius (default: half min cell width)")
    parser.add_argument(
        "--species", nargs=2, metavar=("A", "B"), help="restrict to A-B pairs"
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if not args.data.is_file():
        raise SystemExit(f"error: no such file: {args.data}")
    frames = _read_frames(args.data)[args.skip :]
    if not frames:
        raise SystemExit(f"error: no frames left after --skip {args.skip}")

    cell0 = np.asarray(frames[0].get_cell())
    limit = _half_min_width(cell0)
    rmax = args.rmax if args.rmax is not None else 0.999 * limit
    if rmax > limit + 1e-9:
        raise SystemExit(
            f"error: --rmax {rmax:.3f} exceeds half the smallest cell width "
            f"({limit:.3f} A), where the minimum-image convention breaks; "
            f"enlarge the cell instead"
        )

    edges = np.linspace(0.0, rmax, args.bins + 1)
    counts = np.zeros(args.bins)
    n_a = n_b = 0
    volume_sum = 0.0
    for atoms in frames:
        symbols = np.asarray(atoms.get_chemical_symbols())
        if args.species:
            mask_a = symbols == args.species[0]
            mask_b = symbols == args.species[1]
        else:
            mask_a = mask_b = np.ones(len(atoms), dtype=bool)
        if not mask_a.any() or not mask_b.any():
            raise SystemExit(
                f"error: no {'-'.join(args.species)} pairs; present: "
                f"{', '.join(sorted(set(symbols)))}"
            )
        cell = np.asarray(atoms.get_cell())
        distances = _pair_distances(atoms.get_positions(), cell, mask_a, mask_b)
        counts += np.histogram(distances, bins=edges)[0]
        n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
        volume_sum += abs(float(np.linalg.det(cell)))
    volume = volume_sum / len(frames)

    centers = 0.5 * (edges[:-1] + edges[1:])
    shell = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    pair_density = n_a * n_b / volume  # self-pairs are excluded in the counts
    g = counts / (len(frames) * shell * pair_density)

    peak = int(np.argmax(g))
    if args.json:
        print(
            json.dumps(
                {
                    "r_A": [round(float(r), 5) for r in centers],
                    "g": [round(float(x), 5) for x in g],
                    "first_peak_r_A": float(centers[peak]),
                    "first_peak_g": float(g[peak]),
                    "frames": len(frames),
                    "rmax_A": rmax,
                },
                indent=1,
            )
        )
        return 0
    print(f"{'r (A)':>8}  {'g(r)':>8}")
    for r, x in zip(centers, g, strict=True):
        print(f"{r:>8.3f}  {x:>8.3f}")
    print(
        f"first peak: r = {centers[peak]:.3f} A, g = {g[peak]:.2f} "
        f"({len(frames)} frame(s), rmax {rmax:.3f} A)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
