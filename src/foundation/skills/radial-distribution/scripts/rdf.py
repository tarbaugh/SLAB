"""Radial distribution function g(r) from an ASE-readable file.

    rdf.py md.traj
    rdf.py md.traj --skip 200 --bins 200 --species Cu Cu --json

Averages over frames (after ``--skip``), uses the minimum-image
convention, and caps ``--rmax`` at half the smallest perpendicular cell
width of every frame, the largest radius the convention supports. Reports
the first peak (the first local maximum above 1), the first minimum after
it, the coordination number up to that minimum, the mean of g over the
last tenth of r (1 when the normalisation and the sampling are right),
and a block standard error of the first-peak height. Cells above 2000
atoms use a neighbour list instead of the full pair matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

NEIGHBOR_LIST_ABOVE = 2000
TAIL_FRACTION = 0.1


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
    """Minimum-image A-B distances over ordered pairs (self-pairs excluded)."""
    fractional = positions @ np.linalg.inv(cell)
    delta = fractional[mask_a, None, :] - fractional[None, mask_b, :]
    delta -= np.round(delta)
    vectors = delta @ cell
    distances = np.sqrt((vectors**2).sum(axis=-1))
    same = np.equal(np.nonzero(mask_a)[0][:, None], np.nonzero(mask_b)[0][None, :])
    return np.asarray(distances[~same])


def _neighbor_distances(
    atoms: Any, mask_a: np.ndarray, mask_b: np.ndarray, rmax: float
) -> np.ndarray:
    """Ordered A-B pair distances up to rmax from a neighbour list."""
    from ase.neighborlist import neighbor_list

    first, second, distances = neighbor_list("ijd", atoms, rmax)  # type: ignore[no-untyped-call]
    keep = mask_a[first] & mask_b[second] & (distances > 0)
    return np.asarray(distances[keep])


def first_peak_and_minimum(g: np.ndarray) -> tuple[int, int | None]:
    """Index of the first local maximum above 1, and of the first local
    minimum after it (None when g never turns back up)."""
    peak = None
    for i in range(1, len(g) - 1):
        if g[i] > 1.0 and g[i] >= g[i - 1] and g[i] >= g[i + 1]:
            peak = i
            break
    if peak is None:
        peak = int(np.argmax(g))
    minimum = None
    for j in range(peak + 1, len(g) - 1):
        if g[j] <= g[j - 1] and g[j] <= g[j + 1] and g[j] < g[peak]:
            minimum = j
            break
    return peak, minimum


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, help="an ASE-readable trajectory or structure")
    parser.add_argument("--skip", type=int, default=0, help="frames to drop from the start")
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--rmax", type=float, help="max radius (default: half min cell width)")
    parser.add_argument(
        "--species", nargs=2, metavar=("A", "B"), help="restrict to A-B pairs"
    )
    parser.add_argument(
        "--blocks", type=int, default=5,
        help="frame blocks for the first-peak standard error (default 5)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if not args.data.is_file():
        raise SystemExit(f"error: no such file: {args.data}")
    if args.bins < 10:
        raise SystemExit("error: --bins must be at least 10")
    frames = _read_frames(args.data)[args.skip :]
    if not frames:
        raise SystemExit(f"error: no frames left after --skip {args.skip}")

    limits = [_half_min_width(np.asarray(atoms.get_cell())) for atoms in frames]
    limit = min(limits)
    rmax = args.rmax if args.rmax is not None else 0.999 * limit
    if rmax > limit + 1e-9:
        frame = int(np.argmin(limits))
        raise SystemExit(
            f"error: --rmax {rmax:.3f} exceeds half the smallest cell width "
            f"({limit:.3f} A at frame {frame + args.skip}), where the minimum-image "
            f"convention breaks; enlarge the cell instead"
        )

    edges = np.linspace(0.0, rmax, args.bins + 1)
    counts = np.zeros(args.bins)
    per_frame_counts: list[np.ndarray] = []
    n_a = n_b = overlap = 0
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
        if len(atoms) > NEIGHBOR_LIST_ABOVE:
            distances = _neighbor_distances(atoms, mask_a, mask_b, rmax)
        else:
            distances = _pair_distances(atoms.get_positions(), cell, mask_a, mask_b)
        frame_counts = np.histogram(distances, bins=edges)[0]
        counts += frame_counts
        per_frame_counts.append(frame_counts)
        n_a, n_b = int(mask_a.sum()), int(mask_b.sum())
        overlap = int((mask_a & mask_b).sum())  # atoms whose self-pair was excluded
        volume_sum += abs(float(np.linalg.det(cell)))
    volume = volume_sum / len(frames)

    centers = 0.5 * (edges[:-1] + edges[1:])
    shell = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    ideal_pairs = n_a * n_b - overlap  # ordered pairs, self-pairs removed
    if ideal_pairs <= 0:
        raise SystemExit("error: a single atom has no pairs; g(r) needs at least two")
    normalisation = len(frames) * shell * ideal_pairs / volume
    g = counts / normalisation

    peak, minimum = first_peak_and_minimum(g)
    coordination = (
        float(counts[: minimum + 1].sum() / (len(frames) * n_a)) if minimum is not None else None
    )
    tail_bins = max(1, int(args.bins * TAIL_FRACTION))
    tail_mean = float(g[-tail_bins:].mean())
    warnings: list[str] = []
    if abs(tail_mean - 1.0) > 0.05:
        warnings.append(
            f"g averages {tail_mean:.3f} over the last tenth of r, not 1: the sample "
            f"is too small or too few frames, or the cell is not in equilibrium"
        )
    if minimum is None:
        warnings.append("g has no minimum after the first peak; no coordination number")

    peak_se: float | None = None
    if args.blocks >= 2 and len(frames) >= 2 * args.blocks:
        size = len(frames) // args.blocks
        heights = []
        for b in range(args.blocks):
            block = np.sum(per_frame_counts[b * size : (b + 1) * size], axis=0)
            heights.append(float(block[peak] / (size * shell[peak] * ideal_pairs / volume)))
        peak_se = float(np.std(heights, ddof=1) / np.sqrt(len(heights)))

    if args.json:
        print(
            json.dumps(
                {
                    "r_A": [round(float(r), 5) for r in centers],
                    "g": [round(float(x), 5) for x in g],
                    "first_peak_r_A": float(centers[peak]),
                    "first_peak_g": float(g[peak]),
                    "first_peak_g_se": peak_se,
                    "first_minimum_r_A": float(centers[minimum]) if minimum is not None else None,
                    "coordination_number": coordination,
                    "tail_mean_g": tail_mean,
                    "frames": len(frames),
                    "rmax_A": rmax,
                    "bin_width_A": float(edges[1] - edges[0]),
                    "warnings": warnings,
                },
                indent=1,
            )
        )
        return 0
    print(f"{'r (A)':>8}  {'g(r)':>8}")
    for r, x in zip(centers, g, strict=True):
        print(f"{r:>8.3f}  {x:>8.3f}")
    se_text = f" +/- {peak_se:.2f}" if peak_se is not None else ""
    print(
        f"first peak: r = {centers[peak]:.3f} A, g = {g[peak]:.2f}{se_text} "
        f"({len(frames)} frame(s), rmax {rmax:.3f} A, bins {edges[1] - edges[0]:.3f} A)"
    )
    if minimum is not None:
        print(
            f"first minimum: r = {centers[minimum]:.3f} A; coordination number "
            f"{coordination:.2f}"
        )
    print(f"tail: g averages {tail_mean:.3f} over the last tenth of r")
    for warning in warnings:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
