"""Crystalline fraction and interface velocity from a coexistence trajectory.

    interface_velocity.py coex-1300K.traj --dt-fs 100
    interface_velocity.py coex-1300K-r*.traj --dt-fs 100 --axis z --fit-from 0.2 --json

Each atom is classified as crystalline or liquid by the averaged
Steinhardt bond-order parameter q6-bar (Lechner-Dellago: the q6 vector
averaged over the atom and its neighbours within ``--cutoff``), above
``--threshold``. fcc, bcc, and hcp atoms sit near 0.4 to 0.6, liquid atoms
near 0.1 to 0.2; 0.33 separates them for most metals. The crystalline
fraction f_c per frame times the cell length along ``--axis`` is the
crystal slab's length; its slope over the steady window (``--fit-from``
to ``--fit-to`` as fractions of the run) divided by ``--interfaces`` (2 in
a periodic cell) is the velocity of one interface, positive for growth
and negative for melting, in A/ps and m/s with the fit's standard error.
Several files are replicas and give a mean and a standard error over them.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_A_PS_TO_M_S = 100.0
_AXES = {"x": 0, "y": 1, "z": 2}


def _read_frames(path: Path) -> list[Any]:
    from ase.io import read

    try:
        result = read(path, index=":")
    except Exception as e:
        raise SystemExit(f"error: cannot read {path} with ASE: {e}") from e
    return list(result) if isinstance(result, list) else [result]


def averaged_q6(atoms: Any, cutoff: float) -> np.ndarray:
    """Lechner-Dellago q6-bar per atom."""
    from ase.neighborlist import neighbor_list
    from scipy.special import sph_harm_y  # type: ignore[import-untyped]

    n = len(atoms)
    i, j, vectors = neighbor_list("ijD", atoms, cutoff)  # type: ignore[no-untyped-call]
    if len(i) == 0:
        raise SystemExit(
            f"error: no neighbours within {cutoff:g} A; raise --cutoff above the first shell"
        )
    r = np.linalg.norm(vectors, axis=1)
    theta = np.arccos(np.clip(vectors[:, 2] / r, -1.0, 1.0))
    phi = np.arctan2(vectors[:, 1], vectors[:, 0])
    counts = np.bincount(i, minlength=n).astype(float)
    q6m = np.zeros((n, 13), dtype=complex)
    for k, m in enumerate(range(-6, 7)):
        harmonics = sph_harm_y(6, m, theta, phi)
        q6m[:, k] = np.bincount(i, weights=harmonics.real, minlength=n) + 1j * np.bincount(
            i, weights=harmonics.imag, minlength=n
        )
    safe = np.where(counts > 0, counts, 1.0)
    q6m /= safe[:, None]
    # average over the atom and its neighbours
    summed = q6m.copy()
    np.add.at(summed, i, q6m[j])
    averaged = summed / (counts + 1.0)[:, None]
    return np.asarray(np.sqrt(4.0 * math.pi / 13.0 * (np.abs(averaged) ** 2).sum(axis=1)))


def analyse(
    frames: list[Any], cutoff: float, threshold: float, axis: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(crystalline fraction, cell length along the axis, mean q6-bar) per frame."""
    fractions, lengths, means = [], [], []
    for atoms in frames:
        q6 = averaged_q6(atoms, cutoff)
        fractions.append(float((q6 > threshold).mean()))
        lengths.append(float(atoms.cell.lengths()[axis]))
        means.append(float(q6.mean()))
    return np.array(fractions), np.array(lengths), np.array(means)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, nargs="+", help="coexistence trajectories (replicas)")
    parser.add_argument(
        "--dt-fs", type=float, required=True, help="time between saved frames in fs"
    )
    parser.add_argument("--axis", default="z", choices=sorted(_AXES), help="growth axis")
    parser.add_argument(
        "--cutoff", type=float,
        help="neighbour cutoff in A (default 1.2 times the first frame's median "
        "nearest-neighbour distance; set it from the crystal's first shell)",
    )
    parser.add_argument("--threshold", type=float, default=0.33, help="q6-bar above = crystal")
    parser.add_argument("--interfaces", type=int, default=2, help="moving interfaces in the cell")
    parser.add_argument("--fit-from", type=float, default=0.1, help="window start (fraction)")
    parser.add_argument("--fit-to", type=float, default=0.9, help="window end (fraction)")
    parser.add_argument("--skip", type=int, default=0, help="frames to drop from the start")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if args.dt_fs <= 0:
        raise SystemExit("error: --dt-fs must be a positive time in femtoseconds")
    if not 0.0 <= args.fit_from < args.fit_to <= 1.0:
        raise SystemExit("error: the window needs 0 <= --fit-from < --fit-to <= 1")
    if args.interfaces < 1:
        raise SystemExit("error: --interfaces must be at least 1")
    if not 0.0 < args.threshold < 1.0:
        raise SystemExit("error: --threshold is a q6-bar value between 0 and 1")
    axis = _AXES[args.axis]

    replicas: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in args.data:
        if not path.is_file():
            raise SystemExit(f"error: no such file: {path}")
        frames = _read_frames(path)[args.skip :]
        if len(frames) < 5:
            raise SystemExit(
                f"error: {path} has {len(frames)} frame(s) after --skip; 5 is the floor"
            )
        cutoff = args.cutoff
        if cutoff is None:
            from ase.neighborlist import neighbor_list

            first, distances = neighbor_list("id", frames[0], 6.0)  # type: ignore[no-untyped-call]
            if len(distances) == 0:
                raise SystemExit("error: no pair within 6 A in the first frame; pass --cutoff")
            nearest = np.full(len(frames[0]), np.inf)
            np.minimum.at(nearest, first, distances)
            cutoff = 1.2 * float(np.median(nearest[np.isfinite(nearest)]))
        fractions, lengths, means = analyse(frames, cutoff, args.threshold, axis)
        times_ps = np.arange(len(frames)) * args.dt_fs / 1000.0
        crystal_length = fractions * lengths
        start = round(len(frames) * args.fit_from)
        stop = max(start + 3, round(len(frames) * args.fit_to))
        if stop > len(frames):
            raise SystemExit("error: the fit window holds fewer than 3 frames; record more")
        window_t, window_l = times_ps[start:stop], crystal_length[start:stop]
        if len(window_t) >= 4:
            (slope, intercept), cov = np.polyfit(window_t, window_l, 1, cov=True)
            se = float(math.sqrt(max(cov[0, 0], 0.0))) / args.interfaces
        else:
            slope, intercept = np.polyfit(window_t, window_l, 1)
            se = None
        predicted = slope * window_t + intercept
        total = float(((window_l - window_l.mean()) ** 2).sum())
        r2 = 1.0 if total == 0.0 else 1.0 - float(((window_l - predicted) ** 2).sum()) / total
        velocity = float(slope) / args.interfaces
        replica: dict[str, Any] = {
            "file": str(path),
            "cutoff_A": cutoff,
            "frames": len(frames),
            "fraction_first": float(fractions[0]),
            "fraction_last": float(fractions[-1]),
            "q6bar_mean_first": float(means[0]),
            "v_A_per_ps": velocity,
            "v_se_A_per_ps": se,
            "v_m_per_s": velocity * _A_PS_TO_M_S,
            "r2": r2,
            "window_ps": [float(window_t[0]), float(window_t[-1])],
            "t_ps": [float(t) for t in times_ps],
            "crystal_fraction": [round(float(f), 5) for f in fractions],
        }
        replicas.append(replica)
        if fractions.min() < 0.05 or fractions.max() > 0.95:
            warnings.append(
                f"{path.name}: the crystalline fraction reaches "
                f"{fractions.min():.2f}-{fractions.max():.2f}; the run consumed a phase, "
                f"so the interfaces met and the late frames are not coexistence"
            )
        if r2 < 0.9:
            warnings.append(
                f"{path.name}: the crystal length is not linear in time over the window "
                f"(R^2 = {r2:.2f}); the interface has not reached steady state, or the "
                f"thermostat is fighting the latent heat"
            )
    velocities = np.array([r["v_A_per_ps"] for r in replicas])
    result: dict[str, Any] = {
        "axis": args.axis,
        "threshold": args.threshold,
        "interfaces": args.interfaces,
        "v_A_per_ps": float(velocities.mean()),
        "v_m_per_s": float(velocities.mean()) * _A_PS_TO_M_S,
        "v_se_A_per_ps": (
            float(velocities.std(ddof=1) / math.sqrt(len(velocities)))
            if len(velocities) > 1
            else replicas[0]["v_se_A_per_ps"]
        ),
        "replicas": replicas,
        "warnings": warnings,
    }
    if len(velocities) == 1:
        warnings.append(
            "one trajectory: the error is the fit's, not the run-to-run spread; run "
            "replicas from different seeds"
        )

    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    for replica in replicas:
        se = f" +/- {replica['v_se_A_per_ps']:.4f}" if replica["v_se_A_per_ps"] else ""
        print(
            f"{replica['file']}: f_c {replica['fraction_first']:.2f} -> "
            f"{replica['fraction_last']:.2f}, v = {replica['v_A_per_ps']:+.4f}{se} A/ps "
            f"per interface (R^2 = {replica['r2']:.3f}, cutoff {replica['cutoff_A']:.2f} A)"
        )
    se = f" +/- {result['v_se_A_per_ps']:.4f}" if result["v_se_A_per_ps"] else ""
    print(
        f"v = {result['v_A_per_ps']:+.4f}{se} A/ps = {result['v_m_per_s']:+.2f} m/s per "
        f"interface along {args.axis} ({args.interfaces} interfaces; growth positive)"
    )
    for warning in warnings:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
