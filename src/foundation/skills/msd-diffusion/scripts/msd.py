"""Mean-squared displacement and Einstein-relation diffusion coefficient.

    msd.py md.traj --dt-fs 5.0
    msd.py md.traj --dt-fs 5.0 --skip 200 --species Li --axes xy --json
    msd.py md.traj --dt-fs 5.0 --yeh-hummer 1.5e-3 --temperature 1400

Averages over atoms and time origins, fits MSD = 2 d D t over lags between
``--fit-from`` and ``--fit-to`` of the trajectory (defaults 0.05 and 0.25;
longer lags have too few independent samples), and reports D in A^2/fs
and cm^2/s with:

* the log-log slope beta of MSD(t) over the window (1 in the diffusive
  regime; well below 1 is a cage or a plateau),
* D per axis, and D over the axes named by ``--axes`` (d = 1, 2, or 3),
* a standard error from ``--blocks`` contiguous blocks of the trajectory,
* the Yeh-Hummer finite-size estimate D_inf = D + xi k_B T / (6 pi eta L)
  when ``--yeh-hummer`` gives the shear viscosity in Pa s and
  ``--temperature`` the temperature in K (liquids only).

The centre-of-mass drift of the whole cell is subtracted unless
``--keep-com`` is given. Positions must be unwrapped, and a cell that
changes between frames (an NPT run) is flagged: the barostat's rescaling
inflates the MSD at long lags.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_A2FS_TO_CM2S = 0.1  # 1 A^2/fs = 1e-16 cm^2 / 1e-15 s
_KB_J = 1.380649e-23
_XI_CUBIC = 2.837297
_AXES = {"x": 0, "y": 1, "z": 2}


def _read_frames(path: Path) -> list[Any]:
    """Every frame in the file, as a list whatever ASE returns."""
    from ase.io import read

    try:
        result = read(path, index=":")
    except Exception as e:
        raise SystemExit(f"error: cannot read {path} with ASE: {e}") from e
    return list(result) if isinstance(result, list) else [result]


def msd_by_component(trajectory: np.ndarray, max_lag: int) -> np.ndarray:
    """MSD per Cartesian component, shape (max_lag, 3), lags 1..max_lag,
    averaged over atoms and every time origin."""
    out = np.empty((max_lag, 3))
    for i in range(max_lag):
        lag = i + 1
        displacement = trajectory[lag:] - trajectory[:-lag]
        out[i] = (displacement**2).mean(axis=(0, 1))
    return out


def fit_window(n_lags: int, fit_from: float, fit_to: float) -> tuple[int, int]:
    start = round(n_lags * fit_from)
    stop = max(start + 1, round(n_lags * fit_to))
    return start, stop


def diffusion(
    trajectory: np.ndarray, dt_fs: float, axes: list[int], fit_from: float, fit_to: float
) -> dict[str, Any]:
    """D over *axes* from one trajectory, with the per-axis slopes and beta."""
    n_frames = len(trajectory)
    max_lag = n_frames // 2
    components = msd_by_component(trajectory, max_lag)
    times = np.arange(1, max_lag + 1) * dt_fs
    start, stop = fit_window(max_lag, fit_from, fit_to)
    if stop - start < 3:
        raise SystemExit(
            f"error: the fit window holds {stop - start} lag(s); a fit needs at least 3. "
            f"Record more frames or widen --fit-from/--fit-to"
        )
    msd = components[:, axes].sum(axis=1)
    slope, intercept = np.polyfit(times[start:stop], msd[start:stop], 1)
    per_axis = {
        name: float(np.polyfit(times[start:stop], components[start:stop, k], 1)[0] / 2.0)
        for name, k in _AXES.items()
    }
    window_msd = msd[start:stop]
    beta = (
        float(np.polyfit(np.log(times[start:stop]), np.log(window_msd), 1)[0])
        if (window_msd > 0).all()
        else float("nan")
    )
    return {
        "times": times,
        "msd": msd,
        "start": start,
        "stop": stop,
        "d_A2_per_fs": float(slope) / (2.0 * len(axes)),
        "intercept": float(intercept),
        "per_axis": per_axis,
        "beta": beta,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, help="an ASE-readable trajectory")
    parser.add_argument(
        "--dt-fs", type=float, required=True,
        help="time between saved frames in fs (MD timestep x save interval)",
    )
    parser.add_argument("--skip", type=int, default=0, help="frames to drop from the start")
    parser.add_argument("--species", help="restrict the average to one species")
    parser.add_argument(
        "--axes", default="xyz",
        help="axes the diffusion runs along: xyz (d=3), xy/xz/yz (d=2), x/y/z (d=1)",
    )
    parser.add_argument(
        "--fit-from", type=float, default=0.05,
        help="start of the linear fit as a fraction of the maximum lag (default 0.05)",
    )
    parser.add_argument(
        "--fit-to", type=float, default=0.25,
        help="end of the linear fit as a fraction of the maximum lag (default 0.25; "
        "at most 0.5, where a lag has one independent sample)",
    )
    parser.add_argument(
        "--blocks", type=int, default=5,
        help="contiguous blocks for the standard error (default 5; 0 disables)",
    )
    parser.add_argument(
        "--keep-com", action="store_true",
        help="do not subtract the whole cell's centre-of-mass drift",
    )
    parser.add_argument(
        "--yeh-hummer", type=float, metavar="ETA",
        help="shear viscosity in Pa s: also report the Yeh-Hummer infinite-size D",
    )
    parser.add_argument("--temperature", type=float, help="temperature in K for --yeh-hummer")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if args.dt_fs <= 0:
        raise SystemExit("error: --dt-fs must be a positive time in femtoseconds")
    if not 0.0 <= args.fit_from < args.fit_to <= 0.5:
        raise SystemExit(
            "error: the fit window needs 0 <= --fit-from < --fit-to <= 0.5 "
            "(lags beyond half the run have a single sample)"
        )
    axes_text = args.axes.lower()
    unknown = any(a not in _AXES for a in axes_text)
    if not axes_text or unknown or len(set(axes_text)) != len(axes_text):
        raise SystemExit("error: --axes must be a subset of xyz, each axis once")
    axes = [_AXES[a] for a in axes_text]
    if args.blocks < 0:
        raise SystemExit("error: --blocks must be 0 or positive")
    if (args.yeh_hummer is None) != (args.temperature is None):
        raise SystemExit("error: --yeh-hummer needs --temperature, and the reverse")
    if args.yeh_hummer is not None and (args.yeh_hummer <= 0 or args.temperature <= 0):
        raise SystemExit("error: the viscosity and the temperature must be positive")
    if not args.data.is_file():
        raise SystemExit(f"error: no such file: {args.data}")
    frames = _read_frames(args.data)[args.skip :]
    if len(frames) < 10:
        raise SystemExit(
            f"error: {len(frames)} frame(s) after --skip is not enough for an MSD; "
            f"10 is a bare minimum and hundreds are typical"
        )

    warnings: list[str] = []
    if args.species:
        symbols = np.asarray(frames[0].get_chemical_symbols())
        mask = symbols == args.species
        if not mask.any():
            raise SystemExit(
                f"error: no {args.species} atoms; present: {', '.join(sorted(set(symbols)))}"
            )
    else:
        mask = np.ones(len(frames[0]), dtype=bool)
    positions = np.stack([atoms.get_positions() for atoms in frames])
    masses = np.asarray(frames[0].get_masses())
    if not args.keep_com:
        com = (positions * masses[None, :, None]).sum(axis=1) / masses.sum()
        positions = positions - com[:, None, :]
    trajectory = positions[:, mask, :]

    cells = np.stack([np.asarray(atoms.get_cell()) for atoms in frames])
    if np.abs(cells - cells[0]).max() > 1e-6:
        warnings.append(
            "the cell changes between frames: an NPT trajectory carries the "
            "barostat's rescaling in its positions, which inflates the MSD at long "
            "lags; compute D from an NVE or NVT production run"
        )
    volume = float(np.mean([abs(np.linalg.det(cell)) for cell in cells]))
    box_length = volume ** (1.0 / 3.0) if volume > 0 else None

    result = diffusion(trajectory, args.dt_fs, axes, args.fit_from, args.fit_to)
    d_a2fs = result["d_A2_per_fs"]
    d_cm2s = d_a2fs * _A2FS_TO_CM2S
    beta = result["beta"]
    if np.isnan(beta):
        warnings.append("the MSD is zero somewhere in the window; nothing moved")
    elif beta < 0.3:
        warnings.append(
            f"beta = {beta:.2f}: the MSD has plateaued; the system is solid over this "
            f"window and D is not a diffusion coefficient"
        )
    elif abs(beta - 1.0) > 0.1:
        warnings.append(
            f"beta = {beta:.2f} is not 1: the window is not in the diffusive regime "
            f"(caged below 1, ballistic or drifting above); move or extend it"
        )

    se_cm2s: float | None = None
    block_values: list[float] = []
    if args.blocks >= 2:
        n_frames = len(trajectory)
        size = n_frames // args.blocks
        if size >= 10 and fit_window(size // 2, args.fit_from, args.fit_to)[1] - fit_window(
            size // 2, args.fit_from, args.fit_to
        )[0] >= 3:
            for b in range(args.blocks):
                block = trajectory[b * size : (b + 1) * size]
                block_values.append(
                    diffusion(block, args.dt_fs, axes, args.fit_from, args.fit_to)["d_A2_per_fs"]
                    * _A2FS_TO_CM2S
                )
            se_cm2s = float(np.std(block_values, ddof=1) / np.sqrt(len(block_values)))
        else:
            warnings.append(
                f"too few frames for {args.blocks} blocks; no standard error. Average "
                f"independent runs instead and state their spread"
            )

    d_inf: float | None = None
    if args.yeh_hummer is not None and box_length is not None:
        correction_m2s = _XI_CUBIC * _KB_J * args.temperature / (
            6.0 * np.pi * args.yeh_hummer * box_length * 1e-10
        )
        d_inf = d_cm2s + correction_m2s * 1e4

    times, msd, start, stop = result["times"], result["msd"], result["start"], result["stop"]
    if args.json:
        print(
            json.dumps(
                {
                    "t_fs": [float(t) for t in times],
                    "msd_A2": [round(float(x), 6) for x in msd],
                    "axes": axes_text,
                    "fit_from_fs": float(times[start]),
                    "fit_to_fs": float(times[stop - 1]),
                    "beta": beta,
                    "d_A2_per_fs": d_a2fs,
                    "d_cm2_per_s": d_cm2s,
                    "d_se_cm2_per_s": se_cm2s,
                    "d_blocks_cm2_per_s": block_values,
                    "d_per_axis_A2_per_fs": result["per_axis"],
                    "d_yeh_hummer_cm2_per_s": d_inf,
                    "box_length_A": box_length,
                    "frames": len(trajectory),
                    "atoms_averaged": int(mask.sum()),
                    "warnings": warnings,
                },
                indent=1,
            )
        )
        return 0
    print(f"{'t (fs)':>10}  {'MSD (A^2)':>12}")
    step = max(1, len(times) // 20)
    for t, x in zip(times[::step], msd[::step], strict=True):
        print(f"{t:>10.1f}  {x:>12.4f}")
    se_text = f" +/- {se_cm2s:.2e}" if se_cm2s is not None else ""
    print(
        f"fit over {times[start]:.1f} to {times[stop - 1]:.1f} fs along {axes_text}: "
        f"MSD = {2 * len(axes)} D t + {result['intercept']:.3f}; beta = {beta:.2f}"
    )
    print(f"D = {d_a2fs:.3e} A^2/fs = {d_cm2s:.3e}{se_text} cm^2/s")
    axis_text = ", ".join(f"{k} {v:.3e}" for k, v in result["per_axis"].items())
    print(f"D per axis (A^2/fs): {axis_text}")
    if d_inf is not None:
        print(
            f"Yeh-Hummer D_inf = {d_inf:.3e} cm^2/s for L = {box_length:.2f} A "
            f"(liquids only; the correction is {d_inf - d_cm2s:.2e})"
        )
    print(
        f"({len(trajectory)} frames, {int(mask.sum())} atom(s) averaged, "
        f"{'COM subtracted' if not args.keep_com else 'COM kept'})"
    )
    for warning in warnings:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
