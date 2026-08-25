"""Mean-squared displacement and Einstein-relation diffusion coefficient.

    msd.py md.traj --dt-fs 5.0
    msd.py md.traj --dt-fs 5.0 --skip 200 --species Li --fit-from 0.5 --json

Averages over atoms and time origins, fits MSD = 6 D t from ``--fit-from``
of the maximum lag onward, and reports D in A^2/fs and cm^2/s. Positions
must be unwrapped (see the skill's caveats).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_A2FS_TO_CM2S = 0.1  # 1 A^2/fs = 1e-16 cm^2 / 1e-15 s


def _read_frames(path: Path) -> list[Any]:
    """Every frame in the file, as a list whatever ASE returns."""
    from ase.io import read

    try:
        result = read(path, index=":")
    except Exception as e:
        raise SystemExit(f"error: cannot read {path} with ASE: {e}") from e
    return list(result) if isinstance(result, list) else [result]


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
        "--fit-from", type=float, default=0.5,
        help="start of the linear fit as a fraction of the maximum lag (default 0.5)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if args.dt_fs <= 0:
        raise SystemExit("error: --dt-fs must be a positive time in femtoseconds")
    if not 0.0 < args.fit_from < 1.0:
        raise SystemExit("error: --fit-from must sit strictly between 0 and 1")
    if not args.data.is_file():
        raise SystemExit(f"error: no such file: {args.data}")
    frames = _read_frames(args.data)[args.skip :]
    if len(frames) < 10:
        raise SystemExit(
            f"error: {len(frames)} frame(s) after --skip is not enough for an MSD; "
            f"10 is a bare minimum and hundreds are typical"
        )

    if args.species:
        symbols = np.asarray(frames[0].get_chemical_symbols())
        mask = symbols == args.species
        if not mask.any():
            raise SystemExit(
                f"error: no {args.species} atoms; present: {', '.join(sorted(set(symbols)))}"
            )
    else:
        mask = np.ones(len(frames[0]), dtype=bool)
    trajectory = np.stack([atoms.get_positions()[mask] for atoms in frames])

    n_frames = len(trajectory)
    max_lag = n_frames // 2  # origins never overlap the lag window's tail
    lags = np.arange(1, max_lag + 1)
    msd = np.empty(max_lag)
    for i, lag in enumerate(lags):
        displacement = trajectory[lag:] - trajectory[:-lag]
        msd[i] = float((displacement**2).sum(axis=-1).mean())

    times = lags * args.dt_fs
    start = int(len(lags) * args.fit_from)
    slope, intercept = np.polyfit(times[start:], msd[start:], 1)
    diffusion_a2fs = float(slope) / 6.0
    diffusion_cm2s = diffusion_a2fs * _A2FS_TO_CM2S

    if args.json:
        print(
            json.dumps(
                {
                    "t_fs": [float(t) for t in times],
                    "msd_A2": [round(float(x), 6) for x in msd],
                    "fit_from_fs": float(times[start]),
                    "d_A2_per_fs": diffusion_a2fs,
                    "d_cm2_per_s": diffusion_cm2s,
                    "frames": n_frames,
                    "atoms_averaged": int(mask.sum()),
                },
                indent=1,
            )
        )
        return 0
    print(f"{'t (fs)':>10}  {'MSD (A^2)':>12}")
    step = max(1, len(times) // 20)
    for t, x in zip(times[::step], msd[::step], strict=True):
        print(f"{t:>10.1f}  {x:>12.4f}")
    print(
        f"fit over t >= {times[start]:.1f} fs: MSD = 6 D t + {intercept:.3f}; "
        f"D = {diffusion_a2fs:.3e} A^2/fs = {diffusion_cm2s:.3e} cm^2/s"
    )
    print(
        f"({n_frames} frames, {int(mask.sum())} atom(s) averaged; trust D only "
        f"if the tail is linear - a plateau means solid, not slow diffusion)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
