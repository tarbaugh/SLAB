"""Fit a Birch-Murnaghan equation of state to energy-volume data.

Input is either a JSON file holding a list of ``{"volume", "energy"}``
points in Å^3 and eV (the shape ``assets/eos_scan.py`` writes), or any
ASE-readable trajectory whose images carry energies.

    fit_eos.py eos.json
    fit_eos.py scan.traj --json --natoms 4

Output: V0 (Å^3), E0 (eV), B (GPa), B' (the pressure derivative the
third-order form fits), the RMS residual of the fit, the scanned volume
range as fractions of V0, and with ``--natoms`` the volume per atom and
the residual per atom. Warnings go into the report in both modes: a
fitted V0 outside the inner points, a residual above 1 meV/atom, and a B
outside the plausible range. Exits nonzero with an actionable message on
bad input or a fit that does not converge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_points(path: Path) -> tuple[list[float], list[float]]:
    """Volumes and energies from a JSON point list or an ASE trajectory."""
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise SystemExit(f"error: {path} is not valid JSON: {e}") from e
        if not isinstance(data, list) or not all(
            isinstance(p, dict) and "volume" in p and "energy" in p for p in data
        ):
            raise SystemExit(
                f'error: {path} must hold a list of {{"volume", "energy"}} objects'
            )
        return (
            [float(p["volume"]) for p in data],
            [float(p["energy"]) for p in data],
        )
    from ase.io import read

    try:
        images = read(path, index=":")
    except Exception as e:  # ASE raises a zoo of types for unreadable files
        raise SystemExit(f"error: cannot read {path} with ASE: {e}") from e
    volumes: list[float] = []
    energies: list[float] = []
    for i, atoms in enumerate(images):
        try:
            energies.append(float(atoms.get_potential_energy()))
        except Exception as e:
            raise SystemExit(
                f"error: image {i} in {path} carries no energy ({e}); "
                f"record energies when writing the trajectory"
            ) from e
        volumes.append(float(atoms.get_volume()))
    return volumes, energies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, help="eos.json or an ASE-readable trajectory")
    parser.add_argument(
        "--eos", default="birchmurnaghan", help="functional form (an ase.eos name)"
    )
    parser.add_argument("--natoms", type=int, help="atoms per cell, to report V0 per atom")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if not args.data.is_file():
        raise SystemExit(f"error: no such file: {args.data}")
    volumes, energies = load_points(args.data)
    if len(volumes) < 4:
        raise SystemExit(
            f"error: {len(volumes)} point(s) is not enough to fit an equation of "
            f"state; compute at least 4 (7 is typical)"
        )

    import numpy as np
    from ase.eos import EquationOfState
    from ase.units import GPa

    eos = EquationOfState(volumes, energies, eos=args.eos)
    try:
        v0, e0, b = eos.fit(warn=False)
    except Exception as e:
        raise SystemExit(
            f"error: the {args.eos} fit did not converge ({e}); plot the points — "
            f"the minimum may sit outside the scanned range"
        ) from e
    b_gpa = float(b) / GPa
    parameters = list(eos.eos_parameters)
    # ASE's analytic forms carry [E0, B, B', V0]; the polynomial forms do not.
    b_prime = float(parameters[2]) if args.eos != "p3" and len(parameters) == 4 else None
    fitted = np.array([eos.func(v, *parameters) for v in volumes])
    residual = float(np.sqrt(np.mean((fitted - np.array(energies)) ** 2)))
    residual_per_atom = residual / args.natoms if args.natoms else None
    ordered = sorted(volumes)
    volume_range = (ordered[0] / float(v0), ordered[-1] / float(v0))

    warnings: list[str] = []
    if not (ordered[1] < float(v0) < ordered[-2]):
        warnings.append(
            "the fitted V0 lies outside the inner points of the scan; re-centre the "
            "scan on it and repeat"
        )
    if residual_per_atom is not None and residual_per_atom > 1e-3:
        warnings.append(
            f"RMS residual {residual_per_atom * 1000:.2f} meV/atom exceeds 1 meV/atom; "
            "the points are noisy or the k-mesh and cutoff changed between them"
        )
    if not 1.0 <= b_gpa <= 1000.0:
        warnings.append(
            "B outside the plausible solid range (1-1000 GPa); inspect the E(V) points "
            "before reporting this number"
        )

    if args.json:
        result: dict[str, object] = {
            "v0_A3": float(v0),
            "e0_eV": float(e0),
            "b_GPa": b_gpa,
            "b_prime": b_prime,
            "rms_residual_eV": residual,
            "volume_range_of_v0": [volume_range[0], volume_range[1]],
            "eos": str(args.eos),
            "points": len(volumes),
            "warnings": warnings,
        }
        if args.natoms:
            result["v0_per_atom_A3"] = float(v0) / args.natoms
            result["rms_residual_meV_per_atom"] = residual_per_atom * 1000  # type: ignore[operator]
        print(json.dumps(result, indent=1))
        return 0
    per_atom = f" ({float(v0) / args.natoms:.4f} per atom)" if args.natoms else ""
    print(f"fit: {args.eos} through {len(volumes)} points")
    print(f"V0 = {float(v0):.4f} A^3{per_atom}")
    print(f"E0 = {float(e0):.6f} eV")
    print(f"B  = {b_gpa:.2f} GPa")
    if b_prime is not None:
        print(f"B' = {b_prime:.3f}")
    residual_text = (
        f"{residual_per_atom * 1000:.3f} meV/atom" if residual_per_atom is not None
        else f"{residual * 1000:.3f} meV per cell"
    )
    print(f"RMS residual = {residual_text}")
    print(f"scanned V/V0 = {volume_range[0]:.3f} to {volume_range[1]:.3f}")
    for warning in warnings:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
