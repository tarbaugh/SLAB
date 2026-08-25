"""Fit a Birch-Murnaghan equation of state to energy-volume data.

Input is either a JSON file holding a list of ``{"volume", "energy"}``
points in Å^3 and eV (the shape ``assets/eos_scan.py`` writes), or any
ASE-readable trajectory whose images carry energies.

    fit_eos.py eos.json
    fit_eos.py scan.traj --json --natoms 4

Output: V0 (Å^3), E0 (eV), B (GPa), and with ``--natoms`` the volume per
atom. Exits nonzero with an actionable message on bad input or a fit that
does not converge.
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

    from ase.eos import EquationOfState
    from ase.units import GPa

    try:
        v0, e0, b = EquationOfState(volumes, energies, eos=args.eos).fit()
    except Exception as e:
        raise SystemExit(
            f"error: the {args.eos} fit did not converge ({e}); plot the points — "
            f"the minimum may sit outside the scanned range"
        ) from e
    b_gpa = float(b) / GPa

    if args.json:
        result: dict[str, float | int | str] = {
            "v0_A3": float(v0),
            "e0_eV": float(e0),
            "b_GPa": b_gpa,
            "eos": str(args.eos),
            "points": len(volumes),
        }
        if args.natoms:
            result["v0_per_atom_A3"] = float(v0) / args.natoms
        print(json.dumps(result, indent=1))
        return 0
    per_atom = f" ({float(v0) / args.natoms:.4f} per atom)" if args.natoms else ""
    print(f"fit: {args.eos} through {len(volumes)} points")
    print(f"V0 = {float(v0):.4f} A^3{per_atom}")
    print(f"E0 = {float(e0):.6f} eV")
    print(f"B  = {b_gpa:.2f} GPa")
    if not 1.0 <= b_gpa <= 1000.0:
        print(
            "warning: B outside the plausible solid range (1-1000 GPa); "
            "inspect the E(V) points before reporting this number"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
