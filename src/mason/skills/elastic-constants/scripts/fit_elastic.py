"""Elastic constants from energy-strain scans: C_ij, VRH moduli, E and nu.

    fit_elastic.py elastic.json
    fit_elastic.py elastic.json --json

The input records one energy ladder per deformation mode::

    {"symmetry": "cubic", "v0": 45.9, "modes": {"e1": [{"delta": -0.01,
     "energy": -3.61}, ...], "e4": [...], "e12": [...]}}

``v0`` is the reference cell volume in A^3 and energies are in eV. The
mode names are Voigt strain patterns: ``e1``..``e6`` strain one component
(``e4``..``e6`` are engineering shears), ``e12``/``e13``/``e23`` strain two
normal components together. Symmetries and the modes they need:

    isotropic     e1 e4
    cubic         e1 e4 e12
    orthorhombic  e1 e2 e3 e4 e5 e6 e12 e13 e23

Each ladder is fitted with a quadratic; the curvatures combine into the
independent C_ij, and the 6x6 matrix gives Voigt-Reuss-Hill bulk and shear
moduli, the Young's modulus, and the Poisson ratio.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_EV_A3_TO_GPA = 160.21766208

REQUIRED_MODES = {
    "isotropic": ("e1", "e4"),
    "cubic": ("e1", "e4", "e12"),
    "orthorhombic": ("e1", "e2", "e3", "e4", "e5", "e6", "e12", "e13", "e23"),
}


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"error: no such file: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"error: {path} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise SystemExit('error: expected {"symmetry": ..., "v0": ..., "modes": {...}}')
    symmetry = data.get("symmetry")
    if symmetry not in REQUIRED_MODES:
        raise SystemExit(
            f"error: symmetry must be one of {', '.join(sorted(REQUIRED_MODES))}; "
            f"got {symmetry!r}"
        )
    try:
        v0 = float(data["v0"])
    except (KeyError, TypeError, ValueError) as e:
        raise SystemExit(f'error: "v0" must be the reference volume in A^3: {e}') from e
    if v0 <= 0:
        raise SystemExit('error: "v0" must be a positive volume in A^3')
    modes = data.get("modes")
    if not isinstance(modes, dict):
        raise SystemExit('error: "modes" must map mode names to [{"delta", "energy"}] ladders')
    missing = [m for m in REQUIRED_MODES[symmetry] if m not in modes]
    if missing:
        raise SystemExit(
            f"error: symmetry {symmetry!r} needs modes {', '.join(missing)} "
            f"that the file does not record"
        )
    return data


def _curvature(mode: str, rows: Any, v0: float, warnings: list[str]) -> float:
    """The quadratic curvature of one ladder, as an eV/A^3 modulus combination
    converted to GPa. Appends fit-quality warnings in place."""
    if not isinstance(rows, list) or len(rows) < 4:
        raise SystemExit(
            f"error: mode {mode!r} has {len(rows) if isinstance(rows, list) else 'no'} "
            f"point(s); a quadratic fit worth trusting needs at least 4"
        )
    try:
        deltas = np.array([float(r["delta"]) for r in rows])
        energies = np.array([float(r["energy"]) for r in rows])
    except (KeyError, TypeError, ValueError) as e:
        raise SystemExit(
            f'error: mode {mode!r} rows need numeric "delta" and "energy" fields: {e}'
        ) from e
    span = float(np.abs(deltas).max())
    if span == 0.0:
        raise SystemExit(f"error: mode {mode!r} has zero strain span")
    a2, a1, _ = np.polyfit(deltas, energies, 2)
    if a2 <= 0:
        warnings.append(
            f"mode {mode} has non-convex energy (curvature <= 0): the structure "
            f"is unstable along this strain, or the ladder is noise"
        )
    if abs(a1) > 0.2 * abs(a2) * span:
        warnings.append(
            f"mode {mode} has a large linear term: the reference structure is "
            f"not at its energy minimum along this strain; relax it first"
        )
    return float(2.0 * a2 / v0) * _EV_A3_TO_GPA


def _assemble(symmetry: str, c: dict[str, float]) -> dict[str, float]:
    """The independent C_ij (GPa) from the fitted mode curvatures."""
    if symmetry == "isotropic":
        c11, c44 = c["e1"], c["e4"]
        return {"C11": c11, "C12": c11 - 2.0 * c44, "C44": c44}
    if symmetry == "cubic":
        c11, c44 = c["e1"], c["e4"]
        return {"C11": c11, "C12": (c["e12"] - 2.0 * c11) / 2.0, "C44": c44}
    return {
        "C11": c["e1"], "C22": c["e2"], "C33": c["e3"],
        "C44": c["e4"], "C55": c["e5"], "C66": c["e6"],
        "C12": (c["e12"] - c["e1"] - c["e2"]) / 2.0,
        "C13": (c["e13"] - c["e1"] - c["e3"]) / 2.0,
        "C23": (c["e23"] - c["e2"] - c["e3"]) / 2.0,
    }


def _matrix(symmetry: str, cij: dict[str, float]) -> np.ndarray:
    """The full 6x6 stiffness matrix in GPa."""
    if symmetry == "orthorhombic":
        c11, c22, c33 = cij["C11"], cij["C22"], cij["C33"]
        c44, c55, c66 = cij["C44"], cij["C55"], cij["C66"]
        c12, c13, c23 = cij["C12"], cij["C13"], cij["C23"]
    else:
        c11 = c22 = c33 = cij["C11"]
        c44 = c55 = c66 = cij["C44"]
        c12 = c13 = c23 = cij["C12"]
    matrix = np.zeros((6, 6))
    matrix[0, 0], matrix[1, 1], matrix[2, 2] = c11, c22, c33
    matrix[3, 3], matrix[4, 4], matrix[5, 5] = c44, c55, c66
    matrix[0, 1] = matrix[1, 0] = c12
    matrix[0, 2] = matrix[2, 0] = c13
    matrix[1, 2] = matrix[2, 1] = c23
    return matrix


def _vrh(matrix: np.ndarray, warnings: list[str]) -> dict[str, Any]:
    """Voigt, Reuss, and Hill bulk/shear moduli, E, and nu from the matrix."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues.min() <= 0:
        warnings.append(
            "the stiffness matrix is not positive definite: the structure is "
            "mechanically unstable, or one of the ladders is bad; the averaged "
            "moduli below are not meaningful"
        )
    c = matrix
    b_voigt = (c[0, 0] + c[1, 1] + c[2, 2] + 2 * (c[0, 1] + c[0, 2] + c[1, 2])) / 9.0
    g_voigt = (
        c[0, 0] + c[1, 1] + c[2, 2] - (c[0, 1] + c[0, 2] + c[1, 2])
        + 3 * (c[3, 3] + c[4, 4] + c[5, 5])
    ) / 15.0
    result: dict[str, Any] = {"b_voigt_GPa": float(b_voigt), "g_voigt_GPa": float(g_voigt)}
    try:
        s = np.linalg.inv(c)
    except np.linalg.LinAlgError:
        warnings.append("the stiffness matrix is singular; Reuss and Hill are unavailable")
        b_hill, g_hill = float(b_voigt), float(g_voigt)
    else:
        b_reuss = 1.0 / (s[0, 0] + s[1, 1] + s[2, 2] + 2 * (s[0, 1] + s[0, 2] + s[1, 2]))
        g_reuss = 15.0 / (
            4 * (s[0, 0] + s[1, 1] + s[2, 2]) - 4 * (s[0, 1] + s[0, 2] + s[1, 2])
            + 3 * (s[3, 3] + s[4, 4] + s[5, 5])
        )
        result["b_reuss_GPa"] = float(b_reuss)
        result["g_reuss_GPa"] = float(g_reuss)
        b_hill = float((b_voigt + b_reuss) / 2.0)
        g_hill = float((g_voigt + g_reuss) / 2.0)
    result["b_hill_GPa"] = b_hill
    result["g_hill_GPa"] = g_hill
    result["youngs_GPa"] = 9.0 * b_hill * g_hill / (3.0 * b_hill + g_hill)
    result["poisson"] = (3.0 * b_hill - 2.0 * g_hill) / (2.0 * (3.0 * b_hill + g_hill))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data", type=Path, help="elastic.json from the strain scan")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    data = _load(args.data)
    symmetry: str = data["symmetry"]
    v0 = float(data["v0"])
    warnings: list[str] = []
    curvatures = {
        mode: _curvature(mode, data["modes"][mode], v0, warnings)
        for mode in REQUIRED_MODES[symmetry]
    }
    cij = _assemble(symmetry, curvatures)
    moduli = _vrh(_matrix(symmetry, cij), warnings)

    if args.json:
        print(
            json.dumps(
                {
                    "symmetry": symmetry,
                    "curvatures_GPa": curvatures,
                    "cij_GPa": cij,
                    **moduli,
                    "warnings": warnings,
                },
                indent=1,
            )
        )
        return 0
    print(f"symmetry: {symmetry}, reference volume {v0:.3f} A^3")
    for name, value in cij.items():
        print(f"  {name} = {value:8.2f} GPa")
    print(
        f"bulk modulus  B: Voigt {moduli['b_voigt_GPa']:.2f}, "
        f"Hill {moduli['b_hill_GPa']:.2f} GPa"
    )
    print(
        f"shear modulus G: Voigt {moduli['g_voigt_GPa']:.2f}, "
        f"Hill {moduli['g_hill_GPa']:.2f} GPa"
    )
    print(
        f"E = {moduli['youngs_GPa']:.2f} GPa, nu = {moduli['poisson']:.4f} "
        f"(isotropic VRH averages)"
    )
    for warning in warnings:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
