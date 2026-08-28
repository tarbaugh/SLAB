"""Work of adhesion and heterogeneous-nucleation potency from slab energies.

    adhesion.py --e-interface -1802.10 --e-a -1204.62 --e-b -595.31 --area 187.4
    adhesion.py ... --gamma 0.062 --json

The three energies (eV) are the relaxed interface supercell and the two
isolated slabs, each in the *same* supercell footprint, engine, and
settings. ``--area`` is the interface cross-section in A^2 (one interface
in the cell). The work of adhesion is

    W_adh = (E_a + E_b - E_interface) / area

reported in eV/A^2 and J/m^2. With ``--gamma`` (the nucleating phase's
interfacial free energy against the parent phase, J/m^2), the Young-Dupre
relation cos(theta) = W_adh/gamma - 1 gives the wetting angle and the
classical-nucleation-theory potency factor

    f(theta) = (2 + cos)(1 - cos)^2 / 4

which multiplies the homogeneous barrier.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

_EV_A2_TO_J_M2 = 16.021766208


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--e-interface", type=float, required=True,
                        help="energy of the interface supercell (eV)")
    parser.add_argument("--e-a", type=float, required=True,
                        help="energy of the first isolated slab (eV)")
    parser.add_argument("--e-b", type=float, required=True,
                        help="energy of the second isolated slab (eV)")
    parser.add_argument("--area", type=float, required=True,
                        help="interface cross-section area (A^2)")
    parser.add_argument("--gamma", type=float,
                        help="interfacial free energy of the nucleating phase (J/m^2), "
                             "to also report the wetting angle and f(theta)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if args.area <= 0:
        raise SystemExit("error: --area is a cross-section in A^2; it must be positive")
    if args.gamma is not None and args.gamma <= 0:
        raise SystemExit("error: --gamma is an interfacial energy in J/m^2; it must be positive")

    w_ev_a2 = (args.e_a + args.e_b - args.e_interface) / args.area
    result: dict[str, Any] = {
        "w_adh_eV_A2": w_ev_a2,
        "w_adh_J_m2": w_ev_a2 * _EV_A2_TO_J_M2,
        "warnings": [],
    }
    if w_ev_a2 <= 0:
        result["warnings"].append(
            "non-positive work of adhesion: the slabs do not bind; check that "
            "the interface is relaxed and that all three energies share one "
            "engine, one settings expansion, and one supercell footprint"
        )
    if args.gamma is not None:
        cos_theta = result["w_adh_J_m2"] / args.gamma - 1.0
        if cos_theta > 1.0:
            result["warnings"].append(
                "W_adh exceeds 2*gamma: complete wetting; the contact angle is 0 "
                "and the heterogeneous barrier vanishes (f = 0)"
            )
            cos_theta = 1.0
        elif cos_theta < -1.0:
            result["warnings"].append(
                "W_adh is below 0 relative to gamma: no wetting; the substrate "
                "does not catalyze nucleation (f = 1)"
            )
            cos_theta = -1.0
        result["cos_theta"] = cos_theta
        result["theta_deg"] = math.degrees(math.acos(cos_theta))
        result["f_het"] = (2.0 + cos_theta) * (1.0 - cos_theta) ** 2 / 4.0

    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    print(
        f"W_adh = {result['w_adh_eV_A2']:.6f} eV/A^2 = {result['w_adh_J_m2']:.4f} J/m^2"
    )
    if "f_het" in result:
        print(
            f"cos(theta) = {result['cos_theta']:+.4f} -> theta = "
            f"{result['theta_deg']:.1f} deg, f(theta) = {result['f_het']:.4f}"
        )
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
