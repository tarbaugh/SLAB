"""Work of adhesion, interface energy, and heterogeneous-nucleation potency.

    adhesion.py --e-interface -1802.10 --e-a -1204.62 --e-b -595.31 --area 187.4
    adhesion.py ... --gamma-a 1.20 --gamma-b 0.95 --json
    adhesion.py ... --gamma-a 1.20 --gamma-b 0.95 --gamma-nl 0.40 --gamma-sl 0.25

The three energies (eV) are the relaxed interface supercell and the two
isolated slabs, each in the *same* supercell footprint, engine, and
settings. ``--area`` is the interface cross-section in A^2 (one interface
in the cell). The work of adhesion is

    W_adh = (E_a + E_b - E_interface) / area

reported in eV/A^2 and J/m^2. With the references frozen at the interface
geometry the same number is the work of separation; say so with
``--frozen-references`` and the report names it accordingly.

With the two free-surface energies (``--gamma-a``, ``--gamma-b``, J/m^2,
from the surface-energy skill) the Dupre relation gives the interface
energy

    gamma_int = gamma_a + gamma_b - W_adh

which is the quantity nucleation criteria use. A solid nucleus S forming
from liquid L on substrate N wets it with

    cos(theta) = (gamma_NL - gamma_NS) / gamma_SL

where gamma_NS is the interface energy above (substrate = A, nucleus =
B), gamma_NL the substrate-liquid and gamma_SL the nucleus-liquid
interfacial free energies (``--gamma-nl``, ``--gamma-sl``, J/m^2, from
melt simulations or experiment). The classical-nucleation-theory potency

    f(theta) = (2 + cos)(1 - cos)^2 / 4

multiplies the homogeneous barrier. The vacuum work of adhesion alone
cannot give theta: W_adh/gamma_SL - 1 mixes a vacuum quantity with a
liquid one and lands above 1 for nearly every real pair.

Optional ``--e-a-free`` and ``--e-b-free`` (eV) are the slabs relaxed at
their own lattice; the report then gives each slab's strain energy per
area, which the strained reference carries and the free surfaces do not.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

_EV_A2_TO_J_M2 = 16.021766208
PLAUSIBLE_J_M2 = (0.1, 12.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--e-interface", type=float, required=True,
                        help="energy of the interface supercell (eV)")
    parser.add_argument("--e-a", type=float, required=True,
                        help="energy of the first isolated slab, the substrate (eV)")
    parser.add_argument("--e-b", type=float, required=True,
                        help="energy of the second isolated slab, the nucleus phase (eV)")
    parser.add_argument("--area", type=float, required=True,
                        help="interface cross-section area (A^2)")
    parser.add_argument("--frozen-references", action="store_true",
                        help="the reference slabs were not relaxed after separation, "
                             "so the number is a work of separation")
    parser.add_argument("--e-a-free", type=float,
                        help="energy of slab A relaxed at its own lattice (eV), "
                             "to report its strain energy per area")
    parser.add_argument("--e-b-free", type=float,
                        help="energy of slab B relaxed at its own lattice (eV)")
    parser.add_argument("--gamma-a", type=float,
                        help="surface energy of the free A surface (J/m^2)")
    parser.add_argument("--gamma-b", type=float,
                        help="surface energy of the free B surface (J/m^2)")
    parser.add_argument("--gamma-nl", type=float,
                        help="substrate-liquid interfacial free energy (J/m^2)")
    parser.add_argument("--gamma-sl", type=float,
                        help="nucleus-liquid interfacial free energy (J/m^2)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    if args.area <= 0:
        raise SystemExit("error: --area is a cross-section in A^2; it must be positive")
    for name in ("gamma_a", "gamma_b", "gamma_nl", "gamma_sl"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            flag = "--" + name.replace("_", "-")
            raise SystemExit(
                f"error: {flag} is an interfacial energy in J/m^2; it must be positive"
            )
    if (args.gamma_a is None) != (args.gamma_b is None):
        raise SystemExit("error: give both --gamma-a and --gamma-b, or neither")
    if (args.gamma_nl is None) != (args.gamma_sl is None):
        raise SystemExit("error: give both --gamma-nl and --gamma-sl, or neither")
    if args.gamma_nl is not None and args.gamma_a is None:
        raise SystemExit(
            "error: the wetting angle needs the interface energy, so --gamma-nl and "
            "--gamma-sl also need --gamma-a and --gamma-b"
        )

    w_ev_a2 = (args.e_a + args.e_b - args.e_interface) / args.area
    w_j_m2 = w_ev_a2 * _EV_A2_TO_J_M2
    quantity = "work_of_separation" if args.frozen_references else "work_of_adhesion"
    result: dict[str, Any] = {
        "quantity": quantity,
        "w_adh_eV_A2": w_ev_a2,
        "w_adh_J_m2": w_j_m2,
        "warnings": [],
    }
    if w_ev_a2 <= 0:
        result["warnings"].append(
            "non-positive work of adhesion: the slabs do not bind; check that "
            "the interface is relaxed and that all three energies share one "
            "engine, one settings expansion, and one supercell footprint"
        )
    elif not PLAUSIBLE_J_M2[0] <= w_j_m2 <= PLAUSIBLE_J_M2[1]:
        result["warnings"].append(
            f"W of {w_j_m2:.2f} J/m^2 is outside the range of known interfaces "
            f"({PLAUSIBLE_J_M2[0]}-{PLAUSIBLE_J_M2[1]} J/m^2: metal/metal 0.5-3, "
            f"O-terminated oxides up to about 11); check the area and the energies"
        )
    for label, free, strained in (("a", args.e_a_free, args.e_a), ("b", args.e_b_free, args.e_b)):
        if free is not None:
            strain = (strained - free) / args.area * _EV_A2_TO_J_M2
            result[f"strain_energy_{label}_J_m2"] = strain
            if strain < 0:
                result["warnings"].append(
                    f"slab {label} is lower in energy strained than free; the free "
                    f"reference is not relaxed"
                )
    if args.gamma_a is not None:
        gamma_int = args.gamma_a + args.gamma_b - w_j_m2
        result["gamma_int_J_m2"] = gamma_int
        if gamma_int < 0:
            result["warnings"].append(
                "negative interface energy: the interface is lower in energy than the "
                "bulk it separates, which means W_adh or a surface energy is off"
            )
    if args.gamma_nl is not None:
        cos_theta = (args.gamma_nl - result["gamma_int_J_m2"]) / args.gamma_sl
        if cos_theta > 1.0:
            result["warnings"].append(
                "gamma_NS is below gamma_NL - gamma_SL: complete wetting; the contact "
                "angle is 0 and the heterogeneous barrier vanishes (f = 0)"
            )
            cos_theta = 1.0
        elif cos_theta < -1.0:
            result["warnings"].append(
                "gamma_NS exceeds gamma_NL + gamma_SL: no wetting; the substrate does "
                "not catalyze nucleation (f = 1)"
            )
            cos_theta = -1.0
        result["cos_theta"] = cos_theta
        result["theta_deg"] = math.degrees(math.acos(cos_theta))
        result["f_het"] = (2.0 + cos_theta) * (1.0 - cos_theta) ** 2 / 4.0

    if args.json:
        print(json.dumps(result, indent=1))
        return 0
    name = "W_sep" if args.frozen_references else "W_adh"
    print(f"{name} = {w_ev_a2:.6f} eV/A^2 = {w_j_m2:.4f} J/m^2")
    for label in ("a", "b"):
        key = f"strain_energy_{label}_J_m2"
        if key in result:
            print(f"strain energy of slab {label}: {result[key]:.4f} J/m^2")
    if "gamma_int_J_m2" in result:
        print(f"gamma_int = gamma_a + gamma_b - W = {result['gamma_int_J_m2']:.4f} J/m^2")
    if "f_het" in result:
        print(
            f"cos(theta) = (gamma_NL - gamma_NS)/gamma_SL = {result['cos_theta']:+.4f} -> "
            f"theta = {result['theta_deg']:.1f} deg, f(theta) = {result['f_het']:.4f}"
        )
    for warning in result["warnings"]:
        print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
