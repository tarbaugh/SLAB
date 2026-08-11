"""SLAB end-to-end demo: relax 5 perturbed supercell variants, keep only the best.

Five perturbed variants of a supercell are relaxed, each in its own run with a
stated intent. Verification checks (force convergence, finite energy, declared
units) gate each run to `verified`. Nothing here is "production" — that
decision happens *after* the results exist, from the CLI:

    python examples/demo.py                 # MACE + Si (needs: pip install 'slab[mace]')
    python examples/demo.py --engine emt    # EMT + Cu, no extra install, seconds

    slab list
    slab show <best-run-id>
    slab promote <best-run-id> --reason "lowest energy of 5 variants"
    slab expire --older-than 0d
    slab gc
    slab list --state expired               # hash-only skeletons remain queryable

This script manages its own runs (five of them), so it is executed with plain
``python``, not ``slab run`` — SLAB is a library first.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import ase.io
from ase.build import bulk

from slab import Workspace, check, converged, finite, units
from slab.tasks import relax


def build_base(engine: str):  # returns ase.Atoms
    if engine == "emt":
        # EMT does not parametrize Si; Cu keeps the no-extra-install path honest.
        return bulk("Cu", "fcc", a=3.58) * (2, 2, 2)
    return bulk("Si", "diamond", a=5.43) * (2, 2, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default="mace", choices=["mace", "emt"])
    parser.add_argument("--workspace", default=".slab")
    parser.add_argument("--fmax", type=float, default=0.05)
    args = parser.parse_args()

    base = build_base(args.engine)
    symbol = base.get_chemical_symbols()[0]
    print(
        f"workspace: {args.workspace}   engine: {args.engine}   "
        f"system: {symbol} x {len(base)} atoms"
    )

    results: list[tuple[str, dict]] = []
    with Workspace(args.workspace) as ws:
        for i in range(5):
            atoms = base.copy()
            stdev = 0.02 + 0.02 * i
            atoms.rattle(stdev=stdev, seed=100 + i)
            intent = (
                f"variant {i}: rattle stdev={stdev:.2f} A - "
                f"hunting the lowest-energy relaxation of the batch"
            )
            with ws.start_run(name=f"{symbol.lower()}-relax-{i}", intent=intent) as run:
                relaxed, info = relax(
                    atoms, engine=args.engine, fmax=args.fmax, label=f"variant-{i}"
                )

                @check
                def forces_converged(info=info, fmax=args.fmax):
                    return converged(info["fmax"], below=fmax, label="fmax")

                @check
                def energy_is_finite(info=info):
                    return finite(info["energy"], label="energy")

                @check
                def energy_unit_declared(info=info):
                    return units(info["energy_unit"], "eV")

                # Declared terminal artifacts: what promotion will preserve.
                with tempfile.TemporaryDirectory() as tmp:
                    xyz = Path(tmp) / "relaxed.xyz"
                    ase.io.write(xyz, relaxed)
                    run.keep("relaxed.xyz", xyz)
                run.keep(
                    "result",
                    {
                        "energy_eV": info["energy"],
                        "fmax": info["fmax"],
                        "steps": info["steps"],
                        "n_atoms": info["n_atoms"],
                        "engine": args.engine,
                    },
                )

            final = ws.runs.get(run.id)
            results.append((run.id, info))
            print(
                f"  run {run.id[:10]}  E = {info['energy']:.6f} eV  "
                f"fmax = {info['fmax']:.4f}  steps = {info['steps']:>2}  "
                f"-> {final.state.value}"
            )

    best_id, best_info = min(results, key=lambda r: r[1]["energy"])
    print()
    print(f"lowest energy: run {best_id[:10]}  (E = {best_info['energy']:.6f} eV)")
    print("decide what deserves permanence, then clean up:")
    print(f"  slab promote {best_id[:10]} --reason 'lowest energy of 5 variants'")
    print("  slab expire --older-than 0d")
    print("  slab gc")


if __name__ == "__main__":
    main()
