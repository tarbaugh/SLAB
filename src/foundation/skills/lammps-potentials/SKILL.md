---
name: lammps-potentials
description: Choose the LAMMPS pair_style and pair_coeff lines for a
  potential file (EAM funcfl, setfl, eam/fs, ACE, GRACE, MEAM), check that
  the potential loads and gives physical energies before any production
  run, and diagnose "Not a valid floating-point number" and similar
  potential-file errors. Use when a LAMMPS run needs a classical or
  machine-learned potential, or when LAMMPS rejects a potential file.
license: MIT
metadata:
  mason-agents: "md-expert"
---
# LAMMPS potential files

A potential file's format decides the `pair_style`. The file name usually
says which format it is, and the header confirms it. Read the header
before you write the input, and never edit a potential file to make a
wrong `pair_style` accept it.

## 1. Identify the format

Run the bundled script on the file:

```bash
python scripts/pair_style_for.py /path/to/potential
```

It reads the header, names the format, and prints the `pair_style` and
`pair_coeff` lines to paste into the input. Add `--elements W` when the
file carries no element symbols (ACE and GRACE files) or when you need a
different type-to-element mapping. `--json` gives the same facts as a
machine-readable object. Exit code 2 means the file is none of the
formats below; report that instead of guessing.

| File form | Header signature | `pair_style` | `pair_coeff` |
| --- | --- | --- | --- |
| funcfl (`.eam`) | line 2 `Z mass a lattice`, line 3 `nrho drho nr dr cutoff` | `eam` | `* * FILE` (one element) |
| setfl (`.eam.alloy`, "DYNAMO 86 setfl") | 3 comment lines, then `N El1 El2 ...`, then the grid line | `eam/alloy` | `* * FILE El1 El2 ...` |
| Finnis-Sinclair setfl (`.eam.fs`) | as setfl | `eam/fs` | `* * FILE El1 El2 ...` |
| ACE / PACE (`.yace`, `.ace`) | YAML-like, no elements in the header | `pace` | `* * FILE El1 El2 ...` |
| GRACE checkpoint (`.yaml`, or a saved-model directory) | as the training run wrote it | `grace` | `* * FILE El1 El2 ...` |
| MEAM (`library.meam` + `El.meam`) | two files | `meam` | `* * LIBRARY El... PARAMS El...` |

The element list on a `pair_coeff` line maps LAMMPS atom types, in order,
to elements. Type 1 is the first symbol.

## 2. The most common mistake

A setfl file (`.eam.alloy`) under `pair_style eam` fails with
`Not a valid floating-point number: 'W'`, because the funcfl reader
expects a mass where setfl has the element line. The fix is
`pair_style eam/alloy` with the element symbol on the `pair_coeff` line.
The file is fine. One campaign lost seventy minutes and most of its
reasoning budget rewriting the arrays of a correct file instead of
changing that one keyword. When LAMMPS rejects a potential file, run the
script first, then change the input, and only then question the file.

## 3. Smoke test before production

A potential that loads can still be wrong for the system (wrong units,
wrong element order, a file for another phase). Test on a small cell
before any run that costs compute:

1. Build the conventional cell of the element at its experimental
   lattice constant (2 atoms for bcc, 4 for fcc).
2. Run one `run 0` and print `pe`. The energy per atom must be within
   about 20 % of the element's cohesive energy (W: -8.9 eV/atom, Cu:
   -3.5 eV/atom, Al: -3.4 eV/atom). A positive value or a value of
   thousands of eV means a misread file.
3. Run 500 steps of NVT at 300 K on a 3x3x3 supercell with a 1 fs
   timestep. The temperature must stay near 300 K and the energy must
   not drift by more than a few meV/atom.

Record the smoke test's log as an artifact of the run that uses the
potential, and name the potential file and its `pair_style` in the run's
intent. A potential used outside its fitted domain is fiction with good
statistics, so state the fitting domain from the file's header citation
when you report.
