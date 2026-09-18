---
name: density-of-states
description: Compute a density of states with Quantum ESPRESSO, with the
  projection onto each element and angular momentum, and read the band
  edges and the states at the Fermi level from one traced task. Use when
  asked for a DOS, a PDOS, the states at the Fermi level, or which
  orbitals make up a band.
license: MIT
metadata:
  mason-agents: "dft-expert"
---
# Density of states

`density_of_states` in `foundation.tasks` runs three or four executables
in one scratch directory. The first is an SCF on your k-point mesh. The
second is an NSCF on a denser mesh, and it reads the SCF's charge
density. The third is `dos.x`, which turns the NSCF eigenvalues into a
curve. With `projected=True` a fourth, `projwfc.x`, gives one curve per
element and angular momentum. The task returns the curve, the verdict,
and the engine identity, and it keeps every output as an artifact of the
run.

## 1. When to compute a density of states

- Compute a DOS to answer how many states sit at an energy, not where
  they sit in the Brillouin zone. For the dispersion, use the
  band-structure skill.
- Compute a projected DOS to say which element and which orbital make up
  a band. The d states of a transition metal near the Fermi level, or the
  oxygen p states at the top of an oxide valence band, are PDOS
  questions.
- The states at the Fermi level tell you about conduction, magnetism, and
  the electronic heat capacity of a metal. Report `dos_at_fermi` with the
  broadening that produced it.
- Relax the structure first, under the same functional and
  pseudopotential family.

## 2. Set the denser mesh and the broadening

- A DOS is an integral over the Brillouin zone, so it needs a denser mesh
  than a total energy does. Without `dos_kpts` or `dos_kspacing` the task
  doubles the SCF mesh in each direction, which is eight times the
  k-points. The NSCF is cheap because the charge density is fixed.
- Raise the mesh until the shape of the curve stops changing. A sharp
  peak and a small `dos_at_fermi` both need more k-points than a smooth
  curve does.
- The broadening is a Gaussian of width `degauss`, in Ry. It defaults to
  the SCF's own `degauss`, which a protocol sets for the total energy and
  which is often too wide for a DOS. 0.02 Ry is 0.27 eV, and a Gaussian
  that wide fills a half-eV gap with states.
- Set `degauss=` smaller when you want the gap to show, and raise the
  mesh with it. Broadening and mesh trade off: a narrow Gaussian on a
  coarse mesh gives a spiky curve that is the mesh, not the material.
- `delta_e=` sets the grid step, in eV, and defaults to 0.01. `emin=` and
  `emax=` set the window, which otherwise covers every eigenvalue.
- The task refuses `nspin=2`, `noncolin`, and `lspinorb`. It reads one
  spin channel, so do not use it for a magnetic system yet.
- The task runs on the cell you give it. The curve is per cell, so a
  supercell gives a multiple of the primitive cell's curve.

## 3. Run it in a workflow script

Call the task in a workflow script and launch the script, so the run is
traced and its checks decide the run's state. This script ran with
Quantum ESPRESSO 7.5, and the numbers below are what it printed:

```python
from ase.build import bulk
from foundation import check, within_bounds
from foundation.tasks import density_of_states
from slab.protocols import qe_protocol_options

atoms = bulk("Si", "diamond", a=5.43)
options = qe_protocol_options(atoms, protocol="balanced")
dos, info = density_of_states(
    atoms, calculator_options=options, dos_kpts=[8, 8, 8],
    delta_e=0.05, degauss=0.005, projected=True, label="si",
)
print(f"gap {info['gap']} eV ({info['gap_kind']}), is_metal={info['is_metal']}")
print(f"dos at the Fermi level {info['dos_at_fermi']:.3g} states/eV/cell")
print(f"groups {info['projection_groups']}, mesh {info['dos_kpts']}")

@check
def the_fermi_level_sits_in_the_gap():
    return within_bounds(info["dos_at_fermi"], lo=0.0, hi=1e-3, label="dos at E_F")
```

For a metal, check the other way round:

```python
@check
def aluminium_conducts():
    return info["is_metal"] is True and info["dos_at_fermi"] > 0.1
```

## 4. Read the verdict

`info` carries the verdict:

| Key | Meaning |
|---|---|
| `dos_at_fermi` | The broadened curve at the Fermi level, in states/eV/cell |
| `is_metal` | True when a band crosses the Fermi level |
| `gap`, `direct_gap`, `gap_kind` | The gap from the NSCF eigenvalues, in eV |
| `vbm`, `cbm` | The band edges in eV, on the NSCF's energy scale |
| `fermi` | The NSCF Fermi level in eV, on its denser mesh |
| `degauss_ry`, `delta_e` | The broadening in Ry and the grid step in eV |
| `scf_kpts`, `dos_kpts` | The two meshes |
| `projection_groups` | The group names, such as `Si-s` and `Si-p` |
| `artifacts` | The kept files, named by the label |

The verdict comes from the eigenvalues and never from the curve. A
Gaussian wider than the gap puts states inside it, so a nonzero
`dos_at_fermi` does not make a metal, and `is_metal` says which it is.

A group is one element and one angular momentum, summed over the atoms
of that element and over the shells of that l. `projected_dos` in the
result file holds one curve per group, on the same grid as the total.

The run keeps six artifacts with `projected=True`: `{label}-scf.pwo`,
`{label}-nscf.pwo`, `{label}-dos.dat`, `{label}-dos.out`,
`{label}-projwfc.out`, and `{label}-dos.json`. The json is the result
file, and it holds the grid, the curves, and the summary. Read the json
with `read_artifact`, not the `.dat`. When the arrays hold more than 1000
numbers, the return value leaves them out and names the file in
`dos_in`. A failed step keeps its files as `{label}-scf-failed.*`,
`{label}-nscf-failed.*`, `{label}-dos-failed.out`, or
`{label}-projwfc-failed.out`, and the failure notes name the step.

`scripts/dos_table.py` reads the json. It prints the summary as JSON,
writes a table with `--dat`, and draws the curves with `--png` when
matplotlib is installed. Without matplotlib, `--png` exits 2 and names
the package. The table has the energy relative to the valence band
maximum, or to the Fermi level for a metal, then the total, then one
column per group. These are the first lines of
`dos_table.py si-dos.json --dat si-dos.dat` for the Si run above:

```
# density of states, 450 rows, step 0.05 eV, Gaussian broadening 0.005 Ry
# energies in eV relative to the valence band maximum (6.1642 eV)
# dos in states/eV/cell; Fermi level at 0.2717 eV
# columns: E, total dos, Si-p, Si-s
-12.197200 0.000004 0.000000 0.000004
```

`--png` draws an energy window and not the whole grid, because the grid
covers every band of the run and the part near the Fermi level is what
most questions need. The default window runs from 8 eV below the
reference to 8 eV above the conduction band minimum, or above the Fermi
level for a metal. The vertical axis fits the curves inside the window,
and the report gives the window in `png_window_ev`. Set your own window
with `--window LO HI`, in eV relative to the reference, when the
question needs it. `--whole-grid` draws the whole grid. `--dat` always
holds the whole grid.

## 5. Report it honestly

- Report the broadening and both meshes with every number. A DOS at the
  Fermi level without its Gaussian width is not a number anyone can
  reproduce.
- A PBE or PBEsol gap is a lower bound, exactly as it is for a band
  structure. The Si run above gives 0.5061 eV, and the measured gap is
  1.17 eV.
- The projections are onto the pseudo-atomic orbitals of the
  pseudopotentials. That basis is not complete and not orthogonal between
  atoms, so the group curves sum to near the total and not to it, and a
  band's weights sum to near one and not to one. Report them as weights,
  never as charges.
- A group holds every shell of one l. A pseudopotential with a semicore
  shell puts the semicore and the valence states in one curve, so read
  the `projwfc.x` state list when the split matters.
- Cite the run id. The number rests on the run that kept the json.

## When not to use this

- Do not use it for a magnetic system or for spin-orbit splitting. The
  task refuses both.
- Do not use it for the dispersion or for the shape of a band. Use the
  band-structure skill, which also gives the projected weights per band
  with `projected=True`.
- Do not read `dos_at_fermi` as a verdict on its own. Read `is_metal`.
- Do not use it for a gap that must match experiment. A semilocal
  functional does not give that gap.
