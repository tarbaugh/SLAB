---
name: band-structure
description: Compute a Kohn-Sham band structure with Quantum ESPRESSO
  along a high-symmetry path, and read the gap verdict (metal, direct or
  indirect gap, band edges) from one traced task. Use when asked for a
  band structure, a band gap, or whether a material is a metal.
license: MIT
metadata:
  mason-agents: "dft-expert"
---
# Band structure

`band_structure` in `foundation.tasks` runs two pw.x executions in one
scratch directory. The first is an SCF on your k-point mesh. The second
is `calculation='bands'` on a high-symmetry path, and it reads the SCF's
charge density. The task returns the bands, the gap verdict, and the
engine identity, and it keeps the outputs as artifacts of the run.

## 1. Prepare the structure

- Relax the structure first under the same functional and pseudopotential
  family. A band structure on an unrelaxed cell reports the gap of a
  strained crystal.
- Use the primitive cell. A supercell folds its bands back into a smaller
  zone, and the diagram then shows many more bands than the crystal has.
  `ase.build.bulk` gives the primitive cell for fcc, bcc, diamond,
  zincblende, and rocksalt.
- Pass a cell that is periodic in all three directions. The task refuses
  a slab or a molecule.

## 2. Set up the SCF

- Expand a protocol with `qe_protocol_options(atoms, protocol=...)`. It
  supplies the cutoffs, the k-point mesh, and the smearing. The
  convergence-study skill checks the mesh and the cutoff.
- Keep the protocol's smearing, also for an insulator. The verdict comes
  from band crossings, not from occupations, so the smearing does not
  change it.
- Leave `calculation` out of the options, or set it to `"scf"`. The task
  runs its own SCF and its own bands step, and it refuses any other
  value.
- The task refuses `nspin=2`, `noncolin`, and `lspinorb`. It reads one
  spin channel, so do not use it for a magnetic system or for spin-orbit
  splitting yet.

## 3. Run it in a workflow script

Call the task in a workflow script and launch the script, so the run is
traced and its checks decide the run's state. This script ran with
Quantum ESPRESSO 7.5 and the SSSP PBEsol efficiency family, and the run
it made reached verified:

```python
from ase.build import bulk
from foundation import check, within_bounds
from foundation.tasks import band_structure
from slab.protocols import qe_protocol_options

atoms = bulk("Si", "diamond", a=5.43)
options = qe_protocol_options(atoms, protocol="balanced")
bands, info = band_structure(atoms, calculator_options=options, npoints=60, label="si")
print(f"gap {info['gap']} eV ({info['gap_kind']}), is_metal={info['is_metal']}")
print(f"VBM {info['vbm']} eV at {info['vbm_at']['label']}, "
      f"CBM {info['cbm']} eV at k={info['cbm_at']['k']} near {info['cbm_at']['label']}")
print(f"path {info['path']} ({info['lattice']}), {info['npoints']} k-points, {info['n_bands']} bands")
print("artifacts:", info["artifacts"])

@check
def si_has_a_semilocal_gap():
    return within_bounds(info["gap"], lo=0.3, hi=0.8, label="gap (eV)")
```

For a metal, check the verdict itself:

```python
@check
def aluminium_is_a_metal():
    return info["is_metal"] is True
```

- `path=` takes a path in the lattice's own labels, such as `"GXL"`. A
  comma starts a new segment. With no `path=` the task takes ASE's
  default path for the lattice, and an unknown label is refused with the
  lattice's labels.
- `npoints=` sets the number of k-points on the path. Without it,
  `density=` sets the points per inverse angstrom (20 by default).
- `nbands=` sets the number of bands. The default holds the occupied
  bands plus 20 percent, at least 4 more. A value below the occupied
  bands is refused.

## 4. Read the verdict

`info` carries the verdict:

| Key | Meaning |
|---|---|
| `is_metal` | True when a band crosses the SCF Fermi level |
| `gap`, `direct_gap` | The gap and the smallest gap at one k-point, in eV |
| `gap_kind` | `"direct"`, `"indirect"`, or None for a metal |
| `vbm`, `cbm` | The band edges in eV, on the SCF's energy scale |
| `vbm_at`, `cbm_at` | The fractional k of each edge, the nearest special point, and its distance |
| `fermi` | The SCF Fermi level in eV |
| `artifacts` | The kept files, named by the label |

A band is valence when all its energies lie below the SCF Fermi level,
and conduction when all lie above it. For a metal every gap field is
None. When the listing holds no conduction band the gap fields are also
None, and `note` says to raise `nbands`.

The run keeps three artifacts. `{label}-scf.pwo` and
`{label}-bands.pwo` are the two pw.x outputs, and `{label}-bands.json`
is the result file with every eigenvalue. When the listing holds more
than 1000 eigenvalues, `bands` leaves them out and names the file in
`energies_in`. Read the json with `read_artifact`, not the `.pwo`. A
failed step keeps its files as `{label}-scf-failed.*` or
`{label}-bands-failed.*`, and the failure notes name the step.

`scripts/bands_table.py` reads the json. It prints the summary as JSON,
writes a table with `--dat`, and draws the diagram with `--png` when
matplotlib is installed. Without matplotlib, `--png` exits 2 and names
the package. The table has the distance along the path in 1/Å, then one
column per band, in eV relative to the valence band maximum, or to the
Fermi level for a metal. These are the first lines of
`bands_table.py si-bands.json --dat si-bands.dat` for the Si example
above:

```
# band structure along GXWKGLUWLK,UX (FCC), 60 k-points, 8 bands
# energies in eV relative to the valence band maximum (6.1597 eV)
# special points (label x): G 0.000000 X 1.157124 W 1.735687 K 2.144792 G 3.372108 L 4.374207 U 5.082798 W 5.491903 L 6.310113 K 7.018704 U 7.018704 X 7.427810
# columns: x (1/A), then band 1 to band 8
0.000000 -11.9916 0.0000 0.0000 0.0000 2.5225 2.5225 2.5225 3.3149
```

## 5. Report it honestly

- Report the gap with its kind, the functional, the pseudopotential
  family, the protocol, and the path.
- A PBE or PBEsol gap is a lower bound. Semilocal functionals place the
  conduction bands too low. The Si run above gives 0.468 eV, and
  the measured gap is 1.17 eV. Do not report a semilocal gap as the
  material's gap.
- The band edges are found on the path only. The conduction band minimum
  of Si lies at about 85 percent of the way from Γ to X, and a path with
  few points misses it by a step. `cbm_at` gives the distance to the
  nearest special point, so read it before you name the edge "at X".
  Raise `npoints` when the gap must be resolved to 10 meV.
- A metal has no gap. Report `is_metal` and do not report a gap of zero.
- Cite the run id. The number rests on the run that kept the json.

## When not to use this

- Do not use it for a magnetic system or for spin-orbit splitting. The
  task refuses both.
- Do not use it for a density of states or projected bands. The task
  computes neither.
- Do not use it for a gap that must match experiment. A semilocal
  functional does not give that gap, so use a hybrid functional or GW
  outside this task.
