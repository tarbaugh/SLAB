---
name: lammps-potentials
description: Choose the LAMMPS pair_style and pair_coeff lines for a
  potential file (EAM funcfl, setfl, eam/fs, ACE, GRACE, MEAM), check that
  the potential loads and gives physical energies before any production
  run, run a KOKKOS build on GPUs or threads with the right switches,
  deploy a GRACE potential in the right pair style and read its
  extrapolation grade, and diagnose "Not a valid floating-point number"
  and similar potential-file errors. Use when a LAMMPS run needs a
  classical or machine-learned potential, when a run should use GPUs, or
  when LAMMPS rejects a potential file.
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
different type-to-element mapping. `--layers 2` names the layer count that fixes a Kokkos
weights file's pair style. `--json` gives the same facts as a
machine-readable object. Exit code 2 means the file is none of the
formats below; report that instead of guessing.

| File form | Header signature | `pair_style` | `pair_coeff` |
| --- | --- | --- | --- |
| funcfl (`.eam`) | line 2 `Z mass a lattice`, line 3 `nrho drho nr dr cutoff` | `eam` | `* * FILE` (one element) |
| setfl (`.eam.alloy`, "DYNAMO 86 setfl") | 3 comment lines, then `N El1 El2 ...`, then the grid line | `eam/alloy` | `* * FILE El1 El2 ...` |
| Finnis-Sinclair setfl (`.eam.fs`) | as setfl | `eam/fs` | `* * FILE El1 El2 ...` |
| ACE / PACE (`.yace`, `.ace`) | YAML-like, no elements in the header | `pace` | `* * FILE El1 El2 ...` |
| GRACE saved model (a directory) | as gracemaker wrote it | `grace` (a TensorFlow build) | `* * DIR El1 El2 ...` |
| GRACE Kokkos weights (`.npz`) | from `grace_utils export_kokkos` or `grace_models download --kokkos` | `grace/1l/kk`, `grace/2l/kk`, or `grace/3l/kk` by the model's layers (`--layers`) | `* * FILE El1 El2 ...` |
| GRACE/FS export (`FS_model.yaml`) | from `gracemaker -sf` | `grace/fs` (`grace/fs/kk` under KOKKOS) | `* * FILE El1 El2 ...` |
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

## 4. Run a KOKKOS build on GPUs or threads

KOKKOS is a LAMMPS package. The binary must be built with it, and a GPU
build is compiled for the node's GPU architecture, so a cluster ships
it as a separate module. The `-h` banner of the binary lists the
installed packages, and KOKKOS must be among them. Module loads go in
`[engines.lammps] setup` or in a registry alias, never in the input. A
GPU run is sized where it runs. On a login node it is a batch job on the
GPU partition through `submit_job` with `gpus_per_node` and
`ntasks_per_node` equal to it. Inside a sandbox or an allocation it is a
`launch_workflow` call with `gpus=` and `ntasks=` equal to it, which
reserves that slice before the run starts; a slice that does not fit
what is free is refused with the free amounts, so read `list_engines`
(`budget` and `free`) and size within it. One MPI task per GPU either
way. Never start a GPU run on the login node itself.

The switches ride in a route's `command`. A machine keeps more than one
LAMMPS, so each build is a route with a name: `lammps` is the plain
build under `[engines.lammps]`, and an accelerated build is a registry
alias such as `lammps-gpu` whose options carry the KOKKOS command and
its module. Choose the route by name, `engine="lammps-gpu"` on `relax`,
`single_point`, and `run_lammps`, and keep the plain route plain, so a
smoke test or a small EAM cell never queues for a GPU. ASE appends its
own flags after the switches. SLAB adds no switch: a route without
`-k on` runs the plain styles on the host, silently, whatever the build
contains. The `lammps` entry of `list_engines` lists every route with
its command and the switches parsed from it. A route whose command
holds `{ntasks}`, `{threads}`, or `{gpus}` is marked `sized per launch`:
SLAB fills those from the launch's size, so `ntasks=2, gpus=2` on the
launch runs `-np 2 ... g 2` on such a route (`gpus=2` alone takes every
free cpu and two gpus), and a route that asks `{gpus}` under a
launch without one is refused naming the route. A route that hardcodes
its numbers runs as written whatever the launch held. Read it before a
GPU run. When no accelerated route exists, pass `command=` with the
switches on that call alone, and report the missing route as a machine
fact.

| Hardware | the route's `command` | Meaning |
| --- | --- | --- |
| One GPU | `mpirun -np 1 lmp -k on g 1 -sf kk -pk kokkos newton on neigh half` | one MPI task, one GPU |
| N GPUs on one node | `mpirun -np N lmp -k on g N -sf kk -pk kokkos newton on neigh half` | `-np` equals the number of GPUs on the node |
| Sized per launch | `mpirun -np {ntasks} lmp -k on g {gpus} -sf kk -pk kokkos newton on neigh half` | the launch's `ntasks` and `gpus` fill in; keep them equal |
| CPU threads | `env OMP_PROC_BIND=spread OMP_PLACES=threads mpirun -np 2 lmp -k on t 8 -sf kk` | two MPI tasks with eight OpenMP threads each; tasks times threads never exceeds the physical cores |

- `-k on` enables KOKKOS and issues a `package kokkos` command with its
  defaults. `g Ng` is the number of GPUs per node, `t Nt` the OpenMP
  threads per MPI task.
- `-sf kk` appends `/kk` to every style that has a Kokkos version. A
  style without one runs on the host, and every such fix, compute,
  thermo line, or dump copies the data back from the device, so keep
  the input inside Kokkos-enabled styles and keep output intervals long.
- `-pk kokkos` overrides the package defaults. The GPU defaults are
  `newton off neigh full`; for many-body and machine-learned potentials
  (EAM, ACE, GRACE) `newton on neigh half` is usually faster, and it is
  the setting the GRACE documentation gives. The CPU defaults are
  already `newton on neigh half`.
- One MPI task per GPU. More tasks per GPU than one need CUDA MPS for
  acceptable performance.
- A segmentation fault with several ranks means the MPI library is not
  GPU-aware: add `gpu/aware off` to the `-pk kokkos` options.
- `suffix kk` and `package kokkos ...` can also be lines in the input,
  but the `command` is what SLAB traces and caches against, so keep the
  switches there.
- After a `run_lammps`, `info["kokkos"]` says what the log reported:
  `enabled`, `gpus` per node, `threads` per task, and the `/kk` styles
  that ran. Check that `gpus` equals what the launch held (`gpus=` on
  `launch_workflow`, `gpus_per_node` on `submit_job`) before you trust
  a timing or a number from a GPU run. The transcript records the
  command every run resolved and the slice each launch held, so a
  reviewer can check both.

How SLAB drives MD matters here. The `run_lammps` task (the
lammps-scripting skill) runs a whole input script inside LAMMPS, so the
potential loads once and the switches speed the entire run; that is the
route for production MD. The `lammps` engine asks LAMMPS for forces
from ASE, one `run 0` per step, and every call re-sends the input, the
data file, and the potential lines, so a potential that loads slowly (a
GRACE saved model under TensorFlow) pays its load on every step. Under
the engine the Kokkos weights and the FS export load fast, the switches
speed the force evaluation only, and the positions still cross the host
each step, so the gain is large for machine-learned potentials and large
cells, and small for EAM on a few hundred atoms.

Check before you spend an allocation:

1. Run the smoke test of section 3 with the plain command and with the
   switches. The `run 0` energies must agree to about 1 meV/atom; the
   `/kk/fp32` and `/kk/mixed` GRACE styles are allowed that much, a
   full-precision style far less.
2. Time 100 MD steps of the 3x3x3 cell both ways, and report the two
   wall times with the production plan. When the accelerated run is not
   clearly faster, use the plain build and say so.
3. Name the command, switches included, in the run's intent. It is part
   of the cache identity, so a run under another command is another run.

## 5. GRACE potentials in LAMMPS

One GRACE fit has three deployable forms, and the form decides the pair
style and the build of LAMMPS it needs. The mlip-training skill says how
each form is produced and how its extrapolation grade is built.

| Model | TensorFlow build | KOKKOS build, no TensorFlow | File |
| --- | --- | --- | --- |
| GRACE 1L | `grace`, `grace/1layer/chunk` | `grace/1l/kk` | saved-model directory, or `.npz` |
| GRACE 2L | `grace`, `grace/2layer/chunk`, `grace/2layer/parallel` | `grace/2l/kk` | saved-model directory, or `.npz` |
| GRACE 3L | none | `grace/3l/kk` | `.npz` only |
| GRACE/FS | `grace/fs` (C++, MPI, millions of atoms) | `grace/fs/kk` | `FS_model.yaml` |

- Units are `metal`. The `pair_coeff` line lists the elements in
  atom-type order after the file or directory name, exactly as for ACE.
- `pair_style grace` computes no pairwise forces by default, so a run
  that needs stress adds `pair_forces`:
  `pair_style grace padding 0.05 pad_verbose pair_forces`. `padding 0.05`
  limits the JIT recompilation that a change of neighbour count
  triggers. The chunk, parallel, and `/kk` variants always have stress.
- Large cells under TensorFlow use the chunk variants
  (`grace/1layer/chunk chunksize 2048`). Several GPUs with a 1L model
  need one MPI rank per GPU with `CUDA_VISIBLE_DEVICES` set per rank; a
  2L model on several GPUs uses `grace/2layer/chunk` or
  `grace/2layer/parallel`.
- The `/kk` styles read the weights that `grace_utils export_kokkos`
  or `grace_models download NAME --kokkos` writes, and run without
  TensorFlow. `/kk/fp32` and `/kk/mixed` select the precision, and the
  foundation 3L models run as `grace/3l/kk/fp32`. Run them with the
  switches of section 4. A 3L model has no TensorFlow pair style.
- The Kokkos 2L style fails to compile under CUDA 12.2 to 12.6 (an nvcc
  loop); a site build needs CUDA 12.8 or later. A missing style is a
  machine fact to report, never a reason to install anything.

### The extrapolation grade during a run

GRACE/FS reports its extrapolation grade gamma inside LAMMPS, from an
active set `FS_model.asi` next to the export (the mlip-training skill
builds it). The lines from the gracemaker documentation:

```
pair_style grace/fs extrapolation
pair_coeff * * FS_model.yaml FS_model.asi Mo Nb Ta W

fix grace_gamma all pair 100 grace/fs gamma 1
compute max_grace_gamma all reduce max f_grace_gamma
variable max_grace_gamma equal c_max_grace_gamma
fix extreme_extrapolation all halt 10 v_max_grace_gamma > 25
```

`fix pair` stores the per-atom gamma every 100 steps as
`f_grace_gamma`, the compute reduces it to the largest value, and
`fix halt` stops the run when any atom passes 25. This gamma is on the
D-optimality scale of PACE: above 5 is outside the training set, above
25 far outside. A dump of `f_grace_gamma` with `dump_modify ... skip` on
the same variable keeps only the extrapolating frames for labelling.

These lines belong in a script under the `run_lammps` task (the
lammps-scripting skill): the run keeps the log with the halt, the dump
of the extrapolating frames, and the `fix ave/time` file if you write
one, and a `@check` on the result decides whether the run reached its
steps. They cannot act inside the `lammps` engine, which drives MD from
ASE one `run 0` at a time, and a LAMMPS input started from the shell is
not a run: nothing traces it, and its numbers cannot be reported. For a
1L, 2L, or 3L model, grade the frames of a recorded run afterwards with
`grace_uq predict`, or GRACE/FS frames with `PyGRACEFSCalculator` and
its `.asi`, both in gracemaker's environment (the mlip-training skill,
section 5). The gracemaker documentation names no fix or compute that
reads gamma from the `grace` or `/kk` styles; do not invent one.
