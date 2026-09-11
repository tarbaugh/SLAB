Classical molecular dynamics and statics through the LAMMPS binary. Fast at
real system sizes — thousands of atoms and beyond — and the accuracy is
entirely the interatomic potential's, so name the potential and its source
in every report.

The potential is required and is a science decision: pass `pair_style=` and
`pair_coeff=` (plus `files=` for potential files) in `calculator_options`.
There is no usable default. Entries in `files=` are staged into the
calculation's scratch, and bare basenames in `pair_coeff` resolve to the
staged copies, so an absolute path in `files=` works from any directory.

If a run dies with only `Failed to retrieve any thermo_style-output`, that
is LAMMPS masking its real error. The actual `ERROR: ...` line is in the
LAMMPS log kept in the run's failure evidence — read it with `show_run`.

Two entry points drive this binary. The `lammps` engine answers force
calls from ASE (`relax`, `single_point`), one `run 0` per step. The
`run_lammps` task hands LAMMPS a whole input script as text, so the
dynamics run inside LAMMPS at its own speed, on the threads or GPUs the
command line gives it, and the log, the dumps, and the thermo tables
come back as artifacts of the run. Production MD goes through
`run_lammps`; the lammps-scripting skill carries the input anatomy, the
ensembles, and the checks.

The build follows the slice. A launch sized with `gpus=` runs the gpu
build from `[engines.lammps.gpu]`, and an unsized launch runs the plain
build from `[engines.lammps]`. Never name a build; `engine="lammps"` is
all you pass. Read the `lammps` entry of `list_engines` before a GPU
run, and `info["kokkos"]` after it, because SLAB adds no switch a build
lacks.
