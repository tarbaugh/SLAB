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
come back as artifacts of the run. The rule: any molecular dynamics,
and any static calculation on more than a few hundred atoms, goes
through `run_lammps` sized with `gpus=` when the machine declares a gpu
build and the slice can hold a gpu, with threads through the plain build
as the fallback. The engine is for a small relaxation or single point
that feeds another task. The lammps-scripting skill carries the input
anatomy, the ensembles, and the checks.

The thermo table is read by schema when the script asks for it. Put
`thermo_modify line yaml` after each `thermo_style` line, because a
`thermo_style` line resets it, and LAMMPS prints every table as one YAML
document that `run_lammps` parses without a regex. A script without the
line still parses from the text table, and `info["thermo_format"]` says
which path the run took (`yaml`, `text`, or `mixed`).

The build follows the slice. A launch sized with `gpus=` runs the gpu
build from `[engines.lammps.gpu]`, and an unsized launch runs the plain
build from `[engines.lammps]` on one rank with no GPU. Where the `cpu`
build in `list_engines` shows `requires_gpu: true`, size every launch
with `gpus=1`, dry runs included. Never name a build; `engine="lammps"` is
all you pass. Read the `lammps` entry of `list_engines` before a GPU
run, and `info["kokkos"]` after it, because SLAB adds no switch a build
lacks.
