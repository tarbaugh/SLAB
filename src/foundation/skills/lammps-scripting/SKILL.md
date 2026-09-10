---
name: lammps-scripting
description: Write a LAMMPS input script and run it whole through the
  run_lammps task, so the dynamics run inside LAMMPS at its own speed -
  the input anatomy in order, the ensembles and their damping constants,
  outputs and restarts, the checks that gate a run, a thermo report script
  for equilibration, and the errors LAMMPS prints. Use for production MD
  (NVE, NVT, NPT, Langevin), a minimization in LAMMPS, a long run on GPUs
  or threads, or when an ASE-driven engine run is too slow.
license: MIT
metadata:
  mason-agents: "md-expert"
---
# LAMMPS scripting

LAMMPS is fastest when it runs its own loop. The `run_lammps` task
hands LAMMPS a whole input script, in a slab-managed scratch directory,
and keeps what LAMMPS wrote as artifacts of the run. Everything below
is about writing that script well and judging what came back.

## 1. Script or engine

| Need | Route |
| --- | --- |
| Production MD, thousands of steps or more | `run_lammps` with a script |
| A minimization LAMMPS does better (`fix box/relax`, `min_style fire`) | `run_lammps` |
| GPUs or threads through a KOKKOS build | `run_lammps`; the switches ride in the command (lammps-potentials skill, section 4) |
| A relaxation or single point ASE drives, feeding another task | the `lammps` engine with `relax` or `single_point` |

The engine sends the whole input to LAMMPS once per force call, so a
long MD through it pays a `read_data` and a potential load on every
step. A script pays them once.

## 2. The task

Copy `assets/md_nvt.py` into the project directory as the starting
point. It runs as-is (argon under Lennard-Jones, no potential file) and
launches with `launch_workflow`. The shape:

```python
result, info = run_lammps(SCRIPT, atoms=STRUCTURE, files=["W.eam.fs"], label="w-npt")
```

- `SCRIPT` is the input text, verbatim, never a path. The text enters
  the cache identity, so an identical script under the same binary and
  the same files is a cache hit.
- `atoms=` writes the structure as `structure.data` (`units metal`,
  `atom_style atomic`, masses included); the script must `read_data
  structure.data`. With more than one element pass `specorder=`, and
  `info["types"]` says which element each type is, so `pair_coeff`
  lists the elements in that order.
- `files=` stages potential files, data files, and restarts beside the
  script under their basenames. Name each by bare basename in the
  script; the task refuses a file the script never mentions.
- `command=` and `setup=` override `[engines.lammps]`. A KOKKOS or MPI
  launch is `command="mpirun -np 4 lmp -k on g 4 -sf kk"`; the command
  enters the cache identity. Nothing adds a switch the command lacks:
  the `lammps` entry of `list_engines` shows the configured command and
  the switches parsed from it, and when `kokkos.enabled` is false there,
  a run without `command=` is a host run whatever the build contains.
- `timeout_s` kills the process group; the job's time limit is the
  outer guard, and `timer timeout` inside the script (section 7) stops
  the run cleanly before either.

What comes back:

- `result["thermo"]`: the last thermo row, keyed by column name.
- `result["tables"]`: one entry per thermo table with `columns`,
  `first`, `last`, `rows`, the `loop` line (steps, atoms, seconds), and
  `tail` (mean and std of every column over the last half of the rows),
  the numbers a `@check` judges.
- `result["steps"]` and `result["wall_time"]`.
- `info["kokkos"]`: what the log says KOKKOS did. `enabled`, `gpus` per
  node, `threads` per task, the `/kk` `styles` that ran, and under
  `switches` what the command asked for. Quote it in the notebook for
  every GPU run; a GPU number with `enabled: False` was computed on the
  host. `info["argv"]` is the exact argument vector that ran.
- `info["artifacts"]`: `{label}.in`, `{label}.log`, `{label}.screen`,
  `{label}-structure.data`, `{label}-thermo.json` (every table, every
  row), and every file the script wrote (dumps, restarts, `write_data`
  output, `fix ave/time` files). Read them with `read_artifact`; the
  log digest shows each table's ends and the warnings.

A script that dies keeps `{label}-failed.in`, `.log`, and `.screen`,
and the failure record's notes carry the `ERROR` line with the command
that died. A script that finishes with bad physics is not a failure:
the checks decide.

## 3. Anatomy of an input, in order

LAMMPS reads the script top to bottom, and most commands must come
after the ones they depend on. Keep this order:

```
# 1. Units, style, boundaries, then the structure
units metal
atom_style atomic
boundary p p p
read_data structure.data

# 2. The potential (the lammps-potentials skill gives these lines)
pair_style eam/fs
pair_coeff * * W.eam.fs W

# 3. Neighbor lists
neighbor 2.0 bin
neigh_modify every 1 delay 0 check yes

# 4. Initial velocities and the timestep
velocity all create 300.0 4928459 mom yes rot yes dist gaussian
timestep 0.001

# 5. The integrator (one per group of atoms)
fix integrate all nvt temp 300.0 300.0 0.1

# 6. Output: thermo, dumps, averages, restarts
thermo 100
thermo_style custom step temp pe ke etotal press vol
thermo_modify flush yes
dump traj all custom 1000 w-nvt.dump id type x y z vx vy vz
dump_modify traj sort id

# 7. Run, then write the final state
run 20000
write_data w-nvt-final.data
```

- `units metal`: eV, Å, ps, bar, K. The timestep is in picoseconds
  (`0.001` is 1 fs), pressures in bar, damping constants in ps.
- `read_data` needs `units` and `atom_style` before it. `create_box`
  with `lattice` and `create_atoms` builds a crystal without a data
  file; `mass 1 183.84` is then required.
- `velocity ... create T seed`: the seed reproduces the run; `mom yes
  rot yes` removes the drift; `dist gaussian` is the Maxwell start.
- Several `run` commands in one script give several thermo tables;
  `reset_timestep 0` after equilibration starts the production count at
  zero.

## 4. Ensembles and their constants

| Ensemble | Line | Constants |
| --- | --- | --- |
| NVE | `fix integrate all nve` | none; the energy conservation is the check |
| NVT, Nose-Hoover | `fix integrate all nvt temp T T Tdamp` | `Tdamp` about 100 timesteps (0.1 ps at 1 fs) |
| NPT, Nose-Hoover | `fix integrate all npt temp T T Tdamp iso P P Pdamp` | `Pdamp` about 1000 timesteps; `aniso` lets the cell lengths move separately, `tri` adds the tilts |
| Langevin | `fix thermostat all langevin T T damp seed` and `fix integrate all nve` | `damp` in ps, 0.1 to 1; needs `nve` with it |
| Berendsen | `fix thermostat all temp/berendsen T T Tdamp` with `nve` | reaches a target fast, samples no ensemble; equilibration only |

- Exactly one integrator per atom. Two fixes that both integrate the
  same group give `WARNING: One or more atoms are time integrated more
  than once`, and the trajectory is wrong.
- The thermostat's time constant is part of the method. Record it with
  the ensemble in the report: "NVT, Nose-Hoover, Tdamp 0.1 ps" tells a
  reader what "NVT" alone does not.
- Timestep by the fastest motion: 1 to 2 fs for metals under EAM or a
  GRACE potential, 0.5 fs when hydrogen moves, 2 fs for Lennard-Jones
  argon. When in doubt, halve it and confirm the observable holds.
- Equilibrate first under the production ensemble, then
  `reset_timestep 0` and run production. The tail statistics and the
  thermo report (section 8) prove the equilibration.

## 5. Output

- `thermo N` and `thermo_style custom step temp pe ke etotal press vol
  lx ly lz` print the table the task parses. Keep `N` small enough for
  statistics (hundreds of rows) and large enough not to dominate the
  cost. `thermo_modify flush yes` writes each row at once, so a killed
  run keeps its log.
- `dump ID all custom N file.dump id type x y z vx vy vz` writes frames
  ASE reads back with `format="lammps-dump-text"`, and the analysis
  skills (msd-diffusion, radial-distribution) read those frames.
  `dump_modify ID sort id` keeps atoms in order. Every byte of a dump is
  kept as an artifact, so choose `N` for the analysis, not for comfort:
  a frame every few hundred steps is usual.
- `fix ID all ave/time Nevery Nrepeat Nfreq c_thermo_temp c_thermo_press
  file averages.txt` writes running averages of any compute or variable
  to a file the task keeps. `compute msd all msd` and `compute rdf all
  rdf 100` feed it: `c_msd[4]` is the total mean-squared displacement,
  and `c_rdf[*]` with `mode vector` writes the histogram.
- `write_data final.data` at the end, and `restart 10000 restart.*.bin`
  during the run, so a later script can continue with `read_restart`.
  A restart file does not carry the potential file; restate
  `pair_style` and `pair_coeff` after `read_restart`, and stage the
  restart with `files=`.

## 6. Minimization

```
min_style cg
fix relax all box/relax iso 0.0 vmax 0.001
minimize 1.0e-8 1.0e-8 5000 10000
unfix relax
```

`minimize etol ftol maxiter maxeval` stops on the energy change, the
force norm, or the counts. `fix box/relax` relaxes the cell with the
positions; leave it out to relax positions only. The log's `Stopping
criterion` line says why it stopped, and the task's log digest reports
it. Minimize before MD when the structure came from a builder, so the
first steps do not blow up.

## 7. Guards inside the script

- `neigh_modify every 1 delay 0 check yes`: rebuild when an atom moved
  half the skin. `Dangerous builds` above zero in the log means the
  skin was too small or the rebuild too rare.
- `timer timeout 1:50:00 every 100`: stop cleanly before a two-hour job
  limit, so the log ends with its loop line and the restart is written.
- `fix halt N v_name > value error soft`, with `variable name equal
  ...`, stops the run on a condition: a temperature above a limit, a
  volume that doubled, an extrapolation grade above a threshold
  (lammps-potentials skill, section 5). `error soft` finishes the
  script after the stop.
- `run N upto` runs to an absolute step count, and `label loop` with
  `jump SELF loop` repeats a block; both make a restartable protocol.

## 8. Check, measure, and report

Gate the run with `@check` functions on `result`, the way the template
does: the run finished every step and lost no atoms (the loop line's
atom count equals the structure's), the tail mean of the temperature
sits within a stated tolerance of the target, the tail of the total
energy in NVE does not drift, and the pressure under NPT averages to
the target. State the tolerances before the run.

Then measure the equilibration from the full table:

    python <skill root>/scripts/thermo_report.py w-nvt-thermo.json --tail 0.5

The script reads the `-thermo.json` artifact (or a log), takes the last
table, and prints for each column the mean, the standard deviation, the
block standard error over the tail (five blocks, so correlated rows do
not pass as independent samples), and the drift of a straight-line fit
across the tail in units of that error. A drift above three errors
flags the column: hold longer or discard more before averaging.
`--table 0` picks another table, `--columns Temp Press` restricts the
report, and `--json` gives the numbers machine-readable. Pass the
artifact to the script through `read_artifact` into a project file, or
run the script on the log the same way.

Report the ensemble and its constants, the timestep, the seed, the
potential and its provenance, the number of atoms and steps, the
equilibration span discarded, the tail means with their block errors,
and the run id. A number without a run id is a rumor.

## 9. What LAMMPS says when it fails

| Message | Cause | Fix |
| --- | --- | --- |
| `Lost atoms: original N current M` | atoms flew out of the box: too large a timestep, overlapping atoms, or a bad start | halve the timestep; minimize first; check the density and the initial cell |
| `Non-numeric atom coords - simulation unstable`, `Non-numeric pressure` | the same blow-up, caught by another guard | the same fixes |
| `Unrecognized pair style 'x'` | the build lacks the package, or a typo | the lammps-potentials skill; check the binary's `-h` package list |
| `Cannot open file X` | not staged, or named by a path | pass it in `files=` and name it by bare basename |
| `Incorrect args for pair coefficients` | the element list does not match the atom types | `info["types"]` gives the order; one symbol per type |
| `Unknown identifier in data file`, `Incorrect atom format in data file` | `atom_style` does not match the data file | the task writes `atomic`; set that, or stage your own data file |
| `Neighbor list overflow, boost neigh_modify one` | too many neighbors per atom for the page | `neigh_modify one 10000 page 100000` |
| `Illegal ... command` | syntax | read the context line the failure record carries; it is the command that died |
| `WARNING: One or more atoms are time integrated more than once` | two integrators on one group | keep one |

The failure record carries the `ERROR` line and the line before it,
the kept `{label}-failed.log` has the rest, and `{label}-failed.screen`
holds what MPI or the loader printed when LAMMPS never started.

## 10. KOKKOS and MPI

The command carries the parallel launch; the script stays the same, and
SLAB adds no switch. Read the `lammps` entry of `list_engines` before a
GPU run, and `info["kokkos"]` after it.
`-sf kk` gives every style in the script its Kokkos version where one
exists, and a fix or compute without one runs on the host and copies
data back each step, so keep the script inside Kokkos-enabled styles
and keep the thermo and dump intervals long. The lammps-potentials
skill, section 4, has the switches, the one-task-per-GPU rule, and the
smoke comparison against the plain build. A GPU run is a job on the GPU
partition through `submit_job`, never login-node work.
