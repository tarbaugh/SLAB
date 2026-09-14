---
name: lammps-scripting
description: Write a LAMMPS input script and run it whole through the
  run_lammps task, so the dynamics run inside LAMMPS at its own speed -
  the input anatomy in order, the ensembles and their damping constants,
  outputs and restarts, the checks that gate a run, a thermo report script
  for equilibration, and the errors LAMMPS prints. Use for every
  molecular dynamics run (NVE, NVT, NPT, Langevin), a minimization in
  LAMMPS, and any static calculation on more than a few hundred atoms,
  sized with gpus= when the machine declares a gpu build and the slice
  can hold a gpu; the ASE-driven lammps engine is only for a small
  relaxation or single point that feeds another task.
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

The rule: any molecular dynamics, and any static calculation on more
than a few hundred atoms, runs through `run_lammps` sized with `gpus=`
when the machine declares a gpu build and the slice can hold a gpu.
Threads through the plain build are the fallback when it cannot. The
ASE-driven `lammps` engine is for a relaxation or single point that
feeds another task, on a small cell. The numbers are guidance
thresholds, not limits: a few hundred atoms, a thousand steps.

| Need | Route |
| --- | --- |
| MD of any length | `run_lammps` sized with `gpus=`; threads through the plain build when no gpu can be held |
| A static calculation above a few hundred atoms | `run_lammps` sized with `gpus=`, with the same fallback |
| A minimization LAMMPS does better (`fix box/relax`, `min_style fire`) | `run_lammps` |
| A relaxation or single point on a small cell, feeding another task | the `lammps` engine with `relax` or `single_point`, unsized |

The engine sends the whole input to LAMMPS once per force call, so a
long MD through it pays a `read_data` and a potential load on every
step, and the gpu build cannot speed dynamics through it. A script pays
them once. A smoke test on the small cell still runs plain first, and
the accelerated run must reproduce it within the tolerance the
lammps-potentials skill states in section 4.

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
- `engine="lammps"` is all you pass. The build follows the slice: a
  launch sized with `gpus=` runs the gpu build under
  `[engines.lammps.gpu]`, and an unsized launch runs the plain build
  under `[engines.lammps]`. Never name a build. The `lammps` entry of
  `list_engines` lists every build with its command and the switches
  parsed from it. Run smoke tests and small cells unsized, and size
  production MD on the GPU partition with `gpus=`. A build whose
  command holds `{ntasks}`, `{threads}`, or `{gpus}` is filled from the
  launch's size (the `placeholders` field names them). Nothing adds a
  switch a build lacks: a run whose `kokkos.enabled` is false is a host
  run whatever the build contains. `command=` overrides the chosen
  build's command for that call alone. `setup=` (a list, or one string
  with a line per newline) runs after the build's own setup lines, so
  the build's module environment stays. `setup_mode="replace"` runs the
  per-call lines alone; do not use it unless the build's lines are
  wrong. `info["build"]` names the build
  that ran, and the build and the filled command enter the cache
  identity.
- `timeout_s` kills the process group; the job's time limit is the
  outer guard, and `timer timeout` inside the script (section 7) stops
  the run cleanly before either.

Dry-run every new or edited script before its first real launch:
`launch_workflow(script="md.py", dry_run=true)` (or `slab run --dry-run
md.py`). The script runs to its end or its first exception in a
throwaway workspace, every `run_lammps` call sets LAMMPS up and
integrates no step, and the reply lists each LAMMPS error, the checks,
and the outputs. A check that failed is expected to, because no step
ran, and a check that passed on no data is not evidence. A check that
raised is a bug in the check: fix it before the real launch. It costs one
LAMMPS start and catches a syntax error in the third stage and a wrong
result key in the analysis before the MD leg is paid for. In a dry run
each thermo table holds one row (step 0) and each `fix ave/time` file
holds none, so write the analysis to survive an empty series (`if
rows:`), or read the dry run as clean when every `run_lammps` line says
`setup ok` and the exception sits in the analysis of a series. A real
launch of a script text never dry-run in the session carries a warning.

What comes back is `result`, printed here from the template run as it
is (`pprint(result, sort_dicts=False)`), so nothing about its shape has
to be remembered or guessed:

```python
{'label': 'ar-nvt',
 'thermo': {'Step': 2000,
            'Temp': 287.34524,
            'PotEng': -5.2415743,
            'KinEng': 3.9742248,
            'TotEng': -1.2673495,
            'Press': 7799.2977,
            'Volume': 3929.3526},
 'tables': [{'columns': ['Step',
                         'Temp',
                         'PotEng',
                         'KinEng',
                         'TotEng',
                         'Press',
                         'Volume'],
             'first': {'Step': 0,
                       'Temp': 300,
                       'PotEng': -8.3818426,
                       'KinEng': 4.1492507,
                       'TotEng': -4.2325919,
                       'Press': 1276.3849,
                       'Volume': 3929.3526},
             'last': {'Step': 2000,
                      'Temp': 287.34524,
                      'PotEng': -5.2415743,
                      'KinEng': 3.9742248,
                      'TotEng': -1.2673495,
                      'Press': 7799.2977,
                      'Volume': 3929.3526},
             'n_rows': 21,
             'loop': {'seconds': 0.0471725,
                      'procs': 1,
                      'steps': 2000,
                      'atoms': 108},
             'tail': {'n_rows': 11,
                      'mean': {'Step': 1500.0,
                               'Temp': 303.1883590909091,
                               'PotEng': -4.8320841727272725,
                               'KinEng': 4.193348318181818,
                               'TotEng': -0.6387358388181817,
                               'Press': 8675.960354545454,
                               'Volume': 3929.3525999999997},
                      'std': {'Step': 316.22776601683796,
                              'Temp': 17.687067925615477,
                              'PotEng': 0.26138224059907544,
                              'KinEng': 0.2446269188014934,
                              'TotEng': 0.3973099484859172,
                              'Press': 540.569720729633,
                              'Volume': 4.547473508864641e-13}}}],
 'averages': {'ar-nvt-avg.dat': {'columns': ['TimeStep',
                                             'c_thermo_temp',
                                             'c_thermo_press'],
                                 'first': {'TimeStep': 100,
                                           'c_thermo_temp': 171.058,
                                           'c_thermo_press': 4880.43},
                                 'last': {'TimeStep': 2000,
                                          'c_thermo_temp': 285.596,
                                          'c_thermo_press': 8245.7},
                                 'n_rows': 20,
                                 'loop': None,
                                 'tail': {'n_rows': 10,
                                          'mean': {'TimeStep': 1550.0,
                                                   'c_thermo_temp': 301.6838,
                                                   'c_thermo_press': 8786.126999999999},
                                          'std': {'TimeStep': 287.22813232690146,
                                                  'c_thermo_temp': 19.127423411426857,
                                                  'c_thermo_press': 414.93737997558117}}}},
 'steps': 2000,
 'seconds': 0.0471725,
 'atoms': 108,
 'rate': {'steps_per_s': 42397.58333774975,
          'atom_steps_per_s': 4578939.000476973},
 'wall_time': '0:00:00',
 'artifacts': {'thermo': '19cdefe1a3645216199442cb8e2dcf5b5004fb694261c9c76ba1d7fac4de7198',
               'averages': 'a2def63b38479acf9e5230b36857ab7cdb8af641698d1e0e2df7e801c517cbba'}}
```

- `result["rate"]` and `result["seconds"]` are numbers from the loop
  lines; `wall_time` is LAMMPS's own text, for the report, never for
  arithmetic. `n_rows` is a count; the rows themselves are not in
  `result`.
- `result["averages"]` holds every `fix ave/time` file the script wrote,
  parsed, keyed by basename, in the same shape as a thermo table, with
  `loop` None. The full parse is the `{label}-averages.json` artifact.
- `series(result, 0)` or `series(result, "ar-nvt-avg.dat")` from
  `foundation.tasks` gives the full rows of a thermo table by index or
  of an averages file by basename, one dict per row keyed by column,
  read from the parsed artifacts (a cache hit still resolves them). It
  is the one way to a time series. Never write your own parser for a
  log, a `-thermo.json`, or a `.dat` file.
- Every output the script names (`dump`, `write_data`, `write_restart`,
  `restart`, `fix ... file`) is a bare basename, so the run keeps it.
  A path with a directory component is refused before LAMMPS starts,
  because a file written into the project directory is not an
  artifact of any run and the analysis then rests on nothing.
- `info["kokkos"]`: what the log says KOKKOS did. `enabled`, `gpus` per
  node, `threads` per task, the `/kk` `styles` that ran, and under
  `switches` what the command asked for. Quote it in the notebook for
  every GPU run; a GPU number with `enabled: False` was computed on the
  host. `info["argv"]` is the exact argument vector that ran.
- `info["artifacts"]`: `{label}.in`, `{label}.log`, `{label}.screen`,
  `{label}-structure.data`, `{label}-thermo.json` (every table, every
  row), `{label}-averages.json`, and every file the script wrote
  (dumps, restarts, `write_data` output, `fix ave/time` files). Read
  them with `read_artifact`; the log digest shows each table's ends and
  the warnings, and a `fix ave/time` file digests to its fix, columns,
  and ends.

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
thermo_modify line yaml flush yes
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
- Put `thermo_modify line yaml` after each `thermo_style` line. LAMMPS
  then prints the table as one YAML document, and `run_lammps` reads it
  by schema instead of by a regex over the text table, so a long column
  list or a WARNING inside the table cannot break the parse. Every
  `thermo_style` line resets the setting, so a script with two
  `thermo_style` lines needs the yaml line twice. `info["thermo_format"]`
  reports `yaml` when the run took this path.
- `dump ID all custom N file.dump id type x y z vx vy vz` writes frames
  ASE reads back with `format="lammps-dump-text"`, and the analysis
  skills (msd-diffusion, radial-distribution) read those frames.
  `dump_modify ID sort id` keeps atoms in order. Every byte of a dump is
  kept as an artifact, so choose `N` for the analysis, not for comfort:
  a frame every few hundred steps is usual.
- `fix ID all ave/time Nevery Nrepeat Nfreq c_thermo_temp c_thermo_press
  file averages.dat` writes running averages of any compute or variable
  to a file the task keeps, and the file comes back parsed under
  `result["averages"]["averages.dat"]` with its full rows through
  `series(result, "averages.dat")`. `compute msd all msd` and
  `compute rdf all rdf 100` feed it: `c_msd[4]` is the total
  mean-squared displacement, and `c_rdf[*]` with `mode vector` writes
  the histogram, which comes back as the artifact only (`mode:
  "vector"`, no rows in the summary). Name the file by bare basename.
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

### Close contacts and the push-off

A minimization under the real potential cannot repair a cell whose
atoms overlap: a pair a fraction of a bond apart carries a force the
optimizer follows off a cliff, and the log says `Lost atoms` or ends in
a welded cluster. Check the minimum interatomic distance before the
first run with the atomsk-structures skill's `check_structure.py`, and
treat anything below 0.6 of the shortest expected bond as a close
contact. A crystal with close contacts is a wrong build, so rebuild it.
A disordered cell (random placement, a liquid or amorphous start, a
merge, a polycrystal seam, an interstitial dropped by hand) almost
always has some, so push them apart first under a soft repulsion,
before the real potential sees the cell:

```
pair_style soft 2.5
pair_coeff * * 0.0
variable prefactor equal ramp(0.0,30.0)
fix push all adapt 1 pair soft a * * v_prefactor
fix lim all nve/limit 0.05
velocity all create 300.0 4928459 dist gaussian
timestep 0.001
run 5000
unfix push
unfix lim
```

`pair_style soft` is a bounded cosine repulsion with no force above its
cutoff, so the cutoff is the distance at which two atoms stop counting
as too close: the shortest bond you expect in the cell, 2.5 to 3 Å for
a metal, 1.5 Å for a hydride. The prefactor ramps from zero to about
30 eV under `units metal`, so the first steps move overlapping pairs
gently and the last steps separate them fully. `fix nve/limit` caps
every atom's displacement per step at 0.05 Å, which is the guard that
keeps a strongly overlapping pair from flying out of the box. Then set
the real `pair_style` and `pair_coeff`, minimize as above, and run
`check_structure.py --format lammps-data --species <elements in type
order>` on the written data to confirm no contact remains below the
threshold. The push-off is not part of the physics; the
report lists it as a preparation step, and the trajectory that counts
begins after the minimization under the real potential.

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

Return `(passed, observed, expected)` from each check, as in
`return abs(mean - 300.0) < 15.0, round(mean, 1), "300 +/- 15 K"`. The
record then states the value the check judged. A bare `True` or
`False` leaves a record that says only `returned False`; the store
then keeps the check's source and the keys of the dicts it read, but
not the number.

A check can be wrong while the physics is right: a tolerance too
tight, a key the result does not have, a zero denominator on a short
table. Do not relaunch the run. Fix the check, rehearse the fixed
script on the run's cached results with
`launch_workflow(script=..., dry_run=true, from_run=<run>)`, then call
`reverify_run(run_id=<run>, script=...)` (or `slab runs reverify <run>
<script>`). Every task call takes the run's own result, no engine
starts, and no new run is recorded; the checks become a new
verification pass, and the run moves to verified when all pass. Change
only the checks: a task whose inputs changed is refused, because it
needs a new computation.

A slope, a fit, or any number that needs more than the ends and the
tail reads the rows through `series(result, -1)` for the last thermo
table or `series(result, "msd.dat")` for an averages file, one dict per
row keyed by column. Do not parse the log, the `-thermo.json`, or a
`.dat` file yourself; a hand-written parser is what crashed the
analysis of a finished MD leg in one real campaign.

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
| `KeyError: 0` or `KeyError: 'rows'` on `result["tables"][i]` in your own script | `n_rows` is a count; the summary holds no rows | `series(result, i)` for the rows |
| `TypeError: unsupported operand ... 'str'` on `result["wall_time"]` in your own script | `wall_time` is LAMMPS's text | `result["seconds"]` or `result["rate"]` for arithmetic |

The failure record carries the `ERROR` line and the line before it,
the kept `{label}-failed.log` has the rest, and `{label}-failed.screen`
holds what MPI or the loader printed when LAMMPS never started.

## 10. KOKKOS and MPI

The build carries the parallel launch; the script stays the same, and
SLAB adds no switch. A machine keeps a plain build and a gpu build, and
the build follows the slice: size the launch with `gpus=` and the gpu
build runs, leave it unsized and the plain build runs. Read the
`lammps` entry of `list_engines` before a GPU run, and `info["kokkos"]`
after it.

The production launch, inside a sandbox or an allocation, sizes the run
with `gpus=` alone:

```
launch_workflow(script="md.py", gpus=2)
```

That gives one MPI rank per GPU and the free cpus as threads across the
ranks; `ntasks=` equal to `gpus=` says the same. Never ask for more
ranks than GPUs. The gpu build refuses such a launch before LAMMPS
starts, because every rank past the first on a device fails on an
exclusive-mode device with `cudaErrorDevicesUnavailable`. The call
reserves the GPUs and the cpus before the run starts and is refused
with the free amounts when they are taken. Before a second
concurrent launch, call `free_resources`, because the free amounts in
the environment block were read when the prompt was built and the first
launch now holds its slice. On a login node the same run is a job on
the GPU partition through `submit_job` with `gpus_per_node` and
`ntasks_per_node` equal to it. Never login-node work. After the run,
`info["kokkos"]["gpus"]` must equal what the launch held.

When the machine declares no gpu build, or the budget holds no gpu, the
fallback is threads through the plain build: size the launch with
`ntasks=` and `threads=` so tasks times threads stays within the cores
it holds, and say in the notebook that the run was not accelerated. Do
not hold an MD run for a gpu the budget cannot give.

`-sf kk` gives every style in the script its Kokkos version where one
exists, and a fix or compute without one runs on the host and copies
data back each step, so keep the script inside Kokkos-enabled styles
and keep the thermo and dump intervals long. The lammps-potentials
skill, section 4, has the switches, the one-task-per-GPU rule, and the
smoke comparison against the plain build.
