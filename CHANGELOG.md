# Changelog

All notable changes to SLAB, newest first. Dates are commit dates on
`main`.

## Unreleased

- Purge leaves nothing behind. After `slab fast-forward --include-running`
  and `slab purge --all-sessions --yes` the workspace holds the promoted
  and archived rows with the blobs they reach, the serve record of a
  live job, and nothing under the session, review, lock, record, and
  job directories, and the scratch root holds no `slab-*` directory
  whose owner is not a live process. Every slab-managed scratch
  directory now carries a `.slab-owner` marker naming its process,
  host, and run (`$SLAB_RUN_ID`, which `start_run` exports), and
  `sweep_scratch` removes the ones whose run is over or whose process
  is gone, never by age. A reap, a job cancel, a retire in purge mode,
  and `slab fast-forward` remove the scratch of the runs they fail or
  purge, and purge is the backstop. Purge is inventory first: one
  function lists every category with counts and bytes, `--dry-run`
  prints it, `--json` prints it as JSON, and the confirmation names its
  totals. Orphan delegation transcripts, unrecognised session files,
  stale harness records, and stale session locks are categories of
  their own. `slab doctor` gains a `leftovers` row.
- The sandbox job carries its job id into the container. `apptainer
  exec --cleanenv` stripped `SLURM_JOB_ID`, so the runs a sandbox job
  made carried no job id and `slab hpc cancel` with a workspace failed
  none of them. The render exports it beside `SLURM_NTASKS`, empty
  outside a job, and the sandbox context says the runs carry it.
- The loader checks each keyword after `-pk kokkos` in both LAMMPS
  commands against the keywords `package kokkos` documents, and refuses
  a typo at load naming the table, because LAMMPS would refuse it at
  every run. `slab doctor` gains three kinds of row: one per partition
  with the caps a sized job is checked against, one for a plain LAMMPS
  command with no launcher and no `{ntasks}`, and one for an
  openai-provider `[agent]` with `context_window` unset. Each states a
  choice the config made without saying so, and none fails the doctor.
- The LAMMPS skills state one rule for speed. Any molecular dynamics,
  and any static calculation on more than a few hundred atoms, runs
  through `run_lammps` sized with `gpus=` when the machine declares a
  gpu build and the slice can hold a gpu, with threads through the plain
  build as the fallback. The ASE-driven `lammps` engine is for a small
  relaxation or single point that feeds another task. The
  lammps-scripting and lammps-potentials skills, the md-expert card, the
  lammps note, the cluster and workstation compute profiles, and the
  `relax` and `single_point` docstrings say so.
- The gpu build is `[engines.lammps.gpu]`, and the slice chooses it. The
  table carries a KOKKOS `command`, which must name `{gpus}` or turn
  KOKKOS on with `-k on`, and its `setup`. `[engines.lammps]` stays the
  plain build, and the loader refuses a plain command that carries
  `-k on g`, naming both tables. A launch whose reservation holds gpus
  (`gpus=` on `launch_workflow`, or a batch job that holds gpus) runs
  the gpu build, and a launch without runs the plain build, on `relax`,
  `single_point`, and `run_lammps` alike. The agent never names a build;
  `engine="lammps"` is all it passes. `command=` and `setup=` remain
  per-call overrides. A registry alias whose calculator is the LAMMPS
  factory is a further build under its own name, and is no longer the
  documented way to declare the GPU build. The docs, the
  lammps-scripting and lammps-potentials skills, the md-expert card, and
  the lammps note say to size the launch with `gpus=` and confirm with
  `info["kokkos"]`.
- The partition table is the cap. `check_size` checks a sized job against
  the fields the partition itself declares: `nodes`, `ntasks_per_node` (or
  `ntasks` when only that is set), `cpus_per_task` as cores per node,
  the gpu count in `gres`, and `mem`. A field the partition leaves unset
  is no cap, and SLURM enforces its own limit. Each refusal names the
  field, such as `[hpc.partitions.gpu] gres`, and a size that asks gpus
  on a partition without a `gres` is refused. An unsized job renders
  exactly as before. `slab hpc partitions` prints no node line, and
  `list_engines` reports each partition's `nodes`, `ntasks_per_node`,
  `cpus_per_task`, `mem`, `gres`, `time_limit`, and `description`.
- `run_lammps` info names the build. `info["build"]` is `cpu`, `gpu`, or
  the alias name, `lammps_build()` and `lammps_builds()` in `slab.lammps`
  resolve and list the builds, and `slab mason read` prints `build gpu`
  on a command line whose build is not `cpu`.
- `slab engines list` prints a `lammps builds (gpu build chosen when the
  launch holds gpus):` block, `cpu` first, `gpu` when declared, then the
  aliases, each with its command, its `sized per launch` placeholders,
  and its KOKKOS switches.
- `gres_gpus` in `slab.resources` reads the gpu count from the `gpu`
  entry of a gres string alone, and returns none when the entry names
  no count.
- Free resources as a tool. `free_resources`, in Mason and over MCP,
  returns what `Workspace.free_resources` returns plus one line per live
  reservation with its slice, its run or holder, and its age, read at
  call time. Every `list_runs` answer ends with `free now: N cpu(s), M
  gpu(s)`, and the cpus line of the environment block says to call the
  tool before a concurrent launch, because its own free amounts were
  read when the prompt was built. This is not a mechanism switch. The
  shell could already reach the same rows through `slab runs reservations`.
- A claim is one transaction. `claim_reservation(pid=)` sets `run_id` on
  the reservation, `resources` on the run, and the run's status to
  `running` with its pid and host together, and `start_run(reservation=)`
  uses it, so no moment exists in which a claimed slice counts as free.
  A claimed reservation whose run is still pending follows its holder's
  liveness.
- A hard-killed child is reaped. `process_alive` collects a dead child of
  this process with one non-blocking `waitpid` before it probes, so a
  background launch that an OOM kill or a `SIGKILL` ended no longer keeps
  its run and its reservation live until an unrelated `Popen` reaps it.
- SLURM's gpu ids are renumbered. `budget()` takes only the count of the
  devices `SLURM_JOB_GPUS` or `SLURM_STEP_GPUS` names and numbers them from
  zero, and the sandbox prologue does the same in shell, because a job
  under cgroup device constraints sees its devices renumbered and the
  global ids would name devices a child cannot open.
- The shell refuses `python -m slab_stack.cli run` as it refuses `slab
  run`.
- `list_engines` degrades without the store. When the workspace cannot be
  opened (a `PermissionError` included, which `_open_workspace` now treats
  as a store fault), the tool still lists the engines and the budget, with
  `free` null and a `resources_note` that says why, in Mason and over MCP.
  MCP `launch_workflow` releases a reservation only for the failures a
  launch can report, so an interrupt no longer deletes the row of a child
  that may be running.
- A comma-separated gres is sized entry by entry. `sized_gres` sizes the
  `gpu` entry alone and keeps `nvme:1` or `shard:8` as written, and the
  sandbox reads the gpu count from the `gpu` entry alone.
- `memory_mb` rounds a K value up to the next megabyte and refuses zero,
  so `500K` no longer passes every cap as `0`.
- Size arguments are checked. A `ntasks`, `threads`, `gpus`,
  `ntasks_per_node`, `cpus_per_task`, `gpus_per_node`, or `nodes` that is
  not an integer or is below one (zero for the gpu counts) comes back as
  `refused: <name> must be a positive integer, not <value>` in Mason and
  as a tool error over MCP, and `Workspace.reserve` raises `ValueError`
  instead of clamping to one. A comment that names an `mpirun` no longer
  trips the rank check.
- The refusal of an unsized launch names the free gpus as well as the
  cpus, and the tool descriptions say that `gpus` without `ntasks` takes
  every free cpu and the gpus asked.

- The agent sizes every launch and every job. `slab.resources` is the one
  place that knows what a process may use: `budget()` discovers the cpu
  ids of the affinity mask and the visible gpu ids, `envelope()` reads
  the per-launch `SLAB_CPUS`, `SLAB_GPUS`, `SLAB_NTASKS`, and
  `SLAB_THREADS` variables and falls back to the budget and SLURM's
  counts, `env_for` exports an envelope to a child, and `apply` takes its
  affinity mask. An engine command holds `{ntasks}`, `{threads}`, and
  `{gpus}` where it wants the launch's numbers, and `fill` replaces only
  the placeholders a command asks for; a `{gpus}` under a launch without
  a GPU is refused naming the build. The QE bin form is now literally
  `mpirun -np {ntasks} <bin>/pw.x`, LAMMPS builds fill their command the
  same way, `lammps_builds()` reports each build's `placeholders`, and
  `slab engines list` marks such a build `sized per launch`.
- A batch job is sized per job. `render_sbatch(size=)` takes a `JobSize`
  (`nodes`, `ntasks_per_node`, `cpus_per_task`, `gpus_per_node`, `mem`)
  whose directives replace the partition's own, the gres keeps the type
  the partition names, and `check_size` refuses a size past the fields
  the partition declares, naming the config field. Without a size the
  script is byte for byte what it was. `slab hpc render` and `slab hpc
  submit` take the five size flags.
- A launch reserves its slice before it starts. The run store (schema 5)
  keeps a `reservations` table and a `resources` column on runs.
  `Workspace.reserve` checks out cpu ids and gpu ids on this host inside
  one transaction, so two reservers never overlap, and refuses a slice
  that does not fit with the free amounts; an unsized reservation takes
  the whole free cpu budget. `start_run(reservation=)` claims it, copying
  the slice onto the run record, and the reservation is released when
  the run ends or its holder dies: free is derived from the live
  reservations, never counted. `Workspace.free_resources` is the read
  side, and `reap_dead` (so `slab runs reap` and every `wait_for_run`
  poll) releases the dead ones. `slab run --reservation` takes the mask
  and the GPU variables before the script runs, `launch_child` runs a
  sized launch as such a child, `slab list` gets a `RES` column, `slab
  show` and `show_run` print the slice, and `slab runs reservations`
  lists the reservations with their holder and age.
- The MCP tools take sizes. `launch_workflow` gains `ntasks`, `threads`,
  and `gpus`; the server reserves with its own pid, runs a sized launch
  as a child, and returns the refusal with the free amounts when the
  slice does not fit. `submit_job` gains `nodes`, `ntasks_per_node`,
  `cpus_per_task`, `gpus_per_node`, and `mem`. `list_engines` carries
  `budget` and `free`. The session record's `launch` and `job` events
  carry the slice and the size.
- Mason sizes its launches and its jobs. `launch_workflow` takes
  `ntasks`, `threads`, and `gpus`; the session reserves the slice with
  its own pid, a sized launch runs as a child `foundation run
  --reservation` (waited unless `background=true`), an unsized
  foreground launch runs in-process as before but still reserves the
  whole free budget, and a background launch is always the child. A
  slice that does not fit is refused as tool text with the free amounts,
  a hand-written `mpirun` in a script is judged against the launch's own
  slice, and the `shell` tool refuses `slab run` and `foundation run`,
  pointing at `launch_workflow`. `submit_job` takes the five size
  arguments and returns a `check_size` refusal as tool text.
  `list_engines` carries `budget` and `free`. The `launch` command event
  records the slice and the `job` event the size, and `slab mason read
  --full` prints both. The environment block states the budget, what is
  free right now, and the default rank count of an unsized launch, and
  the `cluster` compute profile says `submit_job` is sized per job up to
  the fields the partition declares.
- The sandbox re-exports what `--cleanenv` strips. The rendered script
  carries `CUDA_VISIBLE_DEVICES` (from `SLURM_JOB_GPUS` when the job did
  not set it) and `SLURM_CPUS_PER_TASK` into the container, and sets
  `OMPI_MCA_hwloc_base_binding_policy=none` so concurrent launches bind
  inside their masks. The context block states the GPU count and where
  the ids come from, and says a launch is sized. `slab mason sandbox
  render` and `launch` and `slab benchmark launch` take the five
  job-size flags, `render.json` records the size, and a bare `launch`
  and `slab doctor` reuse it. `--engine-tasks` keeps its meaning as the
  rank count of an unsized launch.
- The md-expert card and the lammps-scripting and lammps-potentials
  skills size a GPU run where it runs: `gpus_per_node` on `submit_job`
  from a login node, `gpus=` with one rank per GPU on `launch_workflow`
  inside a sandbox or an allocation, and `info["kokkos"]["gpus"]` must
  equal what the launch held. A build with placeholders is `sized per
  launch`. The docs state the guarantee: a launch can oversubscribe only
  its own slice, never a neighbour's.

- Transcripts record every command that ran. The `shell` tool records
  its command line, `launch_workflow` the driver's command, `submit_job`
  the payload with the job id and the kept script, and a finished run
  the engine commands its tasks resolved, read from the run's recipes by
  the new `run_commands` operation. Each `command` event names the agent
  card that ran it. `slab mason read` prints one line per command and
  `--full` the details; `slab mason report` counts them by kind. The MCP
  server records the same events in the harness session record.
- LAMMPS builds are named. `run_lammps` takes `engine=`, as `relax` and
  `single_point` already did, and `command=` and `setup=` override the
  chosen build. The build enters the cache identity and the transcript's
  command events. `slab engines list` and `list_engines` list every
  build with its command and the KOKKOS switches parsed from it, because
  SLAB adds no switch a build lacks. `run_lammps` returns `info["kokkos"]`
  with what the log reported (KOKKOS mode, GPUs per node, threads per
  task, the `/kk` styles that ran) and `info["argv"]`, the exact argument
  vector. The lammps-scripting and lammps-potentials skills and the
  md-expert card say to read the listing before a GPU run and
  `info["kokkos"]` after it.
- A session retires at finish. Mason passes the `run_ids` of a root
  session's `finish` to the new `retire_session` operation, which
  promotes the cited runs that passed their checks, anchors from earlier
  sessions included, and expires the session's other runs. A cited run
  that never verified is reported and left alone. The transcript records
  a `retire` event with the numbers, `slab mason report` and `slab mason
  read` show it, and the benchmark record carries them as `retention`;
  `slab benchmark tables --retention` prints them. `slab retire` and the
  MCP tool `retire_session` run the same operation by hand, `--dry-run`
  reports without writing, and the retention policy's new `finish` rule
  sets the default for the uncited runs (`keep`, `expire`, or `purge`).
  `purge_expired` takes `only=` to restrict a purge to named runs. The
  science review reads a run's history, so a verified run that expired
  at finish is not a failure.
- LAMMPS runs whole input scripts as a traced task. `run_lammps` takes
  the script as text, stages potential files and a structure beside it,
  runs `lmp -in` in slab-managed scratch, and keeps the script, the log,
  the screen capture, every file the script wrote, and the parsed thermo
  tables as artifacts; `result` carries the last thermo row and each
  table's ends, loop line, and tail statistics for the checks. The
  command, the version, the setup lines, and the content of every staged
  file enter the cache identity, and a script that dies keeps its
  evidence with the `ERROR` line and its context as notes. The new
  `lammps-scripting` skill gives the md-expert the input anatomy, the
  ensembles and their constants, outputs and restarts, the guards, the
  errors LAMMPS prints, a tested equilibration report script, and a
  workflow template that runs as-is. `slab.outputs.lammps_thermo` parses
  thermo tables.
- The md-expert card and the lammps-potentials skill say how to run a
  KOKKOS build of LAMMPS: the switches in the engine command, one MPI task
  per GPU, the package options for many-body potentials, and the smoke
  comparison against the plain build. The same skill names the LAMMPS
  route for each GRACE form (the saved model, the Kokkos weights, the FS
  export) and the FS extrapolation grade lines. The mlip-training skill
  gains uncertainty quantification with `grace_uq`, seed ensembles, and
  the FS active set, the fp32 and Kokkos acceleration routes, and an
  active-learning loop. `pair_style_for.py` names the Kokkos `.npz` and
  the `FS_model.yaml` exports, with `--layers` for the Kokkos pair style.
- The `context-hygiene` switch also lifts the output cap on one tool
  result, so the ablation measures the cap with the clearing it precedes.
- The mechanism ledger carries a measured-effect column. Every row reads
  "not yet measured" until the ablation grid has run the mechanism off and
  on, so the ledger says which mechanisms are finished and which are
  measured.
- The benchmark runs one question under three harness conditions, so the
  reliability claim is measured, not asserted. `slab` is Mason as it is.
  `protocol` is a skill collection with a file protocol: a `protocol`
  card with file tools, a shell, and the skill catalog, running scripts as
  `python script.py` and keeping an append-only provenance log in the
  project, the shape of the AICC control plane. `bare` is the model with
  read, write, shell, and finish and a one-paragraph prompt. `--condition`
  on `slab benchmark run`, `launch`, and `render` selects one; the
  transcript header and the record carry it, and the results table keys
  its rows by condition. The scorer's verification rule is the same for
  every arm, so a protocol or bare campaign passes only by producing a
  verified run. `slab benchmark matrix` renders the launch grid of
  conditions by questions by an optional ablation list without
  submitting. Every harness mechanism is now a switch in `[agent]
  mechanisms` (check gating, failure records, the critic gate, machine
  memory, context hygiene, identical-result annotation, the budget hint,
  the looking hint, skills, delegation, adaptive effort), the loop, the
  toolbox, and the
  prompt consult it, and the benchmark page keeps the ledger. Agent cards
  take `core: false` to supply their whole prompt.
- The extended XYZ digest reports the closest pair of atoms in the file,
  through the periodic images, with its frame and species, a flag when it
  is under 60 % of the covalent-radii sum, and the mean nearest-neighbour
  distance. Large files are sampled evenly to bound the pair count.
- The compaction brief asks for evidence behind every failure it records,
  and keeps a reading the agent revised or retracted under OPEN with both
  readings. One summary recorded a misreading of a pw.x output as a
  failure, and the next six compactions carried the argument instead of
  the file.
- Engine outputs come back digested. `read_file` and `read_artifact` return a
  digest of a pw.x output, a LAMMPS log, or an extended XYZ file: the
  system, the SCF trace with its convergence line, the final energy, the
  largest force, the pressure, the warnings, the error block, the wall
  time, and whether the job finished. Band eigenvalue lists never appear.
  `raw=true`, or a window, reads the text as before. The parsers live in
  `slab.outputs` and are tested on the real captures under `tests/data`.
  One session read a 305 KB pw.x output in 400-line windows, took a band
  block in eV for a diverging SCF energy in Ry, and compacted six times in
  sixteen minutes arguing with itself about it.
- The workflow script is kept as the run's `input` artifact under its own
  name, so `show_run` lists it and `read_artifact` reads it. One lead
  searched the project, the workspace, and scratch for a script the run
  record could have shown.
- `[engines.rootstock]` takes `setup` lines. They run once in a login
  shell when a rootstock calculator is built, and the variables they add
  or change are applied to the process before the worker is spawned, so
  the worker inherits them; job submission still restores the environment
  it started from. `slab doctor` runs the lines and counts what they
  applied. The sandbox render snapshots them like the other engines'
  setup, binding the interpreter the setup puts on PATH. One sandboxed
  campaign found every MACE checkpoint dead of a torchvision import that
  worked on the node, because the container had only the install root and
  none of the CUDA environment the user's shell carried.
- The review brief writes the scope a lead had to learn to write by hand.
  It says not to re-run the recorded lookups it carries, and a second
  review of the same subject in one session carries the prior findings and
  asks whether each is resolved. A reply cut at the reply-token ceiling is
  asked once more at low effort with a request for brevity, instead of the
  identical request that would be cut the same way. When a critic is cut
  on the retry too, the verdict line tells the lead its one move and
  leaves the config to the operator. One real review ran 78 minutes and
  returned no verdict; the lead's hand-scoped third review took ten.
- The mlip-training skill separates the dataset for a from-scratch fit
  from the dataset for fine-tuning a foundation model. From scratch: a
  coverage table by region, atom counts over structure counts,
  decorrelated snapshots, one round of active learning, and a transfer
  test set. Fine-tuning: tens to a few hundred targeted structures, the
  PBE provenance of the GRACE OAM labels against SLAB's PBEsol default,
  an anchor slice against forgetting, `eval_init_stats` as the baseline,
  and an untrained property computed with both models. The shared rules
  hold out whole sources and note that `single_point` labels carry no
  stress.
  The engines tutorial gains a paragraph on what the collector leaves to
  the dataset's author, by kind of fit.
- The review brief carries the lead's own catalog and material lookups
  (`list_engines`, `list_tasks`, `describe_task`, `get_material`) from the
  session, newest first, within a 2,000-character cap each and 8,000 in
  all. Two real critic passes spent five steps each re-running those
  lookups before reaching the plan's substance.
- Context economy in the loop. After fifteen consecutive steps made only
  of looking tools, the per-step budget line tells the model to step
  back, and again every five steps. Only the newest plan echo stays
  verbatim; older ones become a one-line marker when a newer one lands. A
  second identical fetch whose first copy is still in context returns a
  pointer to it instead of the body (shell results exempt). Every request
  carries `max_tokens`: `[agent] max_reply_tokens` or 16,000, bounded by
  the room the window has left. One campaign spent 72 minutes in a
  look-only run, carried three plan copies in every prompt, re-read the
  same files, and ended on a 20,086-token reply.
- The test suite sleeps for real again. The fixture that skips the model
  client's retry backoff patched `time.sleep` on the shared module, so every
  wait in the suite became a no-op: the run-id ordering test compared ids
  minted in the same millisecond and failed two runs in three, and the
  tests that wait for an orphaned child or a silent endpoint measured
  nothing. The client now waits through its own `_sleep` attribute and the
  fixture patches that. The run-id test waits for the clock to tick instead
  of sleeping.
- A quarantined run explains itself. A custom check may return
  `(passed, observed)` or `(passed, observed, expected)`, or a dict with
  those keys, and the record shows the value instead of `returned False`.
  `show_run` takes `task=<label or seq>` and returns that one task in
  full, and the new `read_artifact` tool reads a run's artifact as text
  by name, windowed like `read_file`. One real session read a full record
  cut at the cap, then guessed the artifact store's layout by hand for
  six minutes to read one output file.
- A `lammps-potentials` skill for the md-expert. Its script reads a
  potential file's header, names the format (funcfl, setfl, eam/fs, ACE,
  GRACE, MEAM), and prints the `pair_style` and `pair_coeff` lines; the
  skill body carries the smoke test to run before production. One campaign
  spent seventy minutes rewriting a correct setfl file that `pair_style
  eam` rejected, when `eam/alloy` was the fix. The pi card now says to load
  the covering skill or delegate when a command fails the same way twice.
- Two rules join the science review. `no-progress-loop` flags fifteen or
  more consecutive steps that only looked (shell, reads, listings) with
  no run, plan change, note, brief, or finish, on the card; one real
  campaign spent 72 minutes and 80 % of its completion tokens in two such
  windows. `reasoning-heavy` flags a step billed 8,000 or more completion
  tokens that wrote nothing, on the prompt, and names the effort the
  transcript header recorded.
- A model call that gets a 5xx answer retries five times with exponential
  backoff, about 80 s in all, instead of three times inside five seconds,
  so a gateway that answers 502 for a minute is outlasted; a connection
  that cannot be made still fails in seconds. A critic or a specialist whose
  server fails mid-turn returns a result with the steps it reached and its
  transcript, where before the exception discarded the review and the lead
  paid for it twice.
- A reply with no text and no tool call is nudged once, not taken as the
  answer, and the OpenAI-compatible `length` finish reason now reads as
  `max_tokens`, so a truncated turn on vLLM or a gateway carries the
  truncation marker instead of passing as finished. The transcript header
  records `effort` and `version`, and every usage event records
  `finish_reason`. The compaction summarizer runs at `low` effort with a
  4,096-token reply budget. The shell tool decodes binary output with
  replacement characters instead of failing the call. `slab doctor` notes
  a roster table that sets `effort` while `[agent]` does not, and a table
  for a card that only `--agent` can start.
- `slab purge` sweeps a conversation's compaction summaries and review
  records with its transcripts. Both were left behind before: the
  transcript sweep matched `<stem>-*.jsonl` only, so
  `<stem>.compactions.md` and `mason/reviews/<stem>-review-<n>.md`
  accumulated forever. The newest conversation keeps its files as it
  keeps its transcript, and `--all-sessions` removes them too.
- Every bundled skill was checked against the method literature and the
  tool documentation, and fixed one commit per skill. Wrong as written:
  the atomsk lattice name, `--merge`/`-cell` forms, and gap advice; the
  equation-of-state range (linear factors, three times the Delta range
  in volume) and its stiffness caveat; the surface-energy thickness rule
  (a separate bulk energy drifts linearly); the adhesion wetting formula
  (a vacuum work over a solid-liquid energy); the same-species g(r)
  normalisation; the melt-quench "tail" (a ramp, not a hold) and rate
  floor; the thermal ramp's per-cell latent heat and target-temperature
  fit; the two-phase isotropic barostat; the NEMD fit through a
  sawtooth; the CNT mean gamma; the gracemaker fine-tuning key and the
  frame-level test split. Every script now reports an uncertainty where
  one exists (block errors, fit covariances, bootstraps, replica
  spreads), the MD templates sample under the isotropic
  Martyna-Tobias-Klein integrator, and the skills state the numeric
  conditions (tolerances, windows, thresholds, sizes) a reportable number
  needs. New: `--quantity` on the convergence table, B' and residuals on
  the EOS fit, Born margins and a fit spread on the elastic fit, the
  interface energy and the solid-liquid wetting relation on adhesion,
  beta and the Yeh-Hummer correction on the MSD, the coordination number
  on the RDF, weighted Arrhenius with errors, MYEGA, and a windowed
  melting crossing, hold-averaged densities with a log-rate law,
  per-atom c_p with hysteresis, a folded NEMD fit with conductance, a
  cluster-count gamma(T) with a rate mode, and an interface-velocity
  script for coexistence runs.
- `slab mason read` and `slab mason report` find the workspace the
  current directory is inside. Without `--workspace`, the resolution
  used to fall to the project's `./.slab`, so running the viewer from a
  shared workspace reported no transcripts there. Standing in a
  workspace, its `mason/sessions` directory included, now names it, and
  an explicit `--workspace` still wins. A workspace with one conversation
  is read without the number prompt.
- `[agent] effort` reaches an OpenAI-compatible server verbatim. `xhigh`
  and `max` were folded to `high`, which on a server whose top level is
  the unset default meant a planner asking for the most reasoning and
  getting less. `xhigh` now goes out as itself and `max` as `xhigh`. The
  scale gains `none`, the field's own off switch, for a worker that
  should not think at all. The Anthropic provider has no `none` and
  sends `low` for it.
- The science review. A scored campaign now carries flags beside its
  verdict: attributable defects, each with a rule, a target (`skill:`,
  `card:`, `tool:`, or `prompt`), the evidence, and a note. The rules run
  on every `slab benchmark score` and `run`; `--referee` also asks a model
  to argue with the procedure from an evidence pack, and a referee that
  cannot be read leaves the rules' flags in place. Every skill has a
  digest, recorded by the `skill` tool when it loads and in the record
  under `skills`, so a flag is raised against one revision. `slab
  benchmark flags` is the defect list with a status per flag (open,
  pending, unknown), `slab benchmark gate <skill>` refuses a revision
  until a campaign under it passes without regressing or raising the
  flag, and `tables` renders a flags region on the benchmark page.
  `docs/review.md` describes the loop.
- Two campaign transcripts, read and acted on. The run store asks for its
  journaling mode and keeps what the database can give: a store that
  wants rollback journaling opens a database another process holds in
  WAL mode, in WAL, instead of refusing every open under a hot upgrade,
  and a failed open closes its connection instead of leaking one that
  blocks the next. Mason's run tools report an unopenable store with the
  recovery (wait, retry once, report) instead of a bare "database is
  locked", the shell refuses to delete or move the store's files, and the
  prompt says the workspace is a record, not a thing to repair.
  `show_run` folds finished tasks to one line each (`full=true` returns
  the recipes), `wait_for_run` reports each run's task tally, both take a
  run's name as well as its id, `list_runs` takes `status`, and
  background launches write their log line by line. A cleared tool
  result keeps its first line, and the third identical return of one call
  in a session carries a note to write the fact down. `[agent] effort`
  now reaches an OpenAI-compatible server as `reasoning_effort`; it was
  not sent at all before, so a `low` worker reasoned exactly like an
  `xhigh` one.
- The session lock is per project directory, not per workspace. It guards
  `NOTEBOOK.md` and `PLAN.md`, which belong to the project, so two
  campaigns in two project directories now share one workspace instead of
  the second being refused with the first's path. To make that sharing
  safe, the run store opens with rollback journaling when the database
  sits on a network filesystem (Lustre, GPFS, NFS, and the like), where
  WAL's shared-memory index cannot be seen from a second node;
  `SLAB_SQLITE_JOURNAL=wal|delete` overrides the detection.
- A planner and a worker. The `planner` card writes the plan and hands
  every step to the team; its tool allowlist has no shell, no launch, and
  no file edits, and it is refused up front when delegation is off or
  nobody on the roster can take a brief. The `worker` card executes any
  scoped step no specialist's domain names. A card that delegates is a
  lead, never a hand: the two leads are not on each other's team, and a
  brief to one is refused. `--agent` on `slab mason sandbox render` and
  `launch` and on `slab benchmark render` and `launch` names the entry
  card, the render records it, and the `[agent.roster.<name>]` tables now
  travel into the sandbox (minus provider, endpoint, and key), so the
  planner can reason at `xhigh` while the worker runs at `low` inside one
  job.
- Machine memories carry a version stamp. `remember` records the software
  the fact names, at the versions present when it was written, under
  `against` in the frontmatter; the catalog compares each stamp with the
  machine at session start and marks the memories whose software changed
  since, so the agent re-checks those and relies on the rest without
  probing. `recall` and `slab memory list` show the same note. The
  versions are probed once per session, and only when a memory carries a
  stamp.
- Orphan artifact bytes are reclaimed. The tracer now serializes every
  argument, probes the engines, and runs the task's `cache_extra` before
  any byte lands in the store, so a refusal leaves nothing behind. For
  orphans that exist anyway (a process killed mid-write), `gc` drops
  those unreferenced for `orphan_ttl_days` (a new policy field, default
  1 day) and reports them as `orphans_dropped`; younger ones stay listed
  under `orphans`. `null` keeps orphans forever, as before.
- A review of the last two weeks' revisions, six reviewers over the four
  packages, the docs, the tests, and the security surface; every finding
  confirmed by execution before it was fixed. The fixes:
  - `relax_cell(symmetry="isotropic")` can converge on a non-cubic cell.
    ASE's own test demanded every normal stress component vanish while
    the mask moved only the volume, so hexagonal and tetragonal cells
    burned every step; the optimizer now judges the filter's projected
    stress, and `info["smax"]` reports the same quantity.
  - A Ctrl-C during a slow tool no longer leaves the assistant's tool
    calls unanswered (a protocol-invalid history that `--resume` replayed).
  - The sessions-directory fence compared a resolved path against an
    unresolved one, so the default relative workspace and any symlinked
    workspace left it open; `search` followed symlinks out of the fence.
  - The model's API key was inherited by every shell command and workflow
    the model ran; it is now read once, withdrawn from the environment,
    and kept on the session for delegates.
  - The sandbox `verify` step took an HTTP error page for a dark network.
  - The sandbox render quotes the `slab` path, keeps a bind whose
    destination differs from its source, binds a distro-packaged tool as
    a file rather than its `/usr` prefix, keeps launchers and arguments
    when it makes a bare command absolute, escapes control characters in
    the rendered config, and refuses a non-OpenAI provider up front.
  - `slab mason serve stop` clears an unreadable endpoint record; `slab
    purge` reports one as an error instead of a traceback and creates no
    workspace on `--dry-run`.
  - `slab benchmark score` skips a session it cannot judge and scores the
    rest, instead of aborting the sweep.
  - `promote --session --force` no longer sweeps running or pending runs
    into `promoted`; a run whose failure could not be recorded keeps its
    real exception; run timestamps must be timezone-aware; a gracemaker or
    atomsk timeout keeps the partial log; a relative `[builders.mp] root`
    resolves; `atomsk_version` never raises; the shebang fallback only
    trusts an absolute interpreter; a broken config no longer hides the
    built-in engines from `slab engines list` or tracebacks in `slab
    pseudos install`; `--filter is_stable=true` matches the snapshot's
    integer booleans; `api_key_env` must be a shell variable name.
  - Compaction cannot refire every step; the compaction summarizer's own
    call is recorded as usage; the prompt-size estimate counts the tool
    schemas; a delegate's report is never cleared; a resumed session gets
    its own header; the shared prompt names the tools a session lacks.
  - Docs: eleven MCP tools (not nine or seven), five `slab hpc` verbs,
    `remember` asks for approval, the tool table lists every tool, the
    roster and offline doctor captures are re-recorded, and the sandbox
    tests use ephemeral ports so two test runs on one machine no longer
    collide.
- A truncated context is named, not endured. Ollama silently truncates
  every prompt to its `num_ctx` (2048 or 4096 by default), below Mason's
  fixed prefix, so every local llama session so far ran without its
  instructions. `slab mason doctor` now sends a 6,000-word prompt and
  reports whether the server counted it whole; a session whose server
  counts far fewer prompt tokens than were sent records a warning once,
  and `slab mason report` and `slab mason read` show it. The tutorial
  gives the Modelfile fix.
- Context hygiene in three layers. Tool output is capped at 12,000
  characters (was 24,000); once the prompt passes a quarter of the
  context window, tool results older than the newest six are replaced by
  a placeholder that names the tool and the size, in batches so the
  cached prompt prefix is rewritten rarely; compaction stays the rare
  fallback. Errors, skill texts, and plan updates are never cleared.
  Usage events and `slab mason report` now carry the cached share of the
  prompt, the peak prompt size, and the clearing count, so a token total
  can be read as a cost.
- The sandbox render carries builders in. `[builders]` travels whole into
  the rendered `slab.toml` (a sandbox without it reported every builder
  absent, whatever was mounted), a configured atomsk or gracemaker is
  snapshotted like an engine so its install is bound and its setup frozen,
  a console script is followed to its interpreter's real prefix, and the
  context file names each carried tool so the session does not spend its
  opening steps probing for it.
- `slab benchmark` judges a DFT campaign against the reference for the
  functional it used, PBE or PBEsol, read from the traced calculator
  options. The default SSSP families are PBEsol, and the first cluster
  campaign's correct PBEsol lattice constant would have failed the PBE
  band. Question 3 has no checked PBEsol reference yet, so a PBEsol
  campaign on it is refused rather than guessed at.
- `slab benchmark`: five fixed copper campaigns with DFT-PBE references and
  per-engine-class tolerance bands; `run`, `launch`, `render` (the job
  files, for hand edits before `sbatch`), `score`, and `tables`.
  A campaign passes when the agent's structured `finish` result lies in
  band and every run it cites reached `verified`. Records live in
  `benchmarks/results.jsonl`; the docs page and the README tables are
  rendered from them.
- Mason's `finish` tool takes structured `results` and `run_ids`; every
  transcript opens with a header naming the model, provider, endpoint,
  and compute profile; `slab mason report --session` finds a session by id.

## 0.1.0 — 2026-09-01

The first tagged version. Three weeks of work, from an empty repository
to a four-package distribution with a resident agent, verified against
real engines.

### The state layer (`foundation`)

- Run lifecycle state machine with SQLite persistence: quarantined,
  verified, promoted, archived, expired. Promoted data cannot expire,
  structurally.
- Content-addressed artifact store with tiered retention by artifact
  role, and retention policy as data.
- Define-by-run tracing (`@task`) with content-hash caching, and
  verification hooks (`@check`) that decide when a run is verified.
- Failure is evidence: structured failure records, surviving diagnostics,
  and checks that record observed and expected values.
- Session-stamped runs, `slab promote --session`, and machine memory: a
  store for what one session learns and the next needs.
- Ready-made traced tasks: `relax`, `relax_cell`, `single_point`,
  `build_structure`, `fetch_structure`, `collect_training_data`,
  `train_potential`.
- An MCP server (`slab mcp`) with 11 tools over the same operations layer
  as the CLI.

### Access to software (`slab`)

- Engine seam over the ASE `Calculator` contract: EMT and Lennard-Jones
  built in, Quantum ESPRESSO and LAMMPS as built-ins verified against real
  `pw.x` and `lmp`, rootstock-served MLIP checkpoint ids usable directly as
  engine names, and a cluster engine registry for everything else.
- Per-engine environments: setup lines scoped by a wrapper shell, the
  `env` wrapper blessed, import-time environment refused.
- AiiDA's named Quantum ESPRESSO input protocols (`fast`, `balanced`,
  `stringent`) and SSSP pseudopotential families, adopted as
  policy-as-data.
- Layered TOML configuration (site, user, project) with per-table owners,
  a commented template, and a thin SLURM layer (`slab hpc`).
- Three builders as traced tasks: atomsk structures, an offline Materials
  Project snapshot (read-only, no network, absence reported as absence),
  and MLIP training with gracemaker, verified against a real
  tensorpotential fit.

### The resident agent (`mason`)

- A ReAct harness for open-weight models through the OpenAI-compatible API
  and for Claude through the Anthropic API, with a stdlib HTTP client.
- The model served as a batch job whose endpoint is discovered, not
  configured.
- A roster of agent cards (PI plus specialists) with one-level delegation,
  18 skills in the Agent Skills format, curated software notes, and a
  compute profile that sizes calculations to the machine.
- The sandbox: autonomous runs inside a rendered, fail-closed Apptainer
  job, with an authenticating gateway bridge, a file fence, and a session
  lock.
- Campaign tools: `slab mason report`, `slab mason read`,
  `slab mason sandbox launch`, and background workflows with
  `wait_for_run`.

### The distribution (`slab_stack`)

- One command, `slab`, composes all four packages.
- `slab doctor`: a whole-stack preflight that means "ready to launch a
  campaign", with `--deep` probes.
- `slab fast-forward` and `slab purge` for retention housekeeping across
  the layers.

### Documentation and quality

- A documentation site with twelve tutorials whose captured outputs come
  from real executions, and an architecture document.
- Prose follows the ASD-STE100 guiding principles.
- 1600+ tests including every docstring example as a doctest, ~94%
  coverage, mypy `--strict`, a layering test that reads the AST, and a
  test workflow on Python 3.11, 3.12, and 3.13.
