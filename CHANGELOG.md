# Changelog

All notable changes to SLAB, newest first. Dates are commit dates on
`main`.

## Unreleased

- A Quantum ESPRESSO density of states is one traced task.
  `density_of_states(atoms, calculator_options=..., dos_kpts=,
  dos_kspacing=, projected=, emin=, emax=, delta_e=, degauss=, nbands=,
  label=)` in `foundation.tasks` runs an SCF, an NSCF on a denser mesh,
  and `dos.x` in one scratch directory, and with `projected=True` also
  `projwfc.x`. It returns the curve on its energy grid, the Fermi level,
  the density of states there, and the band edges and the gap as the NSCF
  eigenvalues give them. The verdict comes from the eigenvalues and never
  from the broadened curve. Without `dos_kpts` or `dos_kspacing` the task
  doubles the SCF mesh in each direction. The broadening is a Gaussian of
  width `degauss`, in Ry, defaulting to the SCF's own. With `projected`
  the result file carries one curve per element and angular momentum,
  such as `Si-s` and `Si-p`. The task keeps `{label}-scf.pwo`,
  `{label}-nscf.pwo`, `{label}-dos.dat`, `{label}-dos.out`,
  `{label}-projwfc.out`, and the result file `{label}-dos.json`, and a
  failed step keeps its files under the step's name. It follows
  `band_structure`'s contracts for k-points, the scf pin, the cache, the
  gpu build, and the spin refusals.
- `band_structure` gains `projected=`. With `projected=True` it runs
  `projwfc.x` on the bands step's save directory and returns, per k-point
  and band, the weight of each element and angular momentum, which is the
  data of a fat-band diagram. The result file gains `projection_groups`
  and `projections`, and the projections count against the same inline
  limit as the eigenvalues. `bands_table.py` gains `--projection GROUP`,
  which adds one weight column per band to `--dat` and sizes the markers
  of `--png`.
- `dos.x` and `projwfc.x` are reached through `slab.qe_tools`. Each
  command is the resolved `pw.x` line with the `pw.x` token replaced by
  its sibling in the same directory, so a tool follows the same install,
  the same build, the same launcher, and the same setup lines as `pw.x`.
  Flags only `pw.x` takes are dropped, and `dos_command` or
  `projwfc_command` in `calculator_options` overrides the line.
  `slab.dos` reads what the tools write: the table, the per-state files,
  the state list, and `atomic_proj.xml`. `slab.outputs.digest` gains
  digests for both tools' output and for the `.dat` table, so `read_file`
  and `read_artifact` never print thousands of rows.
- The new `density-of-states` skill covers the procedure, the mesh and
  broadening trade-off, the limits of the projections, and the reporting
  rules, and its `dos_table.py` writes a `-dos.json` as a table or a
  plot. The band-structure skill gains a projected-bands section.
- `band_structure` takes its k-point path from seekpath. seekpath finds
  the space group with spglib, builds the standardized primitive cell,
  and gives the recommended path in that cell's reciprocal basis, so the
  task runs both pw.x steps on that cell. A conventional cell or a
  supercell of a perfect crystal reduces to the primitive cell. When the
  lattice changes, the task replaces an explicit `kpts` mesh with
  `slab.bands.matching_mesh`, which is at least as dense, and
  `info["scf_kpts"]` names it. `info` and the result file gain
  `spacegroup`, `spacegroup_number`, `n_atoms_input`, `cell_changed`, the
  primitive cell, and `seekpath_labels`. A `path=` uses seekpath's labels
  with `G` for GAMMA and the underscore dropped. The task gains
  `symprec=`. `seekpath` is a new dependency, and it brings `spglib`. The
  Si and Al fixtures are new real pw.x 7.5 runs on the seekpath path.
- A Quantum ESPRESSO band structure is one traced task.
  `band_structure(atoms, calculator_options=..., path=, npoints=,
  density=, nbands=, label=)` in `foundation.tasks` runs an SCF and then
  `calculation='bands'` in one scratch directory, with the same prefix
  and outdir so the second step reads the first's charge density. It
  returns the bands and a gap verdict: `is_metal`, `gap`, `direct_gap`,
  `gap_kind`, the band edges, and where on the path they sit. The verdict
  reads band crossings against the SCF Fermi level, so smearing on an
  insulator does not change it. Under fixed occupations it counts the
  occupied bands. The task keeps `{label}-scf.pwo`,
  `{label}-bands.pwo`, and the result file `{label}-bands.json`. A failed
  step keeps its files as `{label}-scf-failed.*` or
  `{label}-bands-failed.*`, and a note names the step. It follows
  `single_point`'s contracts for k-points, the scf pin, the cache, and the
  gpu build, and it refuses a non-QE engine, `nspin=2`, `noncolin`,
  `lspinorb`, and an `nbands` below the occupied bands. The new
  `slab.bands` module holds the path, the reader, the verdict, and the x
  axis as pure functions, and `slab.backends.engine_scratch` gives a task
  a marked scratch directory for several engine calls. `read_file` and
  `read_artifact` digest a bands output by its k-point and band counts
  and its energy range, without the eigenvalue rows. The new
  band-structure skill (dft-expert) gives the procedure and the reporting
  rules, and its `bands_table.py` prints the verdict, writes a table, and
  plots when matplotlib is installed. The test fixtures are real pw.x 7.5
  runs on Si and Al.
- A Materials Project snapshot may carry a second id column,
  `material_id_numeric`. When it does, `get_material`, `fetch_structure`,
  and `slab mp show` accept either id and resolve it to the canonical
  `material_id`, which every record, run, and cache key uses. A numeric id
  that two rows carry is refused. `slab mp info`, `slab doctor`, and the
  `list_engines` overview report `numeric ids`. SLAB never writes the
  column.
- Quantum ESPRESSO gains a gpu build, as LAMMPS has. `[engines.qe.gpu]`
  names a GPU-enabled `pw.x` with the same keys as `[engines.qe]`: `bin`
  (SLAB constructs `mpirun -np {ntasks} <bin>/pw.x`) or `command`, never
  both, plus `setup` lines. A launch whose reservation holds gpus runs
  it, and an unsized launch runs the plain build from `[engines.qe]`. The
  sandbox render binds a gpu `bin` install read-only. The gpu build refuses more MPI ranks
  than gpus before `pw.x` starts, and its cache identity carries
  `build: gpu`. `list_engines` (Mason and MCP) lists a `qe` entry with
  both builds, `slab engines list` and `slab engines show qe` print them,
  `slab doctor` checks the gpu build's launcher, and the sandbox render
  snapshots the gpu build as `qe.gpu`. A command that asks for `{gpus}`
  on a launch without one now names the QE build in its refusal, where
  it named LAMMPS before.

- A specialist may hand a script to a helper card, one level further
  down. The new card flag `helper: true` marks a card that takes briefs
  and never delegates, so the tree is at most lead, specialist, helper.
  The first built-in helper is `coding-expert`, which writes, fixes, and
  checks scripts and inputs and returns the file with the evidence it
  ran. A specialist briefed by a lead gets a `delegate` tool whose team
  is the helpers only, one brief at a time, and a helper gets none at any
  depth. A lead's team takes the helper too. Each helper brief counts
  against the specialist's own: the helper runs at most the calls the
  brief has left, and one turn sends at most `[agent] helper_briefs` of
  them (default 3, and a roster table may raise it). The lead reads one
  `[harness] helper coding-expert-1: 7 calls, finished` line per helper
  brief under the specialist's report. A helper's transcript is
  `<stem>-<specialist handle>-<helper>-<n>.jsonl`, and every delegated
  transcript now opens with a `session` header naming its `agent` and
  its `parent`. `slab mason report` prints each helper under its
  specialist, and `slab mason roster` tags it `[helper]`. The
  `delegation` switch turns both depths off. md-expert, dft-expert,
  analysis-expert, and worker brief coding-expert when the same script
  fails the same way twice, or when a script the brief needs does not
  exist.

- The loop refuses a finish that cites no verified run once, and the
  evidence a lead needs is reachable without a worker's shell. Under `check-gating`,
  a lead's `finish` whose cited runs hold none in a passing state comes
  back naming each run and how it stands, for example `01m2hy5ygy
  running, 01m2hvw3ca quarantined 3/5 checks`, and a cited id that
  matches no run is named the same way. The refusal says that a campaign
  is scored on verified runs and that a traced analysis workflow over the
  evidence files produces one. The identical finish after it stands, and
  its transcript event carries `unverified: true`, which the scorer
  copies into the record and adds to its failure line. A table can then
  count a campaign that was told and finished anyway apart from one whose
  runs failed unseen. A specialist's finish and a finish that cites
  nothing are untouched.

- `read_artifact` reads a run's live files. When a name is no registered
  artifact of the run, the tool looks in that run's scratch directory, so
  a running run's LAMMPS log reads through the same digests, headed
  `<name> (live file of run <id>, <n> bytes so far, not an artifact)`.
  The files of a run that died before it registered anything read the
  same way until the reap or the purge that follows removes its scratch,
  because the scratch sweep is unchanged. A registered artifact of the
  same name still wins. `show_run` on a running run lists those files
  under `live_files`. Nothing is copied into the artifact store.
  `read_file` and `list_dir` stay fenced out of the scratch: a run id is
  the only way in.

- A `dry-<stamp>` id goes in `read_artifact`'s `hash` as well as its
  `run_id`, and without a `name` it lists the files that record holds.
  `show_run` on a dry id returns the record itself.

- A machine memory rests on a run that finished, and a fact that is only
  true today expires. `remember` counts a piece of evidence only when it
  names a run in the completed status. A run that is still going, a run
  the workspace does not hold, a failed run, and a dry-run record id are
  kept in the text and mark the memory unverified, and the reply names
  which cited ids counted and which did not. `remember` also takes
  `kind`: `build` for how this machine's software behaves, `resource` for
  what its hardware does, and `outage` for something that is broken now.
  An outage carries `where`, the host its evidence ran on, taken from
  the first completed run cited or else the first cited run with a host
  stamp, and `expires_at`, a week out by default. `recall` puts that line in front
  of the fact, the prompt catalog drops an outage after its day, and
  `slab memory review` lists it for deletion. Nothing deletes a memory
  except the person. A memory that restates what a bundled skill
  documents is refused with the skill section that holds the real answer;
  the table of subjects is `foundation.documented`, and it covers the
  `cna/atom` codes, `dilate`, `fix_modify energy`, `velocity create` and
  `scale`, and `fix halt`. The list a delegate hands its lead carries
  each memory's kind, the state of its evidence runs, and an outage's
  host and expiry, so the lead's forget decision needs no `recall` call.
  `slab memory review` re-judges the evidence of every verified memory
  against a workspace's runs, which is how the rule reaches the memories
  written before it, and a memory with no kind reads as `build`. The
  same rules reach the MCP `remember` and `recall`, and `slab memory add`
  gained `--kind`, `--where`, and `--expires`. `slab memory confirm`
  stamps the memory with `confirmed`, the day a person checked it, so
  the review does not judge that memory's evidence against the runs
  again; it warns when the evidence names no completed run, and it
  moves an outage's expiry a week out, or to `--expires`. `slab memory
  review` finds the workspace as every other command does.

- The two-phase-melting skill states what a hot crystal reads. A new
  section names the deficit: a crystal near T_m classifies a large share
  of its atoms as unknown under an instantaneous order parameter, by an
  amount no one can predict, so a gate against a cold count condemns a
  crystal that is intact. The skill prescribes time-averaged positions
  before `compute cna/atom`, or `compute ptm/atom`, with a crystal and a
  liquid baseline from the pure-phase legs at the run's own temperature,
  and it restates every gate as the calibrated fraction against those
  two baselines. The opening now puts the NPH plateau route first for a
  cell under ten thousand atoms, three starting enthalpies with one run
  each. The velocity ladder stays the route for a large cell and for a
  v(T) table. A table gives each route its cell size, its runs, and its
  steps, so a lead can size a wave against the time a job has left. The
  build recipe gained the traps the last campaign hit. Assemble each
  phase from its own equilibrated leg, and minimise with the crystal
  frozen after the soft-repulsion push-off. Leave no gap at the periodic
  wrap. Write `dilate all`, because a group dilate crashes under KOKKOS.
  Never barostat z to zero pressure on a cell with a free liquid
  surface. The new script `coexistence_fraction.py` enforces the
  cross-section bound. It refuses a cross section under eight unit cells
  unless `--small-cell` is passed, which prints the finite-size caveat
  line the report must carry. Its `--plateau` mode reads a NPH run's
  thermo YAML and its fraction series. It prints the primary window mean
  and the means of the window's two disjoint halves, each with its block
  standard error, then the drift across the primary window with the
  error of the fitted slope, and the two-phase verdict. The halves agree
  only within two combined errors. The script also repeats the block
  error over 16, 8, and 4 blocks and calls it converged only when the
  last ratio is under 1.2. An error that still grows is a lower bound,
  and the drift in errors is then an upper bound. One real run of a
  5120-atom coexistence cell is bundled with it, read at 70 ps and the
  same run continued to 200 ps. Neither window passes the drift gate,
  and their means agree at 1442.6 and 1440.4 K. The skill reads that
  pair as the finite-size wander it is.
  The `md-expert` and `planner` cards point at the new section, and a
  brief for a melting step now names the route and the order parameter.
- A numeric table is for a script, and a cut reply costs one cheap retry.
  A read that returns more than `[agent] table_nudge_rows` rows of
  numbers (default 20) ends with one harness line: a table this long is
  for a script, not for reading. `read_artifact` carries it for a JSON
  table or an artifact with a `.dat`, `.csv`, or `.yaml` name.
  `read_file` carries it for a file with one of those suffixes. A card
  with no shell, such as the planner, reads the variant that sends it to
  its specialist. The line is the `table-nudge` mechanism, so the
  benchmark can ablate it.
  The one retry after a reply cut with no text now runs under half the
  reply ceiling as well as at low effort. So a second failure costs half
  of what the first did. The partial outcome a cut specialist hands back
  counts its cuts, and says when the last one ran under half the ceiling.
  A `cut` event records the completion tokens the ceiling discarded, the
  ceiling itself, and the tool the model read last. `slab mason read`
  prints the tokens and the tool. `slab mason report` sums the tokens
  lost and names the tool most of the cuts followed.
  `[agent.roster.<name>]` already overrode `max_reply_tokens`,
  so a specialist that reads data can run under a lower ceiling than its
  lead. The md, dft, and analysis cards now say to compute a table's
  statistic with a script and never to reason over the rows.
  Benchmark-4 session 1 hit the ceiling seventeen times and lost 544,000
  completion tokens, 28 % of everything it generated, and fourteen of
  the fifteen cut events followed a raw read of a `fix ave/time` table.

- Every run belongs to a session lease, and every reader settles the runs
  of a dead lease before it reports an active one. A Mason session and an
  MCP server each write one `sessions` row when they start (schema 10),
  stamp it every `[agent] lease_beat_s` seconds, at every step, and at
  every `wait_for_run` poll, and close it when they end. A reader draws
  three new verdicts from the row. `session-ended` means the session
  closed its lease. `deadline-passed` means the job that held the lease
  is over; the lease carries that moment from `$SLAB_JOB_END` or
  `$SLURM_JOB_END_TIME`, both epoch seconds. `session-silent` means no
  beat for `[workspace] lease_silence_s`, default 600 s. `reap_dead`
  fails those runs, releases their reservations, and sweeps their
  scratch. It does so from any host and in any job, with no scheduler to
  ask. A beating lease keeps its runs, and a run with no lease row is
  judged exactly as before. The scheduler still outranks a lease that
  beats from a job it calls ended.
  A session that ends also ends the runs it was still executing, with a
  TERM to each process group and the error line naming the session and
  the reason, so no session leaves a run behind it. The loop closes the
  lease on finish, on the error path, and on SIGTERM, and records a
  `session_end` event that `slab mason report` prints. The session reads
  its job's clock. The environment block says when the job ends and how
  long is left, the per-step harness line repeats the minutes under an
  hour, a wait is capped at the job's end less the signal grace, and the
  `pi` and `planner` cards size the last wave to what is left. The
  rendered sandbox job carries `#SBATCH --signal=B:TERM@180` and exports
  `SLAB_JOB_END`, converted on the host from `squeue`'s local stamp to
  epoch seconds. The batch shell runs the container in the background,
  forwards the TERM to it, waits for it, and then runs `slab sessions end
  --job`, the fallback for a container that dies before the session can
  close its own lease. `slab sessions` is now a group. `list` shows every
  lease with its harness, agent, job, beat, deadline, and the runs it
  still has running, `end` closes one lease or one job's leases by hand,
  and `sweep` settles what is over. `slab doctor` gains a `leases` row,
  and `slab purge` settles ended leases as it settles ended jobs.

- A lead continues a specialist it already briefed, and sizes each brief.
  `delegate` takes `continues`, the handle from an earlier report's
  harness line, and gives that specialist another turn with its messages
  intact, so the follow-up costs the reading once. The handle is the
  agent name and the ordinal of the brief that made it, and a specialist
  keeps one transcript across its turns, each marked with a `turn` event.
  A continue rebuilds the specialist's system message first, so the
  notebook the lead wrote between the briefs is in the prompt; after
  `slab mason chat --resume` it replays the specialist's transcript
  instead. An unknown handle, a handle of another agent, and a critic's
  handle are refused, the first with the handles that exist. `delegate`
  also takes `steps` and `effort`, the budget of one brief. Both only
  lower the agent's own budget, a CLI flag outranks both with a harness
  note, and the harness line says which budget stopped the turn, `turn
  budget (8, set by the brief)` against `turn budget (60)`. `slab mason
  report` gives a continued specialist one row with its turn count and
  counts the briefs stopped at a budget the lead set. The `pi` and
  `planner` cards gained the doctrine for both. A wave briefs fresh specialists, so a handle in a
  `delegate_many` brief is refused and sent to `delegate`.

- A lead hands out a wave of independent briefs. The new Mason tool
  `delegate_many(briefs)` takes two or more `{agent, task, context?}`
  briefs and runs each specialist's loop at the same time, in threads
  inside the lead's process and under the lead's session lock. Every
  child session is created before the first thread starts, so the
  ordinals and the transcript names follow brief order. The result
  carries one section per brief, each with the report, the memories that
  brief wrote, and the same bracketed harness line `delegate` returns,
  and a last line stating the wave's wall-clock against what the briefs
  would have cost in sequence. Each brief keeps its own share of the
  tool-result cap, so no section is dropped whole. One brief that fails
  leaves its siblings' reports intact. `delegate` is unchanged and is where a dependent step
  goes. `[agent] parallel_delegations` caps a wave and defaults to 3;
  1 removes the tool, as does the new `parallel-delegation` mechanism
  switch. The token counters, the memory list, the setup digests, the
  notebook, the approval prompt, and the terminal output are each
  guarded by a lock on the session tree, so a wave writes them one
  writer at a time. A person's interrupt stops every child at its next
  step, and the wave is recorded with what came back. Each `delegate`
  event of a wave carries `wave`, `parallel`, and the two spans, and
  `slab mason report` counts the waves and the wall-clock saved. The PI
  and planner cards gained the rule: a wave is briefs that share no file
  and no run, sized together to what is free.

- A cut reply keeps its design. When a reply is cut with no text and no
  call, the loop shows the model its reasoning for one call and asks for
  the design decisions in the notebook. The design call and the request
  for a short answer that follows the notebook call both run at low
  effort. A card without the notebook,
  or a cut with no reasoning, gets the short-answer request as before.
  The loop also cuts a reasoning loop early. Each model call streams
  under the `continue-cut-reply` switch, and when one 200-character
  passage of the reasoning appears three times, the client closes the
  stream there (finish reason `reasoning_loop`). The `cut` event records
  `design` and the passage's first line as `loop`, and
  `slab mason read` prints both. The sandbox bridge relays a streamed
  answer as it arrives, and the Anthropic client marks a streamed call
  whose input never closed as cut inside its arguments.

- A machine memory carries its evidence. `remember` (Mason and MCP) and
  the new `slab memory add` take `evidence`: a run id with one line, a
  dry run, or a failure record. The store refuses a write without it
  unless the writer passes `unverified`, which stamps `unverified: true`.
  A memory with no evidence reads as unverified, so every memory written
  before this change is flagged. The catalog line ends with
  `[unverified]`, and `recall` prints that word first. `recall` also
  reads the state of each run the evidence cites. Re-using a name keeps
  the replaced file under `.history/<name>/`, up to ten versions, and
  `recall` shows the previous body with its date when the body changed.
  A delegate result ends with a harness-built list of every memory the
  delegate wrote, with its evidence and the state of each cited run. The
  planner and PI cards tell the lead to read each entry and forget any
  the evidence does not support. The new Mason `forget` tool undoes
  only a write made in the same session: it removes a new memory, or
  restores the version from before the session. `slab memory review`
  lists the unverified memories and those whose software changed, and
  `slab memory confirm <name> --evidence ...` records what a person
  checked.

- The planner reads evidence, sizes for the machine, and inherits the
  notebook. The planner card keeps `read_artifact` and `read_file`, so
  it reads a run's averages table itself. It still launches nothing.
  Each request it receives ends with the free cpus and gpus at that
  step. The card and the environment's resource line state the wave
  rule, so a wave has as many concurrent launches as free GPUs and one
  wait. A card that writes the plan sees the notebook's earlier
  entries for this project with their dates (`mason.prior`), and the
  `plan` tool refuses a plan whose Goal names a quantity the notebook
  reports until it carries a line `prior result: ...`. A match by one
  shared word is a note, not a refusal. Every numeric gate in a brief
  names its source, and the critic lists a gate without one as an
  advisory finding. The `plan` tool checks each `run:<id>/<name>`
  reference against the run store and rewrites one to a cache-hit run
  to the run that produced the file (`foundation._ops.artifact_holder`,
  `RunStore.find_producing_task`). `delegate` rewrites a brief the same
  way.

- Tool results stop carrying environment dumps and session-wide
  listings. `list_engines` shows each LAMMPS build's setup block as one
  line with its line count and sha256 prefix, and keeps every build's
  command whole. When the listing passes the output cap, the rootstock
  checkpoint ids fold to counts first. `slab engines show lammps
  --setup` prints the lines. `show_run` folds the same blocks in a task
  record, and `show_run task=<n> setup=true` prints them. The first
  engine command event of a conversation keeps a setup block with its
  digest (`setup_digest`, `setup_lines`), later events name the digest
  only, and `slab mason read --full` expands them. The MCP `list_engines`
  and `show_run` fold alike (`setup=True` returns the lines), and its
  session record keeps each block once. `wait_for_run` names the run
  that finished and counts the session's others, and a still-running
  answer names at most five runs and stays under 2 KB. Clearing and
  compaction never take a tool result that no complete reply has read.
  `read_artifact` reads a JSON table as rows, with `columns=`, `every=`,
  and `table=`, and `every=` thins a text artifact's lines.

- Artifacts follow the cache, and a run's artifact can feed the next
  task. `read_artifact` on a run whose task was a cache hit reads the
  file from the run where the task executed, and the first line of the
  answer says so. `show_run` gives every cache-hit task `artifacts_on`,
  the id of that run. `run_lammps(files=)` takes `run:<id>/<name>` and
  `run:<id>/<name> as <basename>`, which stage a run's artifact beside
  the script with no copy through the shell. The reference follows
  cache hits too, and it enters the cache identity through the
  artifact's hash and basename, not the run id (the tracer's new
  `canonical=` hook), while the recipe records the run under
  `provenance`. A dry run reads the reference from the real workspace.
  `read_artifact(hash=)` reads any bytes the workspace holds by a
  SHA-256 prefix, a task's input or output value included, and names
  the runs and tasks that reference them. MCP gains `read_artifact`
  with the same arguments. See the new [Artifacts](docs/tutorials/artifacts.md)
  page.

- The averages summary cannot be mistaken for the rows. Every entry of
  `run_lammps`'s `result["tables"]` and `result["averages"]` is a
  `foundation.tasks.TableSummary`. Reading `rows`, `data`, or `values`
  from it raises a `KeyError` that names the artifact and the call, for
  example `rows live in the md-averages.json artifact; call
  series(result, 'fraction.dat')`. A check that read rows from the
  summary got an empty list and quarantined two three-hour runs, so it
  now fails at once. Each summary carries its call under `series`, and
  a thermo table's summary says whether a `minimize` printed it.
  `series(result, "production")` returns the longest table, not a
  minimize one, whose Step range covers the table the last `run`
  printed, so a leading `minimize` or a trailing `run 0` no longer
  hides the production run. The summaries stay plain JSON in the
  store, and `task` gains `on_return` so a cache hit gets the same
  refusal. `slab.outputs.lammps_thermo` marks a table `minimize` from
  the log's `Minimization stats:` block. The lammps-scripting,
  two-phase-melting, and lammps-potentials skills send the rows through
  `series`.

- A failure record carries the cause, and a failed dry run's LAMMPS
  files can be read. `run_lammps` now takes `Kokkos ERROR`,
  `terminate called`, `what():`, `error while loading shared
  libraries`, `Segmentation fault`, and `MPI_ABORT` lines as error
  lines, and a line with one of them fails the script as an `ERROR`
  line did. A `[warn] Epoll` line is not an error line. When the exit
  code is not zero and no line matches, the error message holds the
  last 30 lines of the screen capture under a `screen tail:` label. A
  dry run copies the `{label}-failed` files of each failed `run_lammps`
  call into a dry-run record, `<workspace>/dry-runs/<stamp>/`, before
  its throwaway workspace is removed. The report names it under
  `record`, and the Mason `read_artifact` tool opens its files by the
  run id `dry-<stamp>`. `slab purge` removes the records and keeps the
  newest conversation's records unless `--all-sessions` is given. The
  Mason dry-run reply shows the traceback once. When the output ends in
  the JSON report, the reply keeps the last frame and the exception.

- GPU launch sizing leaves room for the next launch, and an unsized
  launch holds what it says. A reservation sized with `gpus=` and no
  rank or thread count gives each rank its gpu's share of the free cpus
  (the free cpus divided by the free gpus), where it took every free cpu
  before. So four `gpus=1` launches fit side by side on a 36-cpu, 4-gpu
  job with 9 threads each. A launch that names `threads` keeps them.
  Where the budget holds gpus, an unsized launch holds one rank of the
  default thread count and no gpu. Mason and the MCP server run it as a
  child (`foundation._ops.runs_as_child`), so it sees no gpu and runs
  the plain build instead of a gpu build sized to every rank of the job.
  A budget without gpus keeps the old unsized rule. `[engines.lammps]
  requires_gpu = true` declares a plain build that cannot start without
  a GPU. A launch that holds no gpu is then refused before LAMMPS starts,
  dry runs included, with the advice to size it with `gpus=1`.
  `list_engines` and `slab engines list` show the flag. `slab doctor`
  warns when the plain command asks KOKKOS for a gpu later in its option
  list (`-k on t 4 g 1`), and when the flag is unset and `ldd` finds
  `libcudart` in the plain binary (`slab.lammps.links_cuda_runtime`).
  The launch tool text, the environment block, the sandbox context, the
  lammps-scripting and lammps-potentials skills, the md-expert card, and
  the LAMMPS note state the rule, and the dry-run text says to size the
  rehearsal like the launch.

- A known-bad GPU is kept out of the budget, by config and by refusal.
  `SLAB_GPU_EXCLUDE` (comma-separated ids in the budget's numbering)
  removes ids from `slab.resources.budget`, and `gpu_source` says how
  many went (`cuda_visible_devices, 1 excluded`). A partition's
  `exclude_gpus` makes every job rendered for it export the variable,
  and the sandbox passes it into the container and names the devices in
  its context. `[workspace] exclude_gpus` serves a workstation, exported
  by `slab run`, `slab mcp`, and the Mason session unless the
  environment already sets it. When LAMMPS fails with
  `cudaErrorDevicesUnavailable` within 60 s of its start, on a launch
  with no more ranks than gpus and one gpu or a named ordinal, the run
  store records the device in the new `excluded_gpus` table (schema 9).
  While the row exists, reservations on that host under that job skip
  the id, `free_resources` lists it as `gpu 0: excluded (refused at
  HH:MM, job N)`, and the failure record says so. The row goes when the
  job ends (`slab purge`, `slab hpc cancel`, a reap that finds the job
  ended). `slab runs gpus` lists the rows, and `slab runs gpus --clear
  ID` removes one. An engine command or setup line that sets
  `CUDA_VISIBLE_DEVICES` is refused before launch, because the
  reservation chooses the device. `info["kokkos"]["devices"]` records
  the gpu ids a launch held, and after a refusal Mason suggests a
  `remember` with the device id and the job but never writes one.

- A failed check says what it saw, and a check can be run again
  without a relaunch. A failed check that names no observed value (a
  bare `False`, an `assert`, a raise) stores `evidence`: its source
  (at most 40 lines), the top-level keys of each dict it reads by
  name, and for a raise the exception and the line that raised it.
  `slab show`, `show_run`, and `run_details` carry it. `slab runs
  reverify <run> <script>`, the Mason and MCP `reverify_run` tools,
  and `foundation._ops.reverify_run` run a script's checks on a
  quarantined run's stored results. Every task call takes the run's
  own result (`foundation.runtime.Replay`), so no engine starts and no
  new run is recorded, and the checks become a new verification pass
  on the same run. Schema 8 adds `pass_no` and `evidence` to checks;
  `list_check_results` returns the latest pass, and `all_passes=True`
  returns all. A task call whose inputs differ from the run's is
  refused with `ReplayError`, naming the task. `launch_workflow` with
  `dry_run` and `from_run` (`slab run --dry-run --from-run <run>`)
  rehearses a script on a run's cached results, so the checks run on
  real data. In a dry-run report each check has a `reading`: a raise
  reads `check raised <Exception>: <text>` and carries its line and
  keys, a pass on a zero-step result reads `passed on no data; not
  evidence`, and a raise makes the rehearsal not clean. The report's
  `checks_note` is gone.

- The `setup=` contract of `run_lammps` is stated and correct. A string
  runs as one line per newline, with blank lines dropped. Before, the
  task iterated it per character, so `"set -e\nexport ..."` failed with
  `e: command not found`. One helper (`slab.lammps.setup_lines`) does
  this for the task, the runner, and the cache identity. Per-call lines
  now run after the build's own lines by default (`setup_mode="extend"`),
  so a call cannot drop the build's module environment by accident.
  `setup_mode="replace"` runs the per-call lines alone. The mode enters
  the cache identity, and `info` records the build's lines, the call's
  lines, and the mode. `slab doctor` gains one row per LAMMPS build. The
  row runs the build's setup, then `command -v` on the first word of the
  command, and fails when the launcher is not on PATH, because the
  launch-time checks look through `mpirun` to `lmp` and never find a
  bare `mpirun` that the setup does not provide.
- Waiting on a long run costs one call and never reads as a stall.
  `wait_for_run` blocks for up to 6 hours, where it stopped at 30
  minutes without saying so, and an answer cut by the cap says
  `waited N s, capped from the M s asked`. It reads the run store after
  1 s, then at doubling gaps up to 30 s. Without a run id it returns
  when the first running run of the session finishes, names it, and
  lists the rest; `all=true` keeps the old wait for every run. A
  still-running answer adds the time since the run started and, from
  the live log of a `run_lammps` task, the step LAMMPS has reached
  against the end of the current `run` (`slab.outputs.lammps_run_progress`,
  `foundation._ops.run_advance`). It ends with a line saying that
  waiting again is the right call. The loop never appends the repeat
  note to that answer, and such a step does not count toward the
  step-back hint. A wait records the engine commands of the runs that
  finished during it only, a resumed session skips runs its transcript
  holds, and a setup block is recorded once per transcript
  (`setup_digest` on a later command). The MCP `wait_for_run` has
  the same cap, the same first-finish rule, and the same fields.
- `slab mason read --live` follows a session as it works, like
  `tail -f`. The viewer shows the transcript, then each event the
  session appends, until Ctrl+C. It follows the transcripts of the
  session's delegations too, and a line names the transcript whenever
  the output moves between them. A line still being written waits for
  its newline, so a half-written event is never marked invalid.
- A run is live only while its job is. `run_liveness` adds two
  verdicts: `other-job`, a run stamped with a scheduler job that is not
  this process's, whose pid means nothing here because a sandbox job
  has its own PID namespace, and `job-ended`, a run whose job the
  scheduler reports terminal. `reap_dead` resolves each job once per
  sweep and fails the runs of ended jobs (`job N is cancelled; marked
  failed by <caller>`), so every reader that reaps closes them and
  their session records turn stale. A reservation belongs to a job
  (schema 7 adds `job_id` to reservations): a slice from another
  allocation is never live for this budget. `slab purge` marks failed
  the running runs of every job the scheduler reports ended before it
  deletes anything (`Workspace.settle_ended_jobs`, no pid consulted), and
  `slab purge --job ID` takes a job the scheduler cannot place as ended.
  It refuses a job still in the queue.
- A GPU launch gets one MPI rank per GPU unless told otherwise. A
  reservation sized with `gpus=` and no rank count takes one rank per
  gpu and the free cpus as threads, where it took the job's rank count
  before and put every rank on one device. The gpu build refuses a
  launch with more ranks than gpus before LAMMPS starts
  (`slab.resources.one_rank_per_gpu`, a `slab.errors.ResourcesError`
  that Foundation's `ResourcesError` now derives from). A run that dies
  with `cudaErrorDevicesUnavailable` records the rank and gpu counts of
  its launch beside the gpu ids. The lammps-scripting and
  lammps-potentials skills, the md-expert card, the `launch_workflow`
  description, and the sandbox context state the rule.
- `run_lammps` prefers the YAML thermo output of LAMMPS.
  `slab.outputs.lammps_thermo` reads every table a script printed as a
  YAML document (`thermo_modify line yaml`) by schema, and reads the
  text table when the script did not ask for YAML, in log order, so a
  log that switches between the forms parses fully. The log digest
  says `thermo: yaml`, `text`, or `mixed`, and `info["thermo_format"]`
  reports the same. A run parsed from text under a LAMMPS that could
  have printed YAML carries one warning line. The bundled templates and
  the lammps-scripting skill put the yaml line after each
  `thermo_style` line, and `slab doctor` says whether the plain build
  accepts it.
- The gpu budget is the allocation, and a device error is quoted.
  Inside a job `slab.resources.budget` reads `CUDA_VISIBLE_DEVICES`,
  else `SLURM_JOB_GPUS` or `SLURM_STEP_GPUS` (the ids as written when
  the node shows more devices than the job holds, renumbered from zero
  under a cgroup constraint), else a SLURM count, else none; it probes
  `nvidia-smi` only outside a job. `Budget` gains `gpu_source`, which
  `list_engines`, `free_resources`, and `slab doctor` print, and the
  doctor lists each device `nvidia-smi` shows, marking one outside the
  allocation. The sandbox prologue resolves the ids in the same order
  and exports `SLAB_GPU_SOURCE`. `free_resources` adds one line per
  budget gpu with its memory in use. `run_lammps_script` reads the
  screen's error lines when the log holds none, and a Kokkos abort or
  a CUDA error counts as one; a CUDA device error adds a note to the
  failure record naming the gpu ids the launch held, the budget, and
  its source.
- Mason and the MCP server dry-run a script on request and say when a
  launch skipped it. `launch_workflow` takes `dry_run`; the session
  records a `dry_run` event with the script text's digest, and a real
  launch of a text with no passing dry run in the session carries one
  warning line. A `KeyError`, `TypeError`, `IndexError`, or
  `AttributeError` after a completed `run_lammps` adds a note to the
  failure record naming the result's keys. The md-expert card, the
  lammps-scripting skill, and the planner's briefing say a new or
  edited script is dry-run before it is launched. `remember` tells the
  agent when a memory describes SLAB itself, stamps it against
  slab-stack, and `slab memory list` marks it `[about slab]`.
- The `run_lammps` result needs no memory. It carries `seconds`,
  `atoms`, and `rate` from the loop lines, `n_rows` where a count used
  to be called `rows`, `averages` with every `fix ave/time` file parsed
  (`slab.outputs.lammps_ave_time`, also digested on `read_artifact`),
  `label`, and the hashes of the parsed `-thermo.json` and
  `-averages.json` artifacts, which `foundation.tasks.series` reads
  back as rows keyed by column. A script whose output name carries a
  directory component is refused before LAMMPS starts. The
  lammps-scripting skill shows the captured result and the template
  prints the rate.
- A script can be rehearsed before any step is paid for. `slab run
  --dry-run` (and `launch_script(dry_run=True)`,
  `launch_child(dry_run=True)`) runs a workflow script to its end, or to
  its first exception, inside a throwaway workspace that is removed
  afterwards, so nothing lands in the real store or its cache and no
  reservation is claimed. Every `run_lammps` call runs with its loops
  emptied: every `run` line becomes `run 0` and every `minimize` gets
  zero iterations, so every style, fix, and compute is set up, every
  command runs in order, each loop prints one thermo row and one loop
  line, and no step is integrated. A stage-three error and the Python
  after the dynamics both surface at once, and the result keeps its
  real shape. `info["dry_run"]` says the call was a rehearsal and
  `info["rewritten_lines"]` names the original loop lines. The report
  after the `dry run:` line lists each `run_lammps` call with `setup
  ok` or the `ERROR` line, every check with its outcome, the files a
  real run would keep, and the traceback.
- Close contacts are pushed apart before the real potential sees a
  cell. The md-expert card and the atomsk-structures, atomsk-defects,
  atomsk-interfaces, melt-quench, and lammps-scripting skills say to
  check the minimum interatomic distance of every built configuration,
  to rebuild a crystal that has a close contact, and to push the
  contacts of a disordered cell apart under `pair_style soft` with a
  ramped prefactor and `fix nve/limit` before minimizing under the
  real potential. The lammps-scripting skill carries the recipe.
- The Kokkos GRACE styles are named, not derived. The lammps-potentials
  skill states that `-sf kk` cannot turn `pair_style grace` into a
  Kokkos style, because those styles are `grace/1l/kk`, `grace/2l/kk`,
  and `grace/3l/kk`; a run under a gpu build names the `/kk` style and
  passes the exported `.npz`, and `pair_style grace` on the saved model
  is for a machine without a KOKKOS build or a custom architecture the
  export refuses. The skill gains the precision table of the `/kk`,
  `/mixed`, and `/fp32` variants (a 3L model is natively fp32, so
  `grace/3l/kk` is its mixed style), and the mlip-training skill, the
  md-expert card, and the core prompt say the same.
- Dynamics run inside LAMMPS, at every level the agent reads. The core
  prompt every card shares now states the rule: when `list_engines`
  shows `lammps`, every molecular dynamics run goes through `run_lammps`
  as a whole script, a Python dynamics loop under any engine is not a
  route, and a served MLIP checkpoint id is for `relax` and
  `single_point` only. The planner's briefing rule names `run_lammps`
  and the pair style for a dynamics step and treats an ASE-loop result
  as a step to redo, and the md-expert card says the same of a served
  checkpoint. The melt-quench and thermal-response templates, which
  drove NPT from `ase.md`, now run their melt, ramps, holds, and
  ladders inside LAMMPS through `run_lammps` (argon under Lennard-Jones
  as the shakeout, `pair_style grace` a constant away), read the dumps
  and thermo tables back from the run's artifacts, and write the same
  `.traj` files and `ramp.json` the report scripts read. Their tests run
  under `$SLAB_TEST_LMP`, and a new test refuses any bundled template
  that imports `ase.md`. The msd-diffusion skill names the LAMMPS dump
  columns that keep positions unwrapped.
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
  totals, so its text changed. Orphan delegation transcripts,
  unrecognised session files, stale harness records, and stale session
  locks are categories of their own. A submitted job and a launched
  child never inherit the submitter's `SLAB_RUN_ID`. `slab doctor` gains
  a `leftovers` row.
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
