# Changelog

All notable changes to SLAB, newest first. Dates are commit dates on
`main`.

## Unreleased

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
