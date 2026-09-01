# SLAB — Simplest Layer for Atomistic Backends

An agent-native state layer for atomistic materials modeling.

**Documentation: [tarbaugh.github.io/SLAB](https://tarbaugh.github.io/SLAB/)**
has the overview, the tutorials (every code block executed against the real
API), and the architecture document.

**Every run starts as temporary, and it becomes permanent only when you
promote it.** Nothing is stored permanently by default and deleted later.
Workflows are plain imperative Python, and SLAB traces the task graph as the
script runs. Machine-checkable verification hooks decide when a run counts
as verified, and an explicit promotion command is the only action that makes
data permanent. Everything else expires automatically.

<!-- no-verify -->
```python
from foundation import Workspace, task, check, converged
from foundation.tasks import relax

ws = Workspace(".slab")
with ws.start_run(name="si-relax", intent="baseline lattice constant") as run:
    relaxed, info = relax(atoms, engine="emt", fmax=0.05)    # traced, cached, recorded

    @check
    def forces_converged():
        return converged(info["fmax"], below=0.05, label="fmax")

    run.keep("relaxed.xyz", relaxed)                          # declared terminal artifact
# exit: checks evaluated -> quarantined becomes verified; promote it when you decide it matters
```

## Three packages

SLAB is three packages in one distribution, `slab-stack`, behind one
command, `slab`.

| Package | What it gives you |
|---|---|
| `slab` | Access to computational software: engines and calculators, the cluster engine registry, QE protocols, pseudopotential families, and the SLURM layer. |
| `foundation` | Workflows and state: runs, artifacts, caching, verification, retention, and the MCP server. |
| `mason` | The resident research agent. |

`mason` depends on `foundation` and `slab`, `foundation` depends on `slab`,
and `slab` depends on neither. So you can drive an engine without a
workspace, and keep a workspace without an agent.

## Why

Existing materials workflow engines (AiiDA, atomate2/jobflow, pyiron,
FireWorks) were designed for human experts. Three of their costs are fatal
for agentic use:

1. **Provenance totality.** Every intermediate is stored forever as an
   immutable graph node, including retrieved wavefunctions, trajectories,
   and charge densities that nobody ever reads. Deletion feels like surgery
   instead of housekeeping.
2. **Declaration-time epistemics.** You must decide whether a run is
   "production" at submission time, but that information only exists at
   completion time, after you see convergence and output sanity. So debug
   work runs in the production profile, and the archive fills with failed
   attempts.
3. **Ceremony-heavy APIs.** WorkChain-style declarative process classes are
   hard for trained humans to read, and they are token-expensive for LLM
   agents to generate and re-read while debugging.

SLAB inverts all three. The full argument is in
[ARCHITECTURE.md](ARCHITECTURE.md).

## The lifecycle

```
quarantined ──checks pass──▶ verified ──promote──▶ promoted ──archive──▶ archived
    │  │                        │
    │  └────────force-promote───┘ (recorded)
    │ ttl                       │ ttl
    ▼                           ▼
 expired ◀──────────────────────┘
```

Two rules are structural, not policy. Promoted data cannot expire, because
the transition does not exist and a retention policy with a TTL on
`promoted` fails validation. Unpromoted data expires automatically.

Retention is tiered by artifact role, not by data type. Promoted runs keep
full bytes for **terminal** artifacts and **input** roots, so
recompute-on-demand is a real promise, while **intermediate** bytes are
hash-and-discarded. Their content hash and the complete recipe (inputs, code
version, engine versions, parameters) stay on the run forever.

## Install

```bash
pip install -e .              # core: pydantic + typer + ase
pip install -e ".[rootstock]" # + cluster-served MLIPs (thin client, no torch)
pip install -e ".[mcp]"       # + MCP server for agents
pip install -e ".[dev]"       # tests, lint, types
```

One install brings all three packages and all three commands. Python ≥
3.11. There is no daemon, no database server, and no configuration. A
workspace is a directory (`.slab/` by default) that holds a SQLite file and
a content-addressed store.

## The demo

Relax 5 perturbed variants of a Cu supercell under EMT, promote the best,
expire the rest, and garbage-collect:

```bash
python examples/demo.py                                   # EMT + Cu, no extras, runs in seconds
python examples/demo.py --engine mace-mp-0-medium         # a served MLIP checkpoint through rootstock
```

The output:

```text
workspace: .slab   engine: emt   system: Cu x 32 atoms
  run 01m15d2fv9  E = -0.213146 eV  fmax = 0.0407  steps =  6  -> verified
  run 01m15d2fwh  E = -0.213732 eV  fmax = 0.0439  steps = 10  -> verified
  run 01m15d2fxn  E = -0.214059 eV  fmax = 0.0315  steps = 13  -> verified
  run 01m15d2fyz  E = -0.213770 eV  fmax = 0.0419  steps = 13  -> verified
  run 01m15d2g09  E = -0.214043 eV  fmax = 0.0301  steps = 14  -> verified

lowest energy: run 01m15d2fxn  (E = -0.214059 eV)
decide what deserves permanence, then clean up:
  foundation promote 01m15d2fxn --reason 'lowest energy of 5 variants'
  foundation expire --older-than 0d
  foundation gc
```

Each variant is its own run with a stated intent, and the convergence checks
passed, so all five landed `verified`. Nothing is permanent yet. That
decision happens now, after the results exist:

```bash
$ foundation promote 01m0v7tefx --reason "lowest energy of 5 variants"
promoted 01m0v7tefx69nvg32fs2tjdjze  si-relax-0

$ foundation expire --older-than 0d
expired 01m0v7tph3bwc3dwtya44tb0f4  si-relax-4
expired 01m0v7tmg4gj4af1b8yk9t3ty3  si-relax-3
expired 01m0v7tjhzacze4rw6w52h4k6k  si-relax-2
expired 01m0v7th3146ysdd274zb5b8sp  si-relax-1
4 run(s) expired

$ foundation gc
dropped 31 blob(s), freeing 444335 bytes; 8 kept
```

The archive now contains exactly one run's terminal artifacts, plus its
recompute roots, while everything else is a hash-and-recipe skeleton that
remains fully queryable. The two `show` views below are condensed side by
side; the values come from those runs:

<!-- no-verify -->
```text
$ slab show 01m0v7tefx          $ slab show 01m0v7th31   # expired
  state:   promoted                           state:   expired
  checks:  3/3 passed                         checks:  3/3 passed
  artifacts:                                  artifacts:
    variant-0.traj  ...  hash-only              variant-1.traj  ...  hash-only
    relaxed.xyz     ...  bytes                  relaxed.xyz     ...  hash-only
    result          ...  bytes                  result          ...  hash-only
```

## CLI

| Verb | What it does |
| --- | --- |
| `slab run script.py` | Execute a zero-ceremony workflow script inside a traced run (lands in quarantine). Scripts that manage their own runs are executed with plain `python`. |
| `slab list [--state S] [--status S] [--session S] [-q]` | List runs, newest first. |
| `slab show <id> [--json]` | One run: state, intent, checks, tasks, artifacts, history. Ids accept unique prefixes, git-style. |
| `slab promote <id>... [--reason ...] [--force]` | Make runs permanent. `--force` promotes an unverified run and is recorded as forced. |
| `slab promote --session <id> [--force]` | Promote every run one agent session created, reporting each outcome. Failed runs are never promoted this way. |
| `slab sessions` | List the sessions that created runs, with run counts and state breakdowns. |
| `slab expire [--older-than 30d] [--include-running]` | Expire unpromoted runs past their TTL (state change only). `0d` = everything unpromoted, now. Runs at status `running` are protected unless `--include-running` (for hard-killed processes that can never advance their own status; they are marked failed first). |
| `slab gc [--dry-run]` | Drop artifact bytes no retention rule demands. |
| `slab engines list` / `slab engines verify` | Inspect / smoke-test the cluster engine registry. |
| `slab mcp` | Serve the workspace to agents over MCP (stdio). |

Workspace resolution: `-w/--workspace` flag > `$SLAB_WORKSPACE` > `./.slab`.
Retention policy: `--policy file.json` > `<workspace>/policy.json` > defaults.

## Failure is evidence, not a status

AiiDA-style engines handle errors through predefined protocols, such as
exit codes and automated restart handlers. SLAB's user is an LLM agent, and
such an agent can devise a niche correction, such as a smaller
perturbation, a different engine, or a looser threshold, if it can see what
actually happened. So SLAB's contract is evidence delivery, not an error
protocol:

- Failed runs and tasks carry a structured `failure` record that holds the
  exception type, the message, and a traceback that keeps the entry point
  and the failure site but elides deep middles. It is information-rich and
  token-bounded.
- Tasks annotate their exceptions with diagnostics via plain
  `Exception.add_note`, so `relax` notes the completed step count and the
  last trajectory frame's energy and residual force. The notes land in the
  record, listed separately, so an agent can act without parsing the
  traceback.
- The scratch data that explains a failure survives it, so a crash in the
  middle of an optimization keeps the partial trajectory as a
  `relax-failed.traj` artifact. Retention makes this free, because failed
  runs sit in quarantine with a TTL, so diagnostics self-clean instead of
  accumulating forever.
- Checks store the `observed` and `expected` values their assertions
  compared, and `slab show --json` and the MCP `show_run` tool return them.
  Those are the numbers a correction is computed from, so "fmax was 0.062
  against 0.05" leads to a rerun with more steps.

Listings stay compact, with a one-line `error` per run, and `show` fetches
the full evidence for one run.

## Retention policy as data

```json
{
  "quarantined": {"ttl_days": 30},
  "verified":    {"ttl_days": 90},
  "promoted":    {"keep": ["terminal", "input"]}
}
```

TTLs attach to lifecycle states, anchored to when the run entered its
current state, and roles not listed in `keep` are hash-only. A TTL on
`promoted` or `archived` is rejected at validation, so the asymmetry is
enforced, not conventional.

## Engines on HPC clusters

SLAB manages cluster software the way
[Garden-AI/rootstock](https://github.com/Garden-AI/rootstock) manages MLIPs,
and it uses rootstock itself for the MLIP case.

**MLIPs via rootstock, served by name.** On a cluster with a rootstock
install, any canonical checkpoint id works directly as the engine name.
Rootstock resolves the hosting environment and serves the model, so your
Python environment stays free of torch and model packages, and
`pip install 'slab-stack[rootstock]'` adds only a thin client:

<!-- no-verify -->
```python
relaxed, info = relax(
    atoms,
    engine="mace-mp-0-medium",                                # a checkpoint id IS an engine
    calculator_options={"cluster": "delta", "device": "cuda"},
)
```

Swapping models is a one-word change to `engine`, and because the engine
name and options are traced task inputs, the checkpoint identity is
automatically part of the cache key and the recipe. `slab engines list`
shows every id the install declares. SLAB finds the install via `cluster=`
or `root=` options, or via rootstock's own defaults (`$ROOTSTOCK_ROOT`,
`~/.config/rootstock/config.toml`), and `relax` closes the worker
subprocess when the task finishes. The explicit `engine="rootstock"` form
remains for full control (for example `checkpoint="uma:custom"` with your
own `weights=`).

**Quantum ESPRESSO, built in.** `engine="qe"` drives `pw.x` through ASE's
file-IO calculator wherever the executable and pseudopotentials exist, with
no extra required. Point it at the code, or configure ASE's own config file
once, and everything else is standard `Espresso` options.
`resolve_pseudopotentials` maps elements to `.upf` files, and it refuses
ambiguity rather than guessing:

<!-- no-verify -->
```python
from slab.backends import resolve_pseudopotentials

relaxed, info = relax(
    atoms,
    engine="qe",
    calculator_options={
        "command": "mpirun -np 8 pw.x",
        "pseudo_dir": "/opt/pseudos/sssp",
        "pseudopotentials": resolve_pseudopotentials(atoms, "/opt/pseudos/sssp"),
        "input_data": {"system": {"ecutwfc": 50.0}},
        "kpts": (4, 4, 4),
    },
)
```

Each calculation runs in a slab-managed scratch directory, never your cwd,
and the final SCF's `espresso.pwo` is kept as an intermediate artifact. The
detected `pw.x` version, with the resolved command and `pseudo_dir`, lands
in the recipe and the cache key, so an executable upgrade or a
pseudopotential-library switch honestly invalidates cached results.

**Protocols and pseudopotential families, adopted from AiiDA.** Two more
pieces of the AiiDA ecosystem, reimplemented as policy-as-data.
`slab pseudos install sssp` fetches a pseudopotential family from the
official Materials Cloud archive, following
[aiida-pseudo](https://github.com/aiidateam/aiida-pseudo)'s pattern, and it
verifies every file against the published checksums. Installed families are
addressed by name (`pseudo_family="SSSP/1.3/PBEsol/efficiency"`), and their
cache identity is a digest of their contents, not a path. On top of that,
the named input protocols from
[aiida-quantumespresso](https://aiida-quantumespresso.readthedocs.io/en/stable/topics/protocol.html)
(`fast`, `balanced`, `stringent`) expand a structure into concrete, curated
pw.x inputs, with family-recommended cutoffs, a k-mesh from a reciprocal
spacing, cold smearing, and per-atom-scaled thresholds:

<!-- no-verify -->
```python
from slab.protocols import qe_protocol_options
from foundation.tasks import single_point

options = qe_protocol_options(atoms, protocol="balanced")                        # explicit, never a default
relaxed, info = relax(atoms, engine="mace-mp-0-medium", fmax=0.02)                # cheap geometry, MLIP served
final, dft = single_point(relaxed, engine="qe", calculator_options=options)
```

`single_point` is `relax`'s sibling task, one energy and forces evaluation
with no optimizer, and the pair makes the canonical two-fidelity chain.
Relax under a universal MLIP, then run one DFT evaluation of the relaxed
geometry, with the DFT residual force as the check that the cheap geometry
held up.

A protocol is only ever applied by name, explicitly, and the tracer hashes
the expanded numbers rather than the name, so a retuned protocol data file
can never silently re-serve stale cached results. Protocol values live in a
versioned data file (adapted from aiida-quantumespresso v4.10, MIT), and
AiiDA's pre-rename names (`moderate`/`precise`) are refused with a pointer
rather than aliased, because the rename came with retuned values. When
`pw.x` fails, the failure record speaks QE. The `Error in routine ...`
block, or the `convergence NOT achieved` stop line, is parsed out of the
output into the exception notes, and the input, output, and `CRASH` files
are kept as artifacts. Force printing (`tprnfor`) defaults on, because
slab's tasks drive optimizers with forces.

**LAMMPS as a built-in.** `engine="lammps"` drives the `lmp` binary through
ASE's `lammpsrun` calculator on the same terms. It needs just the executable
(`command=`, `[engines.lammps]` in the slab config, or
`$ASE_LAMMPSRUN_COMMAND`) plus your potential. The potential is required,
because `pair_style` and `pair_coeff` have no default: ASE's silent fallback
is a dimensionless `lj/cut` toy that would return meaningless numbers, and
which potential describes a system is a science decision. `files=` entries
are staged into the slab-managed scratch, and bare-basename `pair_coeff`
references resolve to the staged copies, so options work from any cwd. When
`lmp` fails, the failure record speaks LAMMPS. The real `ERROR: ...` line
dies inside a `lammpsrun` reader thread, and Python sees only "Failed to
retrieve any thermo_style-output". So slab parses the line out of the
retained log, with one line of preceding context, and keeps the input, log,
and data files as artifacts.

**Everything else via the engine registry.** For VASP, site-specific MLIP
aliases, and site-curated QE or LAMMPS setups, SLAB generalizes rootstock's
pattern. The client is only a bootstrap, and a registry file that lives with
the cluster declares how each canonical engine name is built here. Workflow
code says `engine="vasp"` and runs unchanged on any cluster whose registry
declares `vasp`. Resolution order is built-ins, then registry, then
rootstock checkpoint ids, so a maintainer's curated alias (with baked-in
options) always beats bare checkpoint resolution. Entries may not shadow
built-in names (`qe`, `lammps`, `rootstock`, ...), so site aliases pick
distinct names. Names retired from the built-ins (`mace`) are legal, and
declaring them is how a site keeps `engine="mace"` working in existing
scripts by pointing at a rootstock checkpoint.

```json
{
  "layout_version": 1,
  "cluster": "delta",
  "engines": {
    "mace-mp": {"calculator": "rootstock.RootstockCalculator",
                 "options": {"cluster": "delta", "checkpoint": "mace-mp-0-medium"}},
    "qe-delta": {"calculator": "slab.backends.qe_calculator",
                 "options": {"command": "srun pw.x", "pseudo_dir": "/sw/pseudos/sssp"},
                 "version": "7.3.1", "probe": ["pw.x", "-h"]}
  }
}
```

A maintainer ships this file at a shared path and exports `SLAB_ENGINES`
from a module file, and discovery order is an explicit path, then
`$SLAB_ENGINES`, then `~/.config/slab/engines.json`. See
[examples/engines.example.json](examples/engines.example.json). A curated QE
or LAMMPS alias goes through SLAB's own factories
(`slab.backends.qe_calculator` / `lammps_calculator`), which carry the
built-in engines' guards and take JSON-able options. An entry's `env` may
only hold variables the calculator reads at run time (`VASP_PP_PATH`), and
`ASE_CONFIG_PATH` is refused, because ASE parses that file once at import,
before any registry entry runs. Every entry is a dotted path to an ASE
calculator, so the ASE `Calculator` contract stays SLAB's only engine seam.
Declared `version`s land in task recipes as provenance and in the relax
cache key, so bumping `qe-delta` from 7.3 to 7.4 in the registry honestly
invalidates cached results instead of serving the old engine's numbers.
Entries that shadow built-in names are rejected loudly, and a registry with
a newer `layout_version` than the client understands refuses rather than
misreads.

```bash
slab engines list      # built-ins + everything this cluster declares
slab engines verify    # run every entry's probe; exit nonzero on failure
```

Trust model, stated plainly: registry entries execute maintainer-declared
code and environment variables as the calling user. SLAB isolates
configuration, not privilege, so trusting a cluster's `engines.json` is
trusting its module farm, exactly as with rootstock installs.

## Configuring for your HPC

Everything machine-specific is **policy-as-data** in layered TOML. A site
file is shipped by the cluster maintainer (`$SLAB_SITE_CONFIG`, exported
from a module file), a user file lives at `~/.config/slab/config.toml`, and
a project `slab.toml` lives with the project. They merge key-by-key, with
the project winning, and explicit environment variables above all of them.
`slab config init` writes a commented template, and `slab config show`
prints the merged result with each value's origin. Fill-in-the-blank
templates for the config file and the engine registry live in
[`templates/`](templates/), and a checkout gitignores every `slab.toml` and
`engines.json`, so filled-in machine facts (accounts, partitions, paths)
never end up in a commit.

One file declares paths (workspace, pseudopotential root, engine registry),
`[engines.qe]` defaults (`command = "srun pw.x"`), and `[hpc]` SLURM
partitions. The partitions drive a deliberately thin scheduler layer:

```bash
slab hpc partitions                                  # what the config declares
slab hpc render "slab run relax.py" --name si        # the exact sbatch script
slab hpc submit "slab run relax.py" --name si        # sbatch --parsable
slab hpc status 4242314                              # squeue, then sacct
```

Only fields the config sets become `#SBATCH` directives, with no silent
resource defaults, and the submitted script is kept next to the job's
outputs as provenance. Config never reaches a cache key. It supplies
defaults that resolve into explicit values, and those resolved values are
what recipes record.

## Agent surface (MCP)

```json
{"mcpServers": {"foundation": {"command": "foundation", "args": ["mcp"]}}}
```

Tools: `launch_workflow`, `list_runs`, `show_run`, `promote_run`,
`list_sessions`, `promote_session`, `expire_runs`, `gc`, `list_engines`. They use the same code paths as the
CLI and return structured JSON, and script output is captured into the
result, so prints cannot corrupt the protocol channel.

## Mason: the resident research agent

`mason` is a built-in Claude-Code-class agent harness for
**open-weight models** on your own hardware, whether that is Ollama on a
laptop or vLLM on a compute node, through the OpenAI-compatible API with a
stdlib HTTP client and no SDK. It is tuned for long atomistic research
projects. Calculations run as SLAB workflow scripts through its
`launch_workflow` tool, so every number it reports traces to a run id and its
`@check` assertions. Memory lives in `NOTEBOOK.md` and `PLAN.md` in the
project directory, because files outlive context windows, and history
compacts into structured summaries well before the model's window fills.
Facts about the machine rather than the project go to a separate store
that every project on that machine reads, so a quirk of the local
software is worked out once. SLURM tools appear exactly when the config
declares partitions.

```bash
slab mason serve start --wait # start the model on a GPU node (a batch job)
slab mason doctor                  # endpoint reachable? model served? tool calls parsed?
slab mason chat                    # interactive session
slab mason run "..." --auto   # one autonomous goal
```

On a cluster the endpoint is **discovered, not configured**. The GPU node is
the scheduler's choice, so `[agent.serve]` declares the launch (partition,
port, vLLM flags), and the job records the URL it landed on. The job deletes
that record when the server exits, so a dead node never keeps answering for
a live one, and a written-down `[agent] endpoint` always outranks it.

`[agent] compute_profile` (`laptop`, `workstation`, or `cluster`, derived
from your SLURM config when unset) tells the agent how big a calculation it
may run. It also requires the agent to state plainly when a number is a
laptop-sized smoke test rather than a production result. The loop is also
provider-agnostic, so `[agent] provider = "anthropic"` puts Claude behind
the same harness where there is internet and billed API access. A Claude
subscription is a separate product and does not include it.

Mason is a research group, not one agent. Agent cards define a roster: a
principal investigator (`pi`, the default) and specialists it can hand a
scoped task to with a `delegate` tool, one level deep, under the same
approval gate and the same shared notebook. Skills in the open
[Agent Skills format](https://agentskills.io/specification) give every
agent reusable procedures with tested analysis scripts (equation-of-state
and elastic fits, convergence tables, RDF, MSD, melt-quench glasses,
NEMD transport, kinetic laws, nucleation theory), categorized per
specialist. Project
directories can add or replace both: `agents/` and `skills/` shadow the
user layer, which shadows the built-ins.

```bash
slab mason roster                  # the agents: card, layer, model, skills
slab mason skills                  # the skills: layer, audience, scripts
slab mason run --agent dft-expert "..."   # enter as a specialist
```

`slab mason roster` on a laptop serving `llama3.1:8b` through Ollama:

```text
pi                 built-in  llama3.1:8b                  5 skill(s)  [delegates]
analysis-expert    built-in  llama3.1:8b                  3 skill(s)
dft-expert         built-in  llama3.1:8b                  3 skill(s)
md-expert          built-in  llama3.1:8b                  2 skill(s)
```

Per-agent models live in config, so a strong model can orchestrate while
a local model executes: `[agent.roster.pi]` can point at
`provider = "anthropic"` while the specialists stay on the served
open-weight model. [The roster & skills
tutorial](https://tarbaugh.github.io/SLAB/tutorials/roster-and-skills/)
holds the full story, including a captured delegated run.

Mason is verified against a real Llama 3.1 8B via Ollama, where an
autonomous bulk-Cu relaxation reported an energy that matches an
independent calculation exactly, with the run verified by its own checks.
The design distills the 2024–2026 agent-harness literature (ReAct, MemGPT,
SWE-agent's interface results, Anthropic's context-engineering and
long-horizon harness guidance, Manus's cache-first lessons), and each
mechanism and its source is documented in
[the Mason tutorial](https://tarbaugh.github.io/SLAB/tutorials/mason/).

## Non-goals

- **No physics engines** and no new file-format parsers beyond what ASE
  provides. Backends are reached through the ASE `Calculator` contract.
  LAMMPS, VASP, GROMACS, QE, and MLIPs are the BLAS, and SLAB is the NumPy
  one layer up.
- **No workflow-engine scheduler integration.** The SLURM layer is thin
  submission plumbing (`slab hpc`): render, submit, poll, cancel. Runs,
  caching, and verification stay in the workspace regardless of where the
  process executes, and there is no remote state machine.
- **No web UI.**
- **No distributed daemon.** A workspace is a directory, and concurrency is
  handled at the SQLite transaction level.

## Status

MVP vertical slice, working end to end. It includes:

- the lifecycle state machine;
- a content-addressed artifact store with tiered retention;
- define-by-run tracing with content-hash caching;
- verification hooks;
- relaxation and single-point tasks for ASE, Quantum ESPRESSO, LAMMPS, and
  MLIPs served through rootstock;
- AiiDA-style input protocols and SSSP pseudopotential families;
- layered HPC configuration with a SLURM submission layer;
- the Mason agent harness, for open models self-served on a GPU node or for
  Claude, with its model server as a batch job;
- one command (`slab`) and an MCP server.

Quality gates: 1000+ tests (including every docstring example, executed as
doctests), ~95% coverage, mypy `--strict`, and adversarial multi-agent review
passes whose confirmed findings are regression tests. A test reads the AST of
every module and fails on an import that crosses the package layering the
wrong way. The QE engine is verified against a real `pw.x` 7.4.1, the LAMMPS
engine against a real `lmp`, the balanced protocol against a real SSSP
install, and Mason against a real Llama via Ollama. The `RunStore` protocol is the seam for Postgres,
and the backend factory is the seam for more engines.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev,mcp]"
.venv/bin/pytest          # tests + doctests + coverage
.venv/bin/mypy            # strict on src/
.venv/bin/ruff check .
```

Docstring examples are executed as doctests on every test run, so the API
docs cannot silently rot.

License: MIT.
