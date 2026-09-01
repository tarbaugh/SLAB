# Changelog

All notable changes to SLAB, newest first. Dates are commit dates on
`main`.

## Unreleased

- `slab benchmark`: five fixed copper campaigns with DFT-PBE references and
  per-engine-class tolerance bands; `run`, `launch`, `score`, and `render`.
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
