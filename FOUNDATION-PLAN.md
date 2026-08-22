# Foundation split: implementation plan

Status: approved direction, ready to implement. Written 2026-08-22 for an
implementing agent (Opus 5). The decisions in §2 are settled. Do not reopen
them. Questions that the plan does not answer go to the user.

## 0. Read this first

**Goal.** Split the single `slab` package into three peer packages in one
repository and one distribution:

| Package | Owns | Depends on |
|---|---|---|
| `slab` | Access to computational software: engines, calculators, the engine registry, QE protocols, pseudopotential families, the SLURM layer, the config loader | nothing in this repo |
| `foundation` | Workflows and state: runs, lifecycle, artifact store, retention, tracing and caching, checks, failure records, the ready-made tasks, run launching, the MCP server | `slab` |
| `mason` | The resident research agent: LLM clients, the ReAct loop, tools, session, prompts, model serving | `foundation`, `slab` |

The dependency direction is a hard rule. `slab` never imports `foundation`
or `mason`. `foundation` never imports `mason`. A test enforces this (§3,
phase 2).

**Scoping rule for Foundation.** Foundation is everything between the
calculator and the agent: state, provenance, workflows, and the surfaces
that expose them (the run CLI, the MCP server). Anything specific to one
agent implementation stays in `mason`. Apply this rule to future additions
so Foundation does not become a catch-all.

**How to work.**

- Work on a branch named `foundation-split`. Make one commit per phase.
  Stop before merging or pushing to `main`; the user reviews first.
- Use `git mv` for every file move so history follows the file.
- Every phase ends with all four gates green (§5). Do not start the next
  phase on a red tree.
- Run tools from the venv explicitly (`.venv/bin/python`, `.venv/bin/pytest`,
  `.venv/bin/ruff`, `.venv/bin/mypy`, `.venv/bin/mkdocs`). The ambient shell
  resolves an anaconda Python ahead of the venv.
- Do not add compatibility shims (no `slab.tasks` re-export, no
  `slab.mason` alias). There are no external users, and shims hide layering
  violations.
- Do not change behavior beyond what this plan lists. This is a move, not a
  rewrite. Docstrings move with their code; update names inside them.
- Prose in `README.md` and `docs/` follows `CLAUDE.md` (ASD-STE100
  principles, no em-dash or colon parentheticals).
- Captured outputs in the docs are never hand-edited. Phase 5 re-executes
  them.

**Current size, for orientation.** 13,640 lines across 31 modules in
`src/slab`. After the split, roughly: `slab` 5.1k, `foundation` 4.9k,
`mason` 3.6k. "Slim" means focused scope, not a small line count;
`backends.py` (2,158 lines) stays the largest file because the QE, LAMMPS,
and rootstock adapters are detailed by necessity.

## 1. Target layout

```
src/
  slab/                 foundation/              mason/
    __init__.py           __init__.py              __init__.py
    _version.py  (shared) _ids.py                  anthropic.py
    _ops.py               _ops.py                  cli.py
    backends.py           artifacts.py             client.py
    cli.py                checks.py                config.py
    config.py             cli.py                   errors.py
    engines.py            config.py                loop.py
    errors.py             errors.py                prompts.py
    hpc.py                lifecycle.py             serve.py
    protocols.py          mcp_server.py            session.py
    pseudos.py            models.py                tools.py
    data/                 retention.py
      qe_protocols.json   runtime.py
                          serialize.py
                          store.py
                          tasks.py
                          tracing.py
```

### 1.1 Module ownership (every current file)

| Current file | Lines | Destination | Notes |
|---|---|---|---|
| `slab/_ids.py` | 35 | `foundation/_ids.py` | move |
| `slab/_ops.py` | 397 | split | `resolve_root`, `DEFAULT_ROOT`, `parse_duration_days`, `load_policy`, `ttl_override_policy`, `run_summary`, `run_details`, `launch_script`, `_execute_script` and the duration constants → `foundation/_ops.py`. `engines_overview`, `_hpc_overview`, `_rootstock_checkpoints_overview` stay in `slab/_ops.py`. |
| `slab/_version.py` | 3 | stays | Single version source for all three packages (§2.8). |
| `slab/artifacts.py` | 274 | `foundation/artifacts.py` | move |
| `slab/backends.py` | 2158 | stays | |
| `slab/checks.py` | 228 | `foundation/checks.py` | move |
| `slab/cli.py` | 1190 | split three ways | §2.5 |
| `slab/config.py` | 756 | split | Loader and SLAB sections stay. `[workspace]` → `foundation/config.py`. `[agent]`, `[agent.serve]` → `mason/config.py`. §2.3 |
| `slab/engines.py` | 491 | stays | |
| `slab/errors.py` | 372 | split | §2.4 |
| `slab/hpc.py` | 507 | stays | Driver-name set changes (§2.9). |
| `slab/lifecycle.py` | 211 | `foundation/lifecycle.py` | move |
| `slab/mason/*` | 2992 | `mason/*` | move the directory to top level |
| `slab/mcp_server.py` | 143 | `foundation/mcp_server.py` | move; server name and docs snippet change (§2.6) |
| `slab/models.py` | 248 | `foundation/models.py` | move |
| `slab/protocols.py` | 295 | stays | Docstring usage example changes `slab.tasks` → `foundation.tasks`. |
| `slab/pseudos.py` | 531 | stays | |
| `slab/retention.py` | 296 | `foundation/retention.py` | move |
| `slab/runtime.py` | 422 | `foundation/runtime.py` | move |
| `slab/serialize.py` | 105 | `foundation/serialize.py` | move |
| `slab/store.py` | 1094 | `foundation/store.py` | move |
| `slab/tasks.py` | 435 | `foundation/tasks.py` | move whole (§2.2) |
| `slab/tracing.py` | 321 | `foundation/tracing.py` | move; recipe key changes (§2.8) |
| `slab/__init__.py` | 136 | split | §2.7 |
| `slab/data/qe_protocols.json` | | stays | loaded via `resources.files("slab")` |

Already clean: none of `store`, `artifacts`, `retention`, `lifecycle`,
`models`, `runtime`, `tracing`, `serialize`, `checks`, `_ids` import
anything from `backends`, `engines`, `protocols`, `pseudos`, or `hpc`. And
none of those five import the run machinery. The tangled files are exactly
`config.py`, `errors.py`, `tasks.py`, `_ops.py`, `cli.py`, `mcp_server.py`,
`__init__.py`, and everything under `mason/`.

## 2. Decisions (settled)

### 2.1 Names

| Thing | Name |
|---|---|
| Repository, docs site, product | SLAB (unchanged) |
| PyPI distribution | `slab-stack` (`slab`, `foundation`, and `mason` are all taken on PyPI; verify `slab-stack` is free before the first publish, which is not part of this refactor) |
| Import packages and console scripts | `slab`, `foundation`, `mason` |
| Config file, env vars, workspace dir, user config dir | `slab.toml`, `SLAB_*`, `.slab/`, `~/.config/slab/` (unchanged) |
| Pseudopotential install dir | `~/.local/share/slab/pseudos` (unchanged) |

On-disk names stay under the SLAB umbrella because the project is still
SLAB. Nothing on a user's machine needs migrating.

### 2.2 `relax` and `single_point` move whole to `foundation.tasks`

They are the workflow building blocks, and their bodies interleave the ASE
optimization with run-context artifact keeping (`current_run()`,
`_keep_unique`). Move the file as one unit. Do not split it at a
compute/keeping seam in this refactor; that is listed as a follow-up (§6).
`foundation.tasks` imports `slab.backends`, which is the allowed direction.
Heavy imports (ASE, numpy) stay quarantined behind `foundation.tasks` and
`slab.backends`; both package roots stay import-light.

### 2.3 Configuration: one file, one loader, owned sections

The loader lives in `slab.config` because SLAB's own factories read config
(`paths.scratch`, `engines.qe.*`, `engines.lammps.*`, `engines.rootstock.*`,
`paths.engines`, `paths.pseudos`) and SLAB cannot import anything higher.
Each package owns whole tables:

| Table | Owner | Model |
|---|---|---|
| `schema_version` | `slab` | field on `SlabConfig` |
| `[paths]` (`pseudos`, `engines`, `scratch`) | `slab` | `PathsConfig` |
| `[engines.*]` | `slab` | `EnginesConfig` |
| `[hpc]`, `[hpc.partitions.*]` | `slab` | `HpcConfig` |
| `[workspace]` (`root`) | `foundation` | `WorkspaceConfig` (new) |
| `[agent]`, `[agent.serve]` | `mason` | `AgentConfig`, `ServeConfig` (moved) |

`paths.workspace` moves to `[workspace] root`. The old key is refused by a
`model_validator(mode="before")` on `PathsConfig` with the message
`paths.workspace moved to [workspace] root` plus the file origin. Keep
`SCHEMA_VERSION = 1`; the refusal message is the migration.

New loader API in `slab.config`:

```python
KNOWN_TOP_LEVEL_KEYS = frozenset({"schema_version", "paths", "engines", "hpc", "workspace", "agent"})
ExpandedPath = Annotated[str, AfterValidator(_expand_path)]   # "~" and $VAR / ${VAR}; unset var raises ValueError naming it

@dataclass(frozen=True)
class MergedConfig:
    raw: dict[str, Any]                 # deep-merged site < user < project
    origins: dict[str, str]             # dotted key -> "layer (path)"
    files: tuple[tuple[str, Path], ...] # (layer, path), lowest first

def find_config_files(cwd=None) -> list[tuple[str, Path]]         # unchanged
def load_merged(cwd=None) -> MergedConfig                          # per-layer schema_version check, merge, origins, top-level key lint
def validate(merged: MergedConfig, model: type[T]) -> T            # pydantic errors -> ConfigError naming file and key
def load_config(cwd=None) -> SlabConfig                            # validate(load_merged(cwd), SlabConfig)
def config_value(dotted: str, cwd=None) -> Any                     # over SlabConfig; SLAB callers only
def user_config_path() -> Path; def write_template(path, *, force=False) -> Path; CONFIG_TEMPLATE
```

Rules:

- `load_merged` refuses any top-level key not in `KNOWN_TOP_LEVEL_KEYS`,
  naming the file and listing the known keys. This keeps the typo guard for
  table names in every package. `KNOWN_TOP_LEVEL_KEYS` holds strings only;
  SLAB never imports the owners' models.
- Every view model (`SlabConfig`, `FoundationConfig`, `MasonConfig`) uses
  `extra="ignore"` at the top level and `extra="forbid"` inside its tables.
  A typo inside `[agent]` is caught when Mason loads; a typo inside
  `[hpc]` when SLAB loads. Only `SlabConfig` keeps the `schema_version`
  field.
- `${VAR}` expansion moves out of the loader and into the field type.
  `_PATH_KEYS` is deleted. Path-valued fields are typed `ExpandedPath | None`:
  `paths.pseudos`, `paths.engines`, `paths.scratch`, `engines.qe.pseudo_dir`,
  `engines.rootstock.root`, `workspace.root`. Setup lines and every other
  string field keep their literal text by construction. The existing tests
  `test_path_values_expand_home_and_variables`,
  `test_unset_variable_in_path_is_refused`, and
  `test_setup_lines_are_never_expanded` must still pass; the refusal message
  must still name the variable, the key, and the file.
- `foundation/config.py`:

  ```python
  class WorkspaceConfig(BaseModel):      # frozen, extra="forbid"
      root: ExpandedPath | None = None
  class FoundationConfig(BaseModel):     # frozen, extra="ignore"
      workspace: WorkspaceConfig = WorkspaceConfig()
  def load_config(cwd=None) -> FoundationConfig
  def config_value(dotted: str, cwd=None) -> Any
  ```

- `mason/config.py`: `ServeConfig`, `AgentConfig` (with `_OLLAMA`, the
  validators, `resolved_endpoint`, `resolved_api_key_env`,
  `compute_profile` logic) move verbatim; add

  ```python
  class MasonConfig(BaseModel):          # frozen, extra="ignore"
      agent: AgentConfig = AgentConfig()
  def load_config(cwd=None) -> MasonConfig
  ```

- `foundation._ops.resolve_root` order stays: explicit > `$SLAB_WORKSPACE` >
  `[workspace] root` > `./.slab`.
- `MasonSession.__init__` signature becomes
  `(cwd=None, *, workspace_root=None, agent: AgentConfig | None = None, hpc: HpcConfig | None = None, approver=None, auto_approve=False)`.
  `None` loads from files (`mason.config.load_config(cwd).agent`,
  `slab.config.load_config(cwd).hpc`). Tests that pass
  `config=SlabConfig(agent=AgentConfig(...))` pass `agent=AgentConfig(...)`
  instead.
- `slab config show` stays the one place to inspect the file. It prints
  SLAB-owned tables with defaults and origins (as today), then the other
  known tables with the values that are set and their origins, without
  defaults. `slab config init` writes the full template. The template text
  stays in `slab/config.py` and keeps describing `[workspace]` and `[agent]`;
  this is an accepted concession (SLAB knows the other tables' names and
  example text, never their models). No `config` subcommand on `foundation`
  or `mason`.
- Template tests: `test_committed_template_file_is_the_config_template`
  stays as is. `test_every_key_the_template_shows_is_a_key_the_schema_accepts`
  splits the template's tables by owner and validates each against the
  owner's model (the test file may import all three packages).
- Config never reaches a cache key. Unchanged invariant.

### 2.4 Errors: one base per package

| Package | Base | Members |
|---|---|---|
| `slab.errors` | `SlabError(Exception)` | `EngineNotAvailableError`; `ConfigError` stays in `slab.config`; `hpc` keeps raising `SlabError` |
| `foundation.errors` | `FoundationError(Exception)` | `IllegalTransitionError`, `IllegalStatusChangeError`, `RunNotFoundError`, `AmbiguousRunIdError`, `RunExistsError`, `RunStateError`, `ArtifactNotFoundError`, `ArtifactExistsError`, `AmbiguousHashError`, `ScriptExitError`, `SerializationError`, `NoActiveRunError`, `NestedRunError`, `StorageError`, `SchemaVersionError`; plus `failure_record` and its helpers (`_exception_notes`, `_trimmed_traceback`, `_clip_piece`, `_clip`, `_safe_str`) |
| `mason.errors` (new file) | `MasonError(Exception)` | `LlmError`, `ContextOverflowError`, `ModelRefusalError`, `SessionError`, `ServeError` re-based on `MasonError`; the two bare `raise SlabError(...)` in `mason/loop.py` become `MasonError` |

`FoundationError` and `MasonError` do not subclass `SlabError`. Each CLI
catches its own base and the bases below it: `slab` catches `SlabError`;
`foundation` catches `(FoundationError, SlabError)`; `mason` catches
`(MasonError, FoundationError, SlabError)`. Everything else about
`failure_record` (clipping, note handling, never raising) is unchanged.

### 2.5 Three console scripts

| Script | Commands | Source |
|---|---|---|
| `slab` | `engines list\|verify`, `pseudos install\|list\|verify`, `protocols list\|show`, `hpc partitions\|render\|submit\|status\|cancel`, `config show\|init`, `--version` | `slab/cli.py` |
| `foundation` | `run`, `list`, `show`, `promote`, `expire`, `gc`, `mcp`, `--version` | `foundation/cli.py` (takes `_age`, `_render_details`, `_explained_by_task`, `_echo_failure`) |
| `mason` | `chat`, `run`, `doctor`, `serve render\|start\|stop\|status`, `--version` | `mason/cli.py` (takes `_ask_approval`, `_approve_nothing`, `_mason_session`) |

Each `--version` prints `<script> <version>`. The `-w/--workspace` and
`--policy` options stay per-command on the `foundation` verbs, as today.
Help text for the apps: `slab`: "SLAB: access to atomistic engines,
registries, protocols, pseudopotentials, and the scheduler."; `foundation`:
"Foundation: workflows, runs, and state for SLAB."; `mason`: "Mason: the
resident research agent."

User-visible renames that follow from this:

| Before | After |
|---|---|
| `slab run`, `slab list`, `slab show`, `slab promote`, `slab expire`, `slab gc`, `slab mcp` | `foundation run`, `foundation list`, ... `foundation mcp` |
| `slab mason chat\|run\|doctor`, `slab mason serve ...` | `mason chat\|run\|doctor`, `mason serve ...` |
| `slab engines`, `slab pseudos`, `slab protocols`, `slab hpc`, `slab config` | unchanged |

### 2.6 MCP server

`foundation/mcp_server.py`. Server name `"foundation"`. The documented client
config becomes `{"mcpServers": {"foundation": {"command": "foundation", "args": ["mcp"]}}}`.
The tool set is unchanged (`list_runs`, `show_run`, `promote_run`,
`expire_runs`, `gc`, `launch_workflow`, `list_engines`). `list_engines`
imports `slab._ops.engines_overview`, which is the allowed direction. The
`mcp` extra is unchanged.

### 2.7 Package roots

- `slab/__init__.py` exports `__version__`, `SlabError`,
  `EngineNotAvailableError`, `ConfigError`. New docstring: SLAB gives access
  to computational software; point to `slab.backends.get_calculator`,
  `slab.engines`, `slab.protocols`, `slab.pseudos`, `slab.hpc`. No heavy
  imports at the root.
- `foundation/__init__.py` exports everything the current `slab/__init__.py`
  exports except `SlabError` and `EngineNotAvailableError`, and adds
  `FoundationError`. The current root docstring example (Workspace,
  `@task`, `@check`, promote) moves here with `from foundation import ...`.
- `mason/__init__.py` keeps its current exports and adds `MasonError`.

### 2.8 Version and recipe

- `slab/_version.py` stays the single source. `foundation/__init__.py` and
  `mason/__init__.py` do `from slab._version import __version__`. One
  distribution, one version.
- The recipe written by `foundation.tracing` records
  `"slab-stack": __version__` instead of `"slab": __version__`. Old task rows
  keep their old recipes; nothing matches on this key (`find_cached_task`
  matches on `cache_key` only).
- Every existing cache key changes, because `module` and `qualname` are part
  of it and `slab.tasks.relax` becomes `foundation.tasks.relax`. This is
  accepted. Old workspaces stay readable (§3, phase 0 fixture). Do not bump
  the SQLite schema version; the schema does not change.

### 2.9 SLURM driver detection

`slab.hpc._is_driver_payload` (the `Path(token).name == "slab"` check) becomes a
set membership test over `_DRIVERS = frozenset({"slab", "foundation", "mason"})`.
All three are single-process CLIs that must never be replicated under the
partition launcher. The emitted comment line becomes
`# partition launcher omitted: '<name>' is a single-process driver; ...`
with the detected name. Docstrings in `hpc.py` that say `slab run
workflow.py` say `foundation run workflow.py`.

### 2.10 Mason tool names

The four `slab_*` tools are renamed to the MCP server's names, so one
concept has one name across both agent surfaces:

| Before | After |
|---|---|
| `slab_runs` | `list_runs` |
| `slab_show` | `show_run` |
| `slab_launch` | `launch_workflow` |
| `slab_engines` | `list_engines` |

`_add_slab_tools` splits into `_add_workflow_tools` (the first three, backed
by `foundation`) and `_add_engine_tools` (`list_engines`, backed by
`slab._ops.engines_overview`). `prompts.py` changes names only: the tool
names, the embedded example script (`from foundation import check, converged`,
`from foundation.tasks import relax`, `from foundation.tasks import single_point`;
`from slab.protocols import qe_protocol_options` is unchanged), `'slab run
workflow.py'` → `'foundation run workflow.py'`, and the `slab workspace:`
context line → `workspace:`. Keep every rule in the prompt as written
(numbers are copied, failures are evidence, laptop honesty, and so on).

### 2.11 Tests layout

Flat `tests/` with owner prefixes. Do not create `tests/slab/`,
`tests/foundation/`, or `tests/mason/` directories: a test directory named
like a package shadows it under pytest's default import mode.

| Current | New |
|---|---|
| `test_backends_lammps.py`, `test_backends_mlip.py`, `test_backends_qe.py`, `test_engines.py`, `test_hpc.py`, `test_protocols.py`, `test_pseudos.py` | `test_slab_<same>.py` |
| `test_config.py` | `test_slab_config.py` (loader and SLAB tables); `[workspace]` tests → `test_foundation_config.py`; `[agent]` tests → `test_mason_config.py` |
| `test_cli.py` | `test_slab_cli.py` (engines, pseudos, protocols, hpc, config verbs) and `test_foundation_cli.py` (run, list, show, promote, expire, gc, mcp) |
| `test_ops.py` | `test_foundation_ops.py`; the `engines_overview` tests → `test_slab_ops.py` |
| `test_artifacts.py`, `test_checks.py`, `test_failures.py`, `test_ids.py`, `test_lifecycle.py`, `test_models.py`, `test_retention.py`, `test_runtime.py`, `test_serialize.py`, `test_store.py`, `test_task.py`, `test_tasks_relax.py`, `test_tasks_single_point.py`, `test_mcp_server.py` | `test_foundation_<same>.py` |
| `test_mason_*.py` | unchanged names |
| new | `test_layering.py`, `test_foundation_workspace_compat.py` |

`tests/conftest.py` and the root `conftest.py` stay where they are and keep
their fixtures. `tests/data/` gains `workspace-pre-split/` (phase 0).

## 3. Phases

### Phase 0: baseline and fixture

1. Record the baseline on `main`: `.venv/bin/python -m pytest` summary line
   and coverage, `slab --help` and each sub-app's `--help` saved to the
   scratchpad for later comparison of verb sets.
2. Create the pre-split workspace fixture with the *current* code, before
   any move. In a temp dir write a script that relaxes a rattled 8-atom Cu
   cell with `engine="emt"` and one `@check`, run it twice with
   `slab run` (the second run is a cache hit and still reaches `verified`),
   then promote the first run with a reason. The fixture then holds one
   promoted run and one verified run. Copy the resulting `runs.db` and `cas/` into
   `tests/data/workspace-pre-split/` (not named `.slab`, which `.gitignore`
   excludes everywhere). Record the two run ids in a `README.txt` beside
   them.
3. Create the branch `foundation-split`. Commit the fixture.

Acceptance: fixture is under 500 KB; `Workspace("tests/data/workspace-pre-split")`
on `main` lists two runs.

### Phase 1: extract `foundation`

1. `git mv` the clean modules: `_ids`, `artifacts`, `checks`, `lifecycle`,
   `models`, `retention`, `runtime`, `serialize`, `store`, `tracing`,
   `tasks`, `mcp_server` → `src/foundation/`.
2. Split `errors.py` per §2.4 (Foundation part first; Mason's own base comes
   in phase 2, so Mason's errors temporarily keep `SlabError`).
3. Split `_ops.py` per §1.1.
4. Create `foundation/cli.py` with the run verbs and its helpers; register
   the console script `foundation = "foundation.cli:app"` in `pyproject.toml`
   now, and remove those verbs from `slab/cli.py`. Keep `slab config
   show|init` in `slab/cli.py`. The `mason` sub-app stays mounted on `slab`
   until phase 2.
5. Write `foundation/__init__.py` per §2.7. Trim `slab/__init__.py` to its
   new exports.
6. Update every import in `src/` and `tests/`. `foundation.tasks` imports
   `slab.backends`; `mason` modules import `foundation.runtime`,
   `foundation._ops`, `foundation.errors` where they used the `slab`
   equivalents.
7. Rename and split test files per §2.11. Add
   `test_foundation_workspace_compat.py`: copy the fixture to `tmp_path`,
   open it with `foundation.Workspace`, assert both runs list with their
   states, `run_details` renders the promoted run including its task whose
   recipe module is `slab.tasks`, and a fresh traced `relax` of the same
   input in a new run *misses* the cache and completes. Exercise
   `foundation show <id>` through `CliRunner` on the copy.
8. Update `pyproject.toml`: wheel `packages`, `--cov` targets, `testpaths =
   ["tests", "src"]`, mypy override `slab.tasks` → `foundation.tasks`.
   Reinstall the editable package (`.venv/bin/pip install -e '.[dev]'`).
9. Update the recipe key (§2.8) and the `hpc` driver set (§2.9) here, since
   `tracing` and the `foundation` script name both land in this phase.

Acceptance: gates green; `grep -rn "slab\.\(tasks\|runtime\|tracing\|store\|artifacts\|retention\|lifecycle\|models\|checks\|serialize\|mcp_server\|_ids\)" src tests` returns zero hits; `foundation --help` lists exactly the seven verbs; `slab --help` no longer lists them.

### Phase 2: extract `mason`

1. `git mv src/slab/mason src/mason`. Update every `slab.mason` import and
   docstring path.
2. Create `mason/errors.py` per §2.4 and re-base Mason's exceptions.
3. Create `mason/cli.py` from the `mason_app` and `serve_app` sections of
   `slab/cli.py` plus their helpers. Register `mason = "mason.cli:app"`.
   Remove the `mason` sub-app from `slab/cli.py`. Messages that say `slab
   mason ...` say `mason ...` (in `mason/cli.py`, `mason/loop.py`, and the
   config template comments).
4. Rename the tools and update the prompt per §2.10. Update
   `tests/test_mason_tools.py`, `tests/test_mason_loop.py`,
   `tests/test_mason_real.py` for the new names; `tests/test_mason_cli.py`
   imports from `mason.cli`.
5. Add `tests/test_layering.py`: walk every module under `src/` with `ast`,
   collect `import`/`from ... import` targets including those under
   `TYPE_CHECKING` and inside functions, and assert the rule in §0. Also
   assert `slab.__file__`, `foundation.__file__`, and `mason.__file__`
   resolve under `src/`, and that `pyproject.toml` declares exactly the
   three console scripts.
6. `mkdocs` is unaffected by this phase; run it anyway.

Acceptance: gates green; `grep -rn "slab\.mason\|slab mason\|slab_launch\|slab_show\|slab_runs\|slab_engines" src tests` returns zero hits; `mason --help` lists `chat`, `run`, `doctor`, `serve`.

### Phase 3: configuration ownership

1. Implement the loader API in §2.3 in `slab/config.py`: `MergedConfig`,
   `load_merged`, `validate`, `ExpandedPath`, `KNOWN_TOP_LEVEL_KEYS`. Delete
   `_PATH_KEYS` and `_expand_path_values`. Replace `load_config_with_origins`
   (used by `slab config show` and one test) with `load_merged` plus
   `validate`; the origins come from `MergedConfig.origins`. Keep `find_config_files`,
   `_read_toml`, `_merge_into`, `_set_origin_tree`, `_origin_for`,
   `_describe_validation_error`, `_check_schema_version`.
2. Move `ServeConfig` and `AgentConfig` to `mason/config.py`; add
   `MasonConfig` and `mason.config.load_config`.
3. Add `foundation/config.py` with `WorkspaceConfig`, `FoundationConfig`,
   `load_config`, `config_value`. Point `foundation._ops.resolve_root` at
   `workspace.root`.
4. Remove `workspace` from `PathsConfig`; add the `paths.workspace` refusal
   validator. Update `CONFIG_TEMPLATE` and `templates/slab.toml` (`[workspace]
   root` replaces `paths.workspace`; `[agent]` comments say `mason doctor`
   and `mason serve`).
5. Rework `MasonSession.__init__` per §2.3 and update the Mason tests that
   built a `SlabConfig`.
6. `slab config show` per §2.3. `hpc.py`, `backends.py`, `engines.py`,
   `pseudos.py` keep calling `slab.config.config_value` / `load_config`
   unchanged.
7. Split the config tests per §2.11. Add tests for: unknown top-level table
   refused naming the file and listing known keys; a typo inside `[agent]`
   refused by `mason.config.load_config` but ignored by
   `slab.config.load_config`; `paths.workspace` refused with the pointer;
   `[workspace] root` expands `~` and `$VAR`; the template's tables validate
   against their owners.

Acceptance: gates green; every pre-existing `test_config.py` expectation
still holds under its new file; `slab config show` on a file that sets all
five tables prints every set key with an origin.

### Phase 4: packaging and naming sweep

1. `pyproject.toml`: `name = "slab-stack"`; update `description`; confirm
   the three scripts, wheel packages, coverage targets, mypy overrides.
   `.venv/bin/pip uninstall -y slab` then `.venv/bin/pip install -e
   '.[dev,docs]'` (plus `mace` if it was installed) so the old `slab`
   distribution metadata does not linger.
2. `examples/demo.py`: imports, printed CLI hints, and the install hint
   (`pip install 'slab-stack[mace]'`).
3. `templates/README.md`: command names.
4. `CLAUDE.md`: add a "Package layering" section stating the three packages,
   the ownership rule for Foundation (§0), the dependency direction, and
   that `tests/test_layering.py` enforces it.
5. Name sweep. This command must return zero hits outside `FOUNDATION-PLAN.md`
   and git history:

   ```bash
   grep -rn "slab run\|slab list\|slab show\|slab promote\|slab expire\|slab gc\|slab mcp\|slab mason\|slab_launch\|slab_show\|slab_runs\|slab_engines\|slab\.tasks\|slab\.runtime\|slab\.mason\|from slab import \(Workspace\|task\|check\)" src tests docs README.md ARCHITECTURE.md templates examples CLAUDE.md mkdocs.yml
   ```

Acceptance: gates green; `pip show slab-stack` lists the three top-level
packages; `pip show slab` reports nothing installed.

### Phase 5: documentation

Prose edits follow `CLAUDE.md`. Captured outputs are re-executed, never
typed.

1. Write a scratch runner (in the scratchpad, not committed) that, for one
   page, walks fenced blocks in order, skips any block preceded by
   `<!-- no-verify -->`, executes `python` blocks cumulatively in a fresh
   temp directory with a fresh workspace and EMT only, executes runnable
   `bash` blocks in that same directory, and replaces the `text` block that
   follows each executed block with the captured stdout. Outputs contain run
   ids, timestamps, and paths that differ from the previous capture; that
   is expected.
2. Update names in every page, then re-capture. Per page:
   - `README.md`: imports, CLI verbs, install hint, the status inventory, and
     a new short section "Three packages" (see wording below).
   - `docs/index.md`: the same "Three packages" section; "Where to go next"
     unchanged in structure.
   - `quickstart.md`: `from foundation import ...`, `from foundation.tasks
     import relax`, `foundation run|list|show|promote`.
   - `lifecycle-and-retention.md`, `verification.md`, `caching-and-resume.md`
     (the recipe capture shows the `slab-stack` key), `debugging-failures.md`
     (`foundation show` captures): imports and verbs.
   - `engines.md`, `protocols-and-pseudos.md`: `slab engines`, `slab pseudos`,
     `slab protocols` unchanged; task imports change.
   - `hpc-config.md`: `[workspace] root`, the `slab config show` capture, the
     rendered sbatch comment line, `foundation run` payloads.
   - `agents-mcp.md`: server name, client config snippet, `foundation mcp`.
   - `mason.md`: `mason chat|run|doctor|serve`, the four tool names, and the
     `[agent]` template comments.
   - `ARCHITECTURE.md` §7 "The layers": redraw the diagram as three packages
     with the dependency arrows, and state the Foundation scoping rule. This
     file is exempt from the STE rules; keep its register. Do not touch §1
     to §6.
3. `mkdocs.yml`: no nav change is required. Leave it unless a link breaks.

"Three packages" wording (README and index, adjust the link targets):

> SLAB is three packages in one distribution. `slab` gives access to
> computational software: engines, the registry, protocols,
> pseudopotentials, and the scheduler. `foundation` keeps state and runs
> workflows: runs, artifacts, caching, verification, and the MCP server.
> `mason` is the resident agent. `mason` depends on `foundation` and
> `slab`, `foundation` depends on `slab`, and `slab` depends on neither.

Acceptance: `.venv/bin/mkdocs build --strict` clean; every non-`no-verify`
block on every page was executed by the runner after the final prose edit;
the name sweep in phase 4 still returns zero hits; anchors that other pages
link to are unchanged (`engines.md` `#two-fidelities-one-run`,
`#quantum-espresso`, `#lammps`, `#built-ins`; `debugging-failures.md`
`#when-the-engine-writes-files`; `mason.md` `#compute-budget-sizing-the-physics-to-the-machine`,
`#on-the-cluster-end-to-end`, `#claude-behind-the-same-harness`).

### Phase 6: verification sweep

1. All four gates (§5) from a clean checkout of the branch.
2. Smoke the three scripts end to end in a temp directory: `slab engines
   list`, `foundation run` on the phase 0 script, `foundation list|show|
   promote|expire|gc`, `foundation mcp --help`, `mason doctor` against an
   unreachable endpoint (expect the documented loud failure), `mason serve
   render` with a minimal `[hpc]` and `[agent.serve]` config.
3. If an Ollama server is running on this machine (`curl -s
   localhost:11434/v1/models`), run the gated Mason test once:
   `SLAB_TEST_LLM=http://localhost:11434/v1 SLAB_TEST_LLM_MODEL=llama3.1:8b
   .venv/bin/python -m pytest tests/test_mason_real.py --no-cov`. The tool
   renames changed the prompt, so this is the only check that the model
   still calls the tools. An 8B model is flaky; treat a failure as a report
   item, not a gate, and include the transcript path in the report.
4. Report: per-package line counts, test count and coverage before and
   after, the list of user-visible renames (copy §2.5 and §2.10), and any
   behavior you changed that this plan did not list.

## 4. Out of scope

Do not do these in this refactor:

- No untraced `relax`/`single_point` primitive in `slab` (see §6).
- No renames of `slab.toml`, `SLAB_*`, `.slab/`, `~/.config/slab/`, or the
  pseudopotential directory.
- No PyPI publish and no CI workflow. Only the docs workflow exists; leave
  `.github/workflows/docs.yml` as it is (it does not install the package).
- No schema change to the SQLite store and no retention or lifecycle change.
- No rewrite of docstring voice in `src/`.
- No new Mason tools, no `delegate` tool, no sandbox.
- No Postgres `RunStore`.

## 5. Gates

Run all four at the end of every phase, from the repository root:

```bash
.venv/bin/ruff check .
```

```bash
.venv/bin/mypy
```

```bash
.venv/bin/python -m pytest
```

```bash
.venv/bin/mkdocs build --strict
```

`pytest` enforces 80% coverage across the three packages together and runs
the doctests in `src/`. Subset runs need `--no-cov`. `ruff format` is not a
gate. Baseline before this refactor: 1008 passed, 5 skipped, 95.68%
coverage.

## 6. Follow-ups (not part of this refactor)

- A compute/keeping seam: `slab` exposes untraced `relax` and
  `single_point` that accept a `sink(name, path)` callback for produced
  files, and `foundation.tasks` wraps them with tracing and `run.keep`.
  This would make `slab` useful without a workspace. Revisit after the
  split settles.
- Verify `slab-stack` is free on PyPI before the first publish.
- A CI workflow for the four gates.
- Separate distributions per package, if ever needed. The layering test
  keeps that door open.
