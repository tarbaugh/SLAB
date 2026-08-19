"""Layered TOML configuration: one file format for laptops, clusters, and projects.

SLAB runs in three habitats — a laptop, a shared HPC install, a project
directory — and each wants different defaults: where the workspace lives,
which ``pw.x`` to run, what SLURM partitions exist, which LLM endpoint the
resident agent talks to. Hard-coding any of it into the package would go
stale in pinned clients (the registry lesson, :mod:`slab.engines`), so all
of it is **policy-as-data**: TOML files validated against a versioned schema.

Three layers merge, lowest precedence first:

1. **site** — the file ``$SLAB_SITE_CONFIG`` points at. Cluster maintainers
   ship one per machine and export the variable from a module file, exactly
   like ``$SLAB_ENGINES``.
2. **user** — ``$XDG_CONFIG_HOME/slab/config.toml``
   (``~/.config/slab/config.toml``): personal defaults.
3. **project** — ``./slab.toml`` in the working directory, or the file
   ``$SLAB_CONFIG`` points at. Travels with the project in version control.

Deeper tables merge key-by-key; a scalar in a higher layer replaces the
lower one. Every resolved value remembers which file said it — ``slab
config show`` prints the origin next to each key, so "why is it using that
partition?" has a one-command answer.

Configuration sits *below* the explicit environment: ``$SLAB_WORKSPACE``,
``$SLAB_PSEUDOS``, ``$SLAB_ENGINES``, and explicit function arguments all
override it. And it never reaches a cache key — config supplies *defaults*
that resolve into concrete values (a command, a pseudo directory), and those
resolved values are what task recipes record.

Unknown keys are refused with the offending file named — a typo in a
cluster config must surface at load, not silently configure nothing.
"""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from slab.errors import SlabError

SCHEMA_VERSION = 1
CONFIG_ENV_VAR = "SLAB_CONFIG"
SITE_CONFIG_ENV_VAR = "SLAB_SITE_CONFIG"
PROJECT_FILE_NAME = "slab.toml"

# Path-valued keys get ~ and ${VAR} expanded at load time (the slab process
# itself opens them). Everything else — notably [hpc] setup lines — is left
# verbatim: those are shell fragments whose variables must expand on the
# compute node at run time, not on the login node at load time.
_PATH_KEYS = frozenset(
    {"paths.workspace", "paths.pseudos", "paths.engines", "engines.qe.pseudo_dir"}
)
_VAR_PATTERN = re.compile(r"\$\{(\w+)\}|\$(\w+)")
_OLLAMA = "http://localhost:11434/v1"


class ConfigError(SlabError):
    """A configuration file is missing, malformed, or declares the impossible."""


class QeEngineConfig(BaseModel):
    """Defaults for the built-in ``qe`` engine (``[engines.qe]``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: str | None = None
    pseudo_dir: str | None = None


class LammpsEngineConfig(BaseModel):
    """Defaults for the built-in ``lammps`` engine (``[engines.lammps]``).

    Only the machine fact lives here — how to invoke the binary. The
    interatomic potential (``pair_style``/``pair_coeff``/``files``) is a
    science decision passed per call in ``calculator_options``, never a
    machine default.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: str | None = None


class EnginesConfig(BaseModel):
    """Per-engine defaults (``[engines]``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    qe: QeEngineConfig = QeEngineConfig()
    lammps: LammpsEngineConfig = LammpsEngineConfig()


class PathsConfig(BaseModel):
    """Where things live on this machine (``[paths]``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace: str | None = None
    pseudos: str | None = None
    engines: str | None = None


class Partition(BaseModel):
    """One SLURM partition as the cluster config declares it.

    Only fields that are set become ``#SBATCH`` directives — SLAB adds no
    silent defaults of its own; what the file says is what the scheduler
    sees. ``setup`` lines (module loads, environment) run before the job
    body; ``launcher`` (e.g. ``srun``) prefixes commands that should run
    under the parallel launcher; ``sbatch_extra`` is the explicit escape
    hatch for directives this schema does not model.

    Examples:
        >>> Partition.model_validate({"time_limit": "24:00:00", "gres": "gpu:a100:4"}).gres
        'gpu:a100:4'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    description: str = ""
    account: str | None = None
    qos: str | None = None
    time_limit: str | None = None
    nodes: int | None = Field(default=None, ge=1)
    ntasks: int | None = Field(default=None, ge=1)
    ntasks_per_node: int | None = Field(default=None, ge=1)
    cpus_per_task: int | None = Field(default=None, ge=1)
    mem: str | None = None
    gres: str | None = None
    constraint: str | None = None
    reservation: str | None = None
    setup: tuple[str, ...] = ()
    launcher: str | None = None
    sbatch_extra: tuple[str, ...] = ()


class HpcConfig(BaseModel):
    """The cluster around this SLAB install (``[hpc]``).

    ``account`` and ``setup`` apply to every partition unless a partition
    overrides them. ``default_partition`` is resolved (and a dangling name
    refused) by :meth:`resolve_partition`, not at load: a site file may name
    the default while a project file supplies the partition table, and each
    file must validate on its own.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cluster: str | None = None
    scheduler: Literal["slurm"] = "slurm"
    account: str | None = None
    default_partition: str | None = None
    setup: tuple[str, ...] = ()
    partitions: dict[str, Partition] = Field(default_factory=dict)

    @field_validator("partitions")
    @classmethod
    def _partition_names_are_sane(cls, value: dict[str, Partition]) -> dict[str, Partition]:
        for name in value:
            if not name or any(c.isspace() for c in name):
                raise ValueError(f"partition name {name!r} is empty or contains whitespace")
        return value

    def resolve_partition(self, name: str | None = None) -> tuple[str, Partition]:
        """The named partition (default: ``default_partition``), or a loud refusal.

        Examples:
            >>> hpc = HpcConfig.model_validate({
            ...     "default_partition": "cpu", "partitions": {"cpu": {}}})
            >>> hpc.resolve_partition()[0]
            'cpu'
        """
        target = name or self.default_partition
        if target is None:
            raise ConfigError(
                "no partition requested and no default_partition configured; "
                "set [hpc] default_partition or pass a partition name "
                f"(declared: {', '.join(sorted(self.partitions)) or 'none'})"
            )
        if target not in self.partitions:
            raise ConfigError(
                f"partition {target!r} is not declared in [hpc.partitions] "
                f"(declared: {', '.join(sorted(self.partitions)) or 'none'})"
            )
        return target, self.partitions[target]


class ServeConfig(BaseModel):
    """How to stand the agent's own model server up on this cluster (``[agent.serve]``).

    Mason talks to an OpenAI-compatible endpoint. On a laptop that endpoint is
    Ollama at a fixed localhost URL; on a cluster it is a server *you* start,
    on a GPU node the scheduler picks — so the URL cannot be written in a
    config file. This section declares the launch (which partition, which
    port, which flags); the node and the URL are discovered at run time from
    the record the job writes (:mod:`slab.mason.serve`).

    ``command`` is the escape hatch for a server this schema does not model;
    it may reference ``$port``, which the rendered script binds. Everything
    here is shell for the *compute node*, so no variable is expanded at load.

    Examples:
        >>> ServeConfig(partition="gpu", tool_call_parser="llama4_pythonic").port
        8000
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    partition: str | None = None
    time_limit: str | None = None
    port: int = Field(default=8000, ge=1024, le=65535)
    job_name: str = "mason-serve"
    tool_call_parser: str | None = None
    args: tuple[str, ...] = ()
    setup: tuple[str, ...] = ()
    command: str | None = None
    ready_timeout_s: float = Field(default=1800.0, gt=0)


class AgentConfig(BaseModel):
    """The resident research agent's model connection and budgets (``[agent]``).

    Two providers, one harness. ``provider = "openai"`` (the default) talks to
    any OpenAI-compatible ``/v1`` server — vLLM on a compute node, Ollama on a
    laptop — and keeps working where compute nodes have no internet, which is
    the normal HPC case. ``provider = "anthropic"`` talks to the Claude
    Messages API, which needs reachable internet and *billed API access* (a
    Claude subscription is a separate product and does not grant it).

    ``endpoint`` may be left unset on a cluster: :func:`slab.mason.serve.
    discover_endpoint` reads the URL from the running server job's record.

    ``api_key_env`` names an environment variable holding the key; the key
    itself never belongs in a config file (config files get committed and
    shipped). ``compute_profile`` shapes what the agent *chooses* to run, not
    what SLAB permits — see :func:`slab.mason.prompts.compute_profile_block`.

    Examples:
        >>> AgentConfig().resolved_endpoint
        'http://localhost:11434/v1'
        >>> AgentConfig(provider="anthropic").resolved_endpoint
        'https://api.anthropic.com/v1'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["openai", "anthropic"] = "openai"
    endpoint: str | None = None
    model: str | None = None
    api_key_env: str | None = None
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    compute_profile: Literal["laptop", "workstation", "cluster"] | None = None
    context_window: int = Field(default=65_536, ge=4_096)
    compact_at: float = Field(default=0.7, gt=0.0, le=1.0)
    max_turns: int = Field(default=60, ge=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    request_timeout_s: float = Field(default=600.0, gt=0)
    max_reply_tokens: int | None = Field(default=None, ge=256)
    max_tool_output_chars: int = Field(default=24_000, ge=1_000)
    shell_timeout_s: float = Field(default=120.0, gt=0)
    tool_protocol: Literal["native", "fenced"] = "native"
    approval: Literal["ask", "auto"] = "ask"
    shell_allowlist: tuple[str, ...] = ()
    serve: ServeConfig = ServeConfig()

    @property
    def resolved_endpoint(self) -> str:
        """The endpoint to call: explicit, else the provider's own default."""
        if self.endpoint:
            return self.endpoint
        return "https://api.anthropic.com/v1" if self.provider == "anthropic" else _OLLAMA

    @property
    def resolved_api_key_env(self) -> str | None:
        """The env var holding the key — Anthropic's is required and defaulted."""
        if self.api_key_env:
            return self.api_key_env
        return "ANTHROPIC_API_KEY" if self.provider == "anthropic" else None


class SlabConfig(BaseModel):
    """The merged, validated configuration — every section optional.

    Examples:
        >>> SlabConfig().agent.resolved_endpoint
        'http://localhost:11434/v1'
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = 1
    paths: PathsConfig = PathsConfig()
    engines: EnginesConfig = EnginesConfig()
    hpc: HpcConfig = HpcConfig()
    agent: AgentConfig = AgentConfig()

    @field_validator("schema_version")
    @classmethod
    def _understood_version(cls, value: int) -> int:
        if value > SCHEMA_VERSION:
            raise ValueError(
                f"config schema_version {value} is newer than this slab understands "
                f"({SCHEMA_VERSION}); upgrade slab to read it"
            )
        if value < 1:
            raise ValueError(f"config schema_version must be >= 1, got {value}")
        return value


def user_config_path() -> Path:
    """Where the user-layer config lives (``$XDG_CONFIG_HOME`` honored)."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path("~/.config").expanduser()
    return base / "slab" / "config.toml"


def find_config_files(cwd: str | os.PathLike[str] | None = None) -> list[tuple[str, Path]]:
    """The config files that would load, as ``(layer, path)`` — lowest first.

    ``$SLAB_SITE_CONFIG`` and ``$SLAB_CONFIG`` must point at existing files:
    someone exported them on purpose, so a dangling path is an error, never a
    silent skip. The user and project files are optional.

    Examples:
        >>> import os, tempfile
        >>> os.environ.pop("SLAB_SITE_CONFIG", None) and None
        >>> os.environ.pop("SLAB_CONFIG", None) and None
        >>> os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp()
        >>> find_config_files(tempfile.mkdtemp())  # no files anywhere -> empty
        []
    """
    found: list[tuple[str, Path]] = []
    site = os.environ.get(SITE_CONFIG_ENV_VAR)
    if site:
        path = Path(site).expanduser()
        if not path.is_file():
            raise ConfigError(f"${SITE_CONFIG_ENV_VAR} points to {path}, which does not exist")
        found.append(("site", path))
    user = user_config_path()
    if user.is_file():
        found.append(("user", user))
    explicit = os.environ.get(CONFIG_ENV_VAR)
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigError(f"${CONFIG_ENV_VAR} points to {path}, which does not exist")
        found.append(("project", path))
    else:
        project = Path(cwd if cwd is not None else Path.cwd()) / PROJECT_FILE_NAME
        if project.is_file():
            found.append(("project", project))
    return found


def load_config(cwd: str | os.PathLike[str] | None = None) -> SlabConfig:
    """Load and merge every layer into one validated :class:`SlabConfig`.

    No files at all is the normal laptop case and yields pure defaults.
    A malformed file is a loud :class:`ConfigError` naming the file — a
    broken cluster config must surface, not degrade to defaults.
    """
    merged, _ = load_config_with_origins(cwd)
    return merged


def load_config_with_origins(
    cwd: str | os.PathLike[str] | None = None,
) -> tuple[SlabConfig, dict[str, str]]:
    """:func:`load_config`, plus ``{dotted.key: "path (layer)"}`` for each value set."""
    merged: dict[str, Any] = {}
    origins: dict[str, str] = {}
    for layer, path in find_config_files(cwd):
        raw = _read_toml(path)
        # Each layer declares its own schema version, and each is checked:
        # merging first would let a project file's `schema_version = 1` mask a
        # site file written for a newer slab (whose other keys we would then
        # misread as if they meant what they mean today).
        _check_schema_version(raw, path)
        _merge_into(merged, raw, origins, f"{path} ({layer})", prefix="")
    _expand_path_values(merged, origins)
    try:
        return SlabConfig.model_validate(merged), origins
    except ValidationError as e:
        raise ConfigError(_describe_validation_error(e, origins)) from e


def config_value(dotted: str, cwd: str | os.PathLike[str] | None = None) -> Any:
    """One value from the merged config by dotted key, or None when unset.

    The convenience accessor integration points use (``resolve_root``,
    ``pseudos_root``, the qe engine): a missing file or unset key is None;
    a *broken* file still raises.

    Examples:
        >>> import os, tempfile
        >>> os.environ.pop("SLAB_SITE_CONFIG", None) and None
        >>> os.environ.pop("SLAB_CONFIG", None) and None
        >>> os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp()
        >>> config_value("paths.workspace", tempfile.mkdtemp()) is None
        True
    """
    node: Any = load_config(cwd)
    for part in dotted.split("."):
        if node is None:
            return None
        node = getattr(node, part, None)
    return node


def _check_schema_version(raw: dict[str, Any], path: Path) -> None:
    """Refuse one file written for a newer slab, before its keys are merged."""
    declared = raw.get("schema_version")
    if isinstance(declared, int) and declared > SCHEMA_VERSION:
        raise ConfigError(
            f"{path} declares schema_version {declared}, newer than this slab "
            f"understands ({SCHEMA_VERSION}); upgrade slab to read it"
        )


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        with open(path, "rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path} is not valid TOML: {e}") from e
    except OSError as e:
        raise ConfigError(f"cannot read config file {path}: {e}") from e


def _merge_into(
    target: dict[str, Any],
    incoming: dict[str, Any],
    origins: dict[str, str],
    source: str,
    prefix: str,
) -> None:
    """Deep-merge *incoming* over *target*, recording each leaf's origin."""
    for key, value in incoming.items():
        dotted = f"{prefix}{key}"
        if (
            isinstance(value, dict)
            and isinstance(target.get(key), dict)
        ):
            _merge_into(target[key], value, origins, source, prefix=f"{dotted}.")
        else:
            target[key] = value
            _set_origin_tree(origins, dotted, value, source)


def _set_origin_tree(origins: dict[str, str], dotted: str, value: Any, source: str) -> None:
    """Record origins for a value and, when it is a table, all its leaves."""
    if isinstance(value, dict):
        for key, child in value.items():
            _set_origin_tree(origins, f"{dotted}.{key}", child, source)
    else:
        origins[dotted] = source


def _expand_path_values(merged: dict[str, Any], origins: dict[str, str]) -> None:
    """Expand ``~`` and ``${VAR}`` in path-valued keys; unset variables refuse."""
    for dotted in _PATH_KEYS:
        parts = dotted.split(".")
        node: Any = merged
        for part in parts[:-1]:
            if not isinstance(node, dict):
                return  # a type error the schema will report properly
            node = node.get(part)
            if node is None:
                break
        else:
            leaf = parts[-1]
            if isinstance(node, dict) and isinstance(node.get(leaf), str):
                node[leaf] = _expand_path(node[leaf], dotted, origins.get(dotted, "config"))


def _expand_path(text: str, dotted: str, source: str) -> str:
    """``~`` and ``${VAR}``/``$VAR`` expansion that refuses unset variables.

    ``os.path.expandvars`` leaves unknown variables literal — a path like
    ``/scratch/$USRE/slab`` would quietly become a directory named ``$USRE``.
    Loud beats quiet.

    Examples:
        >>> import os
        >>> os.environ["SLAB_DOCTEST_USER"] = "tom"
        >>> _expand_path("/scratch/${SLAB_DOCTEST_USER}/x", "paths.workspace", "demo")
        '/scratch/tom/x'
        >>> del os.environ["SLAB_DOCTEST_USER"]
    """

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        value = os.environ.get(name)
        if value is None:
            raise ConfigError(
                f"{source}: {dotted} = {text!r} references ${name}, "
                f"which is not set in the environment"
            )
        return value

    return os.path.expanduser(_VAR_PATTERN.sub(substitute, text))


def _origin_for(dotted: str, origins: dict[str, str]) -> str:
    """The file responsible for a key, walking up when the leaf is not a leaf.

    Pydantic reports errors at positions origins does not track verbatim —
    inside a list (``hpc.setup.0``, where the whole list is one origin) or on
    a whole table (``hpc.partitions.cpu``). Walking up finds the file that
    actually supplied the value; a table falls back to every file that
    contributed to it.
    """
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        source = origins.get(".".join(parts[:cut]))
        if source is not None:
            return source
    contributors = {v for k, v in origins.items() if k.startswith(f"{dotted}.")}
    return ", ".join(sorted(contributors)) or "configuration"


def _describe_validation_error(error: ValidationError, origins: dict[str, str]) -> str:
    """One actionable line per problem, each naming the file that caused it."""
    lines = ["configuration is invalid:"]
    for item in error.errors():
        dotted = ".".join(str(piece) for piece in item["loc"])
        source = _origin_for(dotted, origins)
        message = item["msg"]
        if item["type"] == "extra_forbidden":
            message = "unknown key (check spelling against 'slab config init' template)"
        lines.append(f"  {dotted}: {message} [{source}]")
    return "\n".join(lines)


CONFIG_TEMPLATE = '''\
# SLAB configuration (TOML). Layers, lowest precedence first:
#   site:    the file $SLAB_SITE_CONFIG points at (cluster-wide, from a module file)
#   user:    ~/.config/slab/config.toml
#   project: ./slab.toml, or the file $SLAB_CONFIG points at
# Higher layers override lower ones key-by-key. Explicit environment variables
# ($SLAB_WORKSPACE, $SLAB_PSEUDOS, $SLAB_ENGINES) override all of them.
# 'slab config show' prints the merged result with each value's origin.
schema_version = 1

[paths]
# workspace = "/scratch/${USER}/slab-workspace"   # run database + artifacts
#                                      # (parallel scratch works — slab falls back
#                                      # from WAL journaling automatically — but
#                                      # avoid heavy login-node polling of a
#                                      # workspace a running job is writing to)
# pseudos = "/shared/sw/slab/pseudos"             # pseudopotential family root
# engines = "/shared/sw/slab/engines.json"        # cluster engine registry

[engines.qe]
# command = "pw.x"                     # login-node smoke tests / serial runs
# command = "srun pw.x"                # inside batch jobs only: srun outside an
#                                      # allocation queues or hangs, so slab
#                                      # refuses it there — keep this form for
#                                      # jobs submitted via 'slab hpc submit'
# command = "env OMP_NUM_THREADS=4 pw.x"          # the env wrapper scopes
#                                      # variables to this engine's subprocess
#                                      # alone (bare VAR=x needs a shell and
#                                      # is refused; ASE execs argv directly)
# pseudo_dir = "/shared/sw/pseudos"    # used when no pseudo_family is given

[engines.lammps]
# command = "lmp"                      # login-node smoke tests / serial runs
# command = "srun lmp"                 # inside batch jobs only (same srun rule)
# command = "env OMP_NUM_THREADS=4 lmp"           # same env-wrapper rule as qe

[hpc]
# cluster = "delta"
# account = "abc-123"                  # default charge account for all partitions
# default_partition = "cpu"
# setup = ["module load quantum-espresso/7.4"]    # runs before every job body

# [hpc.partitions.cpu]
# description = "general CPU nodes"
# time_limit = "24:00:00"
# nodes = 1
# ntasks_per_node = 64
# mem = "240G"
# launcher = "srun"

# [hpc.partitions.gpu]
# description = "A100 nodes"
# time_limit = "12:00:00"
# gres = "gpu:a100:4"
# qos = "gpu"
# setup = ["module load cuda/12.4"]    # replaces [hpc] setup for this partition? no:
#                                      # partition setup runs AFTER the [hpc] setup lines
# sbatch_extra = ["--exclusive"]       # raw directives the schema does not model

[agent]
# provider = "openai"                       # "openai" = any OpenAI-compatible server; "anthropic"
# model = "meta-models/Muse-Glimmer-30B"    # as the server names it ('slab mason doctor'
#                                           # lists); an absolute path to a downloaded
#                                           # model directory also works, and never
#                                           # touches the network
# endpoint = "http://gpu-node-01:8000/v1"   # leave unset on a cluster: 'slab mason serve'
#                                           # records the URL of the node it landed on
# api_key_env = "SLAB_AGENT_API_KEY"        # NAME of the env var holding the key, never the key
# context_window = 131072                   # tokens the endpoint actually serves
# compact_at = 0.7                          # compact history at this fraction of the window
# max_turns = 60                            # model calls per goal; the agent stops loudly
# approval = "ask"                          # "ask" gates mutating tools; "auto" trusts them
# shell_allowlist = ["git status", "ls"]    # command prefixes that never need approval
# tool_protocol = "native"                  # "fenced" for servers without a tool-call parser
# compute_profile = "cluster"               # laptop | workstation | cluster — how big a
#                                           # calculation the agent should reach for
#                                           # (default: cluster if [hpc] partitions exist)

# -- Claude instead of a locally served model. Still the [agent] table above:
# uncomment these keys there, do not add a second [agent] header. Needs reachable
# internet and billed API access — a Claude subscription is a separate product and
# does not grant it, and compute nodes are frequently firewalled. temperature does
# not apply here (current Claude models reject sampling parameters); use effort.
# provider = "anthropic"
# model = "claude-opus-5"
# api_key_env = "ANTHROPIC_API_KEY"         # the default for this provider
# effort = "medium"                         # low | medium | high | xhigh | max
# max_reply_tokens = 16000                  # bounds thinking AND reply together

# How 'slab mason serve' starts that server as a batch job. The GPU node is the
# scheduler's choice, so the endpoint URL is discovered, never configured.
# Compute nodes rarely have internet: download the model once, on the login node
# ('HF_HOME=/path/to/hf-cache hf download meta-models/Muse-Glimmer-30B'), and let
# the setup lines below serve it from that cache — offline, so a model missing
# from the cache fails loudly at startup instead of hanging on a download.
[agent.serve]
# partition = "gpu"                         # a partition from [hpc.partitions]
# time_limit = "08:00:00"                   # serve jobs are long-lived
# port = 8000
# tool_call_parser = "llama4_pythonic"      # vLLM's --tool-call-parser: model-specific,
#                                           # and required for native tool calls
#                                           # ('vllm serve --help' lists your build's)
# args = ["--tensor-parallel-size 4", "--max-model-len 131072"]   # extra vllm flags
# setup = [                                 # compute-node shell, run before the server
#   "source /path/to/venvs/vllm/bin/activate",  # vLLM gets its own venv (its torch
#                                               # pin and mace-torch's rarely agree)
#   "export HF_HOME=/path/to/hf-cache",         # the cache 'hf download' filled
#   "export HF_HUB_OFFLINE=1",                  # serve from disk; refuse every download
# ]
# command = "..."                           # a server this schema does not model; may use $port
'''


def write_template(path: str | os.PathLike[str], *, force: bool = False) -> Path:
    """Write the commented config template to *path* (refuses to overwrite).

    Examples:
        >>> import tempfile
        >>> target = Path(tempfile.mkdtemp()) / "slab.toml"
        >>> write_template(target).name
        'slab.toml'
    """
    target = Path(path).expanduser()
    if target.exists() and not force:
        raise ConfigError(f"{target} already exists; pass force=True (or --force) to replace it")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    return target
