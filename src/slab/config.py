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
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from slab.errors import SlabError

SCHEMA_VERSION = 1
CONFIG_ENV_VAR = "SLAB_CONFIG"
SITE_CONFIG_ENV_VAR = "SLAB_SITE_CONFIG"
PROJECT_FILE_NAME = "slab.toml"

# Every table a config file may declare, and which package validates it.
# SLAB holds the names as strings only: it lints the spelling of a table so a
# typo cannot masquerade as a section owned by a package that is not loaded,
# and it never imports the owners' models.
KNOWN_TOP_LEVEL_KEYS = frozenset(
    {"schema_version", "paths", "engines", "hpc", "workspace", "agent"}
)

# Keys that moved between packages. Refusing them by name is the migration:
# every view ignores tables it does not own, so a key left where it used to be
# would be ignored by all of them and configure nothing in silence. The check
# belongs here, in the one read every package shares, and not on the model
# that used to hold the key — only one package validates that model.
_MOVED_KEYS = {
    "paths.workspace": "[workspace] root, which foundation owns",
}

_VAR_PATTERN = re.compile(r"\$\{(\w+)\}|\$(\w+)")


class ConfigError(SlabError):
    """A configuration file is missing, malformed, or declares the impossible."""


def _expand_path(text: str, info: ValidationInfo) -> str:
    """``~`` and ``${VAR}``/``$VAR`` expansion that refuses unset variables.

    ``os.path.expandvars`` leaves unknown variables literal — a path like
    ``/scratch/$USRE/slab`` would quietly become a directory named ``$USRE``.
    Loud beats quiet.

    This is a field type, not a pass over the merged dictionary, so only the
    fields declared :data:`ExpandedPath` expand. Shell that runs on a compute
    node (``[hpc] setup``, ``[engines.qe] setup``, a serve ``command``) keeps
    its variables literal by construction rather than by being left off a
    list. The file that supplied the value is added by
    :func:`_describe_validation_error`, which knows every value's origin.

    Examples:
        >>> import os
        >>> os.environ["SLAB_DOCTEST_USER"] = "tom"
        >>> from pydantic import TypeAdapter
        >>> TypeAdapter(ExpandedPath).validate_python("/scratch/${SLAB_DOCTEST_USER}/x")
        '/scratch/tom/x'
        >>> del os.environ["SLAB_DOCTEST_USER"]
    """
    key = info.field_name or "path"

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        value = os.environ.get(name)
        if value is None:
            raise ValueError(
                f"{key} = {text!r} references ${name}, "
                f"which is not set in the environment"
            )
        return value

    return os.path.expanduser(_VAR_PATTERN.sub(substitute, text))


ExpandedPath = Annotated[str, AfterValidator(_expand_path)]
"""A config string holding a filesystem path: ``~`` and ``$VAR`` expand."""


class QeEngineConfig(BaseModel):
    """Defaults for the built-in ``qe`` engine (``[engines.qe]``).

    Two ways to name the code. ``command`` is the full invocation, written
    by hand. ``bin`` names the install's ``bin`` directory instead, and the
    command is constructed: ``mpirun -np N <bin>/pw.x``, with N taken from
    ``$SLURM_NTASKS`` (the allocation a batch job runs in) and 1 outside
    one, and a bundled ``<bin>/mpirun`` preferred over the PATH's. The two
    are exclusive — a command that names a different binary than ``bin``
    would silently win, so declaring both is refused.

    ``setup`` lines (module loads, exports) run in a private login-shell
    wrapper around THIS engine's subprocess only — the per-engine home for
    dependencies that must not apply job-wide the way ``[hpc] setup`` does.
    Shell for the node that runs the engine; nothing is expanded at load.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: str | None = None
    bin: ExpandedPath | None = None
    pseudo_dir: ExpandedPath | None = None
    setup: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _one_way_to_name_the_code(self) -> QeEngineConfig:
        if self.command is not None and self.bin is not None:
            raise ValueError(
                "[engines.qe] sets both command and bin; pick one — command is "
                "the full invocation, bin constructs it (mpirun -np N bin/pw.x)"
            )
        return self


class LammpsEngineConfig(BaseModel):
    """Defaults for the built-in ``lammps`` engine (``[engines.lammps]``).

    Only machine facts live here — how to invoke the binary, and the
    ``setup`` lines (module loads, exports) its install needs, run in a
    private login-shell wrapper scoped to this engine's subprocess alone.
    The interatomic potential (``pair_style``/``pair_coeff``/``files``) is a
    science decision passed per call in ``calculator_options``, never a
    machine default.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    command: str | None = None
    setup: tuple[str, ...] = ()


class RootstockEngineConfig(BaseModel):
    """Where this machine's rootstock install lives (``[engines.rootstock]``).

    Machine facts for the built-in ``rootstock`` engine AND for serving
    checkpoint ids directly as engine names. ``root`` is the local-install
    form (the path holding ``envs/<name>/env_source.py``); ``cluster`` is
    the site-maintained form (a name rootstock's own cluster table knows) —
    nothing to do with ``[hpc] cluster``, which labels the SLURM cluster.
    Explicit ``calculator_options`` win key-by-key, and rootstock's own
    fallbacks (``$ROOTSTOCK_ROOT``, ``~/.config/rootstock/config.toml``)
    still apply when neither this section nor the caller says anything.
    The install location deliberately never enters cache identity:
    rootstock's contract is that canonical checkpoint ids are stable
    identities wherever they are served from.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    root: ExpandedPath | None = None
    cluster: str | None = None


class EnginesConfig(BaseModel):
    """Per-engine defaults (``[engines]``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    qe: QeEngineConfig = QeEngineConfig()
    lammps: LammpsEngineConfig = LammpsEngineConfig()
    rootstock: RootstockEngineConfig = RootstockEngineConfig()


class PathsConfig(BaseModel):
    """Where things live on this machine (``[paths]``).

    ``scratch`` is where slab-managed per-calculation scratch directories
    are created (unset = the platform default, $TMPDIR). On clusters set it
    to a *shared* scratch filesystem: SLURM's node-local $TMPDIR is often a
    few-GB tmpfs that pw.x's wavefunction files overflow, and MPI ranks on
    other nodes cannot see node-local files at all.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pseudos: ExpandedPath | None = None
    engines: ExpandedPath | None = None
    scratch: ExpandedPath | None = None


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


class SlabConfig(BaseModel):
    """The SLAB view of the merged configuration — every section optional.

    ``extra="ignore"`` at this level, because one file carries every
    package's tables and SLAB validates only its own. A table name that no
    package owns is still refused, by :func:`load_merged`, before any view
    sees it. Inside SLAB's own tables unknown keys stay forbidden, so a typo
    in ``[hpc]`` is refused naming the file.

    Examples:
        >>> SlabConfig().hpc.partitions
        {}
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    schema_version: int = 1
    paths: PathsConfig = PathsConfig()
    engines: EnginesConfig = EnginesConfig()
    hpc: HpcConfig = HpcConfig()

    @field_validator("schema_version")
    @classmethod
    def _understood_version(cls, value: int) -> int:
        if value > SCHEMA_VERSION:
            raise ValueError(
                f"config schema_version {value} is newer than this slab understands "
                f"({SCHEMA_VERSION}); upgrade slab-stack to read it"
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


T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class MergedConfig:
    """Every layer, merged and attributed, before any package validates it.

    One read of the files serves all three packages: ``raw`` is the deep
    merge, ``origins`` maps each dotted key to the file that supplied it, and
    ``files`` lists the layers in the order they were applied.
    """

    raw: dict[str, Any]
    origins: dict[str, str]
    files: tuple[tuple[str, Path], ...]


def load_merged(cwd: str | os.PathLike[str] | None = None) -> MergedConfig:
    """Read, version-check, and merge every layer, without validating tables.

    No files at all is the normal laptop case and yields an empty merge.
    A malformed file is a loud :class:`ConfigError` naming the file — a
    broken cluster config must surface, not degrade to defaults.

    A top-level key no package owns is refused here rather than by a view
    model. Each view ignores tables it does not own, so an unknown table
    would otherwise be ignored by all of them and configure nothing in
    silence.
    """
    merged: dict[str, Any] = {}
    origins: dict[str, str] = {}
    files = tuple(find_config_files(cwd))
    for layer, path in files:
        raw = _read_toml(path)
        # Each layer declares its own schema version, and each is checked:
        # merging first would let a project file's `schema_version = 1` mask a
        # site file written for a newer slab (whose other keys we would then
        # misread as if they meant what they mean today).
        _check_schema_version(raw, path)
        unknown = sorted(set(raw) - KNOWN_TOP_LEVEL_KEYS)
        if unknown:
            known = ", ".join(sorted(KNOWN_TOP_LEVEL_KEYS))
            raise ConfigError(
                f"{path} declares unknown top-level "
                f"{'keys' if len(unknown) > 1 else 'key'} {', '.join(repr(k) for k in unknown)}; "
                f"known: {known}"
            )
        _merge_into(merged, raw, origins, f"{path} ({layer})", prefix="")
    for dotted, moved_to in _MOVED_KEYS.items():
        if _has_dotted(merged, dotted):
            raise ConfigError(
                f"{_origin_for(dotted, origins)}: {dotted} moved to "
                f"{moved_to}; move the value into that table"
            )
    return MergedConfig(raw=merged, origins=origins, files=files)


def _has_dotted(raw: dict[str, Any], dotted: str) -> bool:
    """True when the merged mapping actually sets this dotted key."""
    node: Any = raw
    *parents, leaf = dotted.split(".")
    for part in parents:
        if not isinstance(node, dict):
            return False
        node = node.get(part)
    return isinstance(node, dict) and leaf in node


def validate(merged: MergedConfig, model: type[T]) -> T:
    """Validate one package's view of *merged*, naming the file on failure."""
    try:
        return model.model_validate(merged.raw)
    except ValidationError as e:
        raise ConfigError(_describe_validation_error(e, merged.origins)) from e


def load_config(cwd: str | os.PathLike[str] | None = None) -> SlabConfig:
    """Load every layer and validate the tables SLAB owns."""
    return validate(load_merged(cwd), SlabConfig)


def load_config_with_origins(
    cwd: str | os.PathLike[str] | None = None,
) -> tuple[SlabConfig, MergedConfig]:
    """:func:`load_config`, plus the merge that produced it (files, origins)."""
    merged = load_merged(cwd)
    return validate(merged, SlabConfig), merged


def config_value(dotted: str, cwd: str | os.PathLike[str] | None = None) -> Any:
    """One value from SLAB's own config by dotted key, or None when unset.

    The convenience accessor SLAB's integration points use (``pseudos_root``,
    the qe and lammps engines): a missing file or unset key is None; a
    *broken* file still raises. Foundation and Mason have their own.

    Examples:
        >>> import os, tempfile
        >>> os.environ.pop("SLAB_SITE_CONFIG", None) and None
        >>> os.environ.pop("SLAB_CONFIG", None) and None
        >>> os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp()
        >>> config_value("paths.pseudos", tempfile.mkdtemp()) is None
        True
    """
    node: Any = load_config(cwd)
    for part in dotted.split("."):
        if node is None:
            return None
        node = getattr(node, part, None)
    return node


def _check_schema_version(raw: dict[str, Any], path: Path) -> None:
    """Refuse one file written for a newer slab, before its keys are merged.

    This runs in :func:`load_merged`, so it is the check every package gets.
    A view model can only refuse a version in a table it validates, and each
    package validates only its own, so leaving the bound to ``SlabConfig``
    would let Foundation and Mason read a future file's keys as if they meant
    what they mean today.
    """
    if "schema_version" not in raw:
        return
    declared = raw["schema_version"]
    if not isinstance(declared, int) or isinstance(declared, bool):
        raise ConfigError(
            f"{path} declares schema_version {declared!r}, which is not an "
            f"integer; this slab understands {SCHEMA_VERSION}"
        )
    if declared > SCHEMA_VERSION:
        raise ConfigError(
            f"{path} declares schema_version {declared}, newer than this slab "
            f"understands ({SCHEMA_VERSION}); upgrade slab-stack to read it"
        )
    if declared < 1:
        raise ConfigError(f"{path} declares schema_version {declared}; it must be >= 1")


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
            if isinstance(target.get(key), dict) and not isinstance(value, dict):
                # A scalar replacing a whole table erases the table's leaves;
                # their origins must go with them, or the show command reports
                # phantom keys with confident attributions.
                for stale in [k for k in origins if k.startswith(f"{dotted}.")]:
                    del origins[stale]
            target[key] = value
            _set_origin_tree(origins, dotted, value, source)


def _set_origin_tree(origins: dict[str, str], dotted: str, value: Any, source: str) -> None:
    """Record origins for a value and, when it is a table, all its leaves."""
    if isinstance(value, dict):
        for key, child in value.items():
            _set_origin_tree(origins, f"{dotted}.{key}", child, source)
    else:
        origins[dotted] = source


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

# One file, three packages. Each table below is owned by exactly one of them:
# [paths], [engines], and [hpc] by slab; [workspace] by foundation; [agent] by
# mason. A table's owner validates it, so a typo inside [agent] is refused by
# 'mason doctor' rather than by 'slab engines list'. A table name no package
# owns is refused by every command that reads the file.

[workspace]
# root = "/scratch/${USER}/slab-workspace"        # run database + artifacts
#                                      # (parallel scratch works — slab falls back
#                                      # from WAL journaling automatically — but
#                                      # avoid heavy login-node polling of a
#                                      # workspace a running job is writing to)

[paths]
# pseudos = "/shared/sw/slab/pseudos"             # pseudopotential family root
# engines = "/shared/sw/slab/engines.json"        # cluster engine registry
# scratch = "/scratch/${USER}/slab-scratch"       # per-calculation scratch root:
#                                      # set on clusters — the default ($TMPDIR)
#                                      # is often node-local tmpfs, too small
#                                      # for pw.x wavefunctions and invisible
#                                      # to MPI ranks on other nodes

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
# bin = "/shared/sw/qe-7.4/bin"        # a custom install by its bin directory,
#                                      # instead of command: the command becomes
#                                      # 'mpirun -np N <bin>/pw.x' with N from
#                                      # $SLURM_NTASKS (1 outside a job), and a
#                                      # bundled <bin>/mpirun wins over PATH's.
#                                      # 'mason sandbox render' binds the whole
#                                      # install read-only automatically
# setup = ["module purge", "module load qe/7.4", "export OMP_NUM_THREADS=4"]
#                                      # THIS engine's dependencies: run in a
#                                      # private login-shell wrapper around the
#                                      # engine subprocess only — never job-wide
#                                      # like [hpc] setup; PATH and version are
#                                      # then checked inside that same shell.
#                                      # That wrapper INHERITS the job's env and
#                                      # your login profile, so the leading
#                                      # 'module purge' is how you choose what
#                                      # this engine starts from — drop it to
#                                      # build on the job's modules instead
# pseudo_dir = "/shared/sw/pseudos"    # used when no pseudo_family is given

[engines.lammps]
# command = "lmp"                      # login-node smoke tests / serial runs
# command = "srun lmp"                 # inside batch jobs only (same srun rule)
# command = "env OMP_NUM_THREADS=4 lmp"           # same env-wrapper rule as qe
# setup = ["module purge", "module load lammps/2025.07"]
#                                      # same per-engine setup rule as qe —
#                                      # purge first so whatever the job (or
#                                      # your profile) loaded can't reach it

[engines.rootstock]
# root = "/path/to/rootstock-install"  # a LOCAL rootstock install (the directory
#                                      # holding envs/<name>/env_source.py); lets
#                                      # served checkpoint ids work as engine names
#                                      # with no per-call options
# cluster = "delta"                    # OR a site-maintained install rootstock
#                                      # knows by name — this is rootstock's label,
#                                      # nothing to do with [hpc] cluster below

[hpc]
# cluster = "delta"                    # THIS SLURM cluster's name: run provenance and
#                                      # serve-record identity (job ids are per-cluster,
#                                      # so cross-cluster stop/status refuse) — not
#                                      # rootstock's cluster=, see [engines.rootstock]
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
# model = "meta-models/Muse-Glimmer-30B"    # as the server names it ('mason doctor'
#                                           # lists); an absolute path to a downloaded
#                                           # model directory also works, and never
#                                           # touches the network
# endpoint = "http://gpu-node-01:8000/v1"   # leave unset on a cluster: 'mason serve'
#                                           # records the URL of the node it landed on
# api_key_env = "SLAB_AGENT_API_KEY"        # NAME of the env var holding the key, never the key
# context_window = 131072                   # tokens the endpoint actually serves
# compact_at = 0.7                          # compact history at this fraction of the window
# max_turns = 60                            # model calls per goal; the agent stops loudly
# approval = "ask"                          # "ask" gates mutating tools; "auto" trusts them
# shell_allowlist = ["git status", "ls"]    # command prefixes that never need approval
# show_reasoning = true                     # 'mason chat' prints the model's reasoning
#                                           # between tool calls (needs the server's
#                                           # reasoning parser); false hides it. The
#                                           # transcript records reasoning either way
# software_notes = true                     # the system prompt carries curated notes on
#                                           # the engines this file enables; a file
#                                           # ~/.config/slab/notes/<engine>.md replaces
#                                           # a note for machine-local tweaks
# file_scope = "project"                    # file tools stay inside the project dir +
#                                           # workspace (skill dirs readable); "anywhere"
#                                           # lifts the fence. A workflow control, not a
#                                           # security boundary (the shell tool is gated
#                                           # separately by the allowlist + approval)
# session_lock = true                       # refuse a second concurrent mason in this
#                                           # workspace (false allows them)
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

# How 'mason serve' starts that server as a batch job. The GPU node is the
# scheduler's choice, so the endpoint URL is discovered, never configured.
# Compute nodes rarely have internet: download the model once, on the login node
# ('HF_HOME=/path/to/hf-cache hf download meta-models/Muse-Glimmer-30B'), and let
# the setup lines below serve it from that cache — offline, so a model missing
# from the cache fails loudly at startup instead of hanging on a download.
[agent.serve]
# partition = "gpu"                         # a partition from [hpc.partitions]
# time_limit = "08:00:00"                   # serve jobs are long-lived
# port = 8000
# tool_call_parser = "muse_glimmer"         # vLLM's --tool-call-parser: model-specific,
#                                           # and required for native tool calls
#                                           # ('vllm serve --help' lists your build's)
# args = [                                  # extra vllm flags. Some models pair the tool
#   "--reasoning-parser muse_glimmer",      # parser with a reasoning parser (Muse Glimmer
#   "--tensor-parallel-size 2",             # does; its vLLM recipe names both) — without
#   "--max-model-len 131072",               # the pair, tool calls arrive as broken markup
# ]
# setup = [                                 # compute-node shell, run before the server
#   "source /path/to/venvs/vllm/bin/activate",  # vLLM gets its own venv (its torch
#                                               # pin and mace-torch's rarely agree)
#   "export HF_HOME=/path/to/hf-cache",         # the cache 'hf download' filled
#   "export HF_HUB_OFFLINE=1",                  # serve from disk; refuse every download
# ]
# command = "..."                           # a server this schema does not model; must bind "$port"
# include_hpc_setup = false                 # serve jobs skip [hpc]-level setup by default:
#                                           # global engine module loads fight the server's
#                                           # venv; the partition's own setup still applies

# The no-network container for autonomous runs. 'mason sandbox render' derives
# the bind mounts from the tables above; this holds only what it cannot derive.
# [agent.sandbox]
# image = "/containers/slab-sandbox.sif"    # Apptainer image the job runs in
# binds = [                                 # extra binds (src:dest:mode) for what
#   "/opt/qe-7.4:/opt/qe-7.4:ro",           # derivation cannot see: engine installs
#   "/opt/openmpi:/opt/openmpi:ro",         # and their library closures ('ldd pw.x')
# ]
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
