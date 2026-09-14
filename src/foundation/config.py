"""Foundation's slice of the shared ``slab.toml``: the ``[workspace]`` table.

The file, the loader, the layering, and the origin tracking all live in
:mod:`slab.config`, because SLAB is the bottom package and its own engine
factories read configuration. What lives here is the one table Foundation
owns and the view that validates it.

``[workspace] root`` is where the run store and the artifact store live. It
sits below the explicit environment: ``-w/--workspace``, then
``$SLAB_WORKSPACE``, then this, then ``./.slab``. Like every other
configured value it supplies a *default* that resolves into a concrete path,
and the resolved path is what a run records. Configuration never reaches a
cache key.

``[workspace] exclude_gpus`` names the gpu ids of this machine that no
launch may hold, for a workstation with a broken device.
:func:`apply_gpu_exclusion` exports the list as ``SLAB_GPU_EXCLUDE`` for
the processes that reserve (``slab run``, the MCP server, the Mason
session), so their budget and every child's leave the ids out.
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from slab.config import ConfigError, ExpandedPath, gpu_id_list, load_merged, validate
from slab.resources import GPU_EXCLUDE_ENV


class WorkspaceConfig(BaseModel):
    """Where runs and artifacts live (``[workspace]``), and the gpus this machine excludes.

    Examples:
        >>> WorkspaceConfig().root is None
        True
        >>> WorkspaceConfig(exclude_gpus=[0]).exclude_gpus
        ('0',)
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    root: ExpandedPath | None = None
    exclude_gpus: tuple[str, ...] = ()

    @field_validator("exclude_gpus", mode="before")
    @classmethod
    def _gpu_ids(cls, value: object) -> object:
        return gpu_id_list(value)


class FoundationConfig(BaseModel):
    """The ``[workspace]`` view of the merged configuration.

    ``extra="ignore"`` at this level: the same file carries ``[paths]``,
    ``[engines]``, ``[hpc]``, and ``[agent]``, which other packages validate.
    Inside ``[workspace]`` unknown keys stay forbidden.

    Examples:
        >>> FoundationConfig().workspace.root is None
        True
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    workspace: WorkspaceConfig = WorkspaceConfig()


def load_config(cwd: str | os.PathLike[str] | None = None) -> FoundationConfig:
    """Load every layer and validate the tables Foundation owns."""
    return validate(load_merged(cwd), FoundationConfig)


def config_value(dotted: str, cwd: str | os.PathLike[str] | None = None) -> Any:
    """One value from Foundation's own config by dotted key, or None when unset.

    Examples:
        >>> import os, tempfile
        >>> os.environ.pop("SLAB_SITE_CONFIG", None) and None
        >>> os.environ.pop("SLAB_CONFIG", None) and None
        >>> os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp()
        >>> config_value("workspace.root", tempfile.mkdtemp()) is None
        True
    """
    node: Any = load_config(cwd)
    for part in dotted.split("."):
        if node is None:
            return None
        node = getattr(node, part, None)
    return node


def apply_gpu_exclusion(cwd: str | os.PathLike[str] | None = None) -> str | None:
    """Export ``[workspace] exclude_gpus`` as ``SLAB_GPU_EXCLUDE``; return what is in force.

    A value already in the environment wins, an empty one included: the
    sandbox render exports the partition's list, and an operator can
    override the file for one command. A config that cannot be read
    leaves the environment as it is, because the command that calls this
    reports a broken config itself.

    Examples:
        >>> import os, tempfile
        >>> for name in ("SLAB_SITE_CONFIG", "SLAB_CONFIG", "SLAB_GPU_EXCLUDE"):
        ...     _ = os.environ.pop(name, None)
        >>> os.environ["XDG_CONFIG_HOME"] = tempfile.mkdtemp()
        >>> project = tempfile.mkdtemp()
        >>> _ = open(os.path.join(project, "slab.toml"), "w").write(
        ...     "[workspace]\\nexclude_gpus = ['0']\\n")
        >>> apply_gpu_exclusion(project), os.environ["SLAB_GPU_EXCLUDE"]
        ('0', '0')
        >>> os.environ["SLAB_GPU_EXCLUDE"] = ""
        >>> apply_gpu_exclusion(project)
        ''
        >>> del os.environ["SLAB_GPU_EXCLUDE"]
    """
    if GPU_EXCLUDE_ENV in os.environ:
        return os.environ[GPU_EXCLUDE_ENV]
    try:
        ids = load_config(cwd).workspace.exclude_gpus
    except (ConfigError, OSError, ValueError):
        return None
    if not ids:
        return None
    os.environ[GPU_EXCLUDE_ENV] = ",".join(ids)
    return os.environ[GPU_EXCLUDE_ENV]
