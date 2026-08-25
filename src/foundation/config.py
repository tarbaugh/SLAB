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
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, ConfigDict

from slab.config import ExpandedPath, load_merged, validate


class WorkspaceConfig(BaseModel):
    """Where runs and artifacts live (``[workspace]``).

    Examples:
        >>> WorkspaceConfig().root is None
        True
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    root: ExpandedPath | None = None


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
