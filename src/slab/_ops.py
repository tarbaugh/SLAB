"""What can be computed here: engines, protocols, pseudopotential families.

One capability report, assembled from the engine registry, the ambient
rootstock install, the named QE protocols, the installed pseudopotential
families, and the ``[hpc]`` config section. ``slab engines list`` renders it
as text and Foundation's MCP ``list_engines`` tool returns it as structure,
so the two cannot drift.
"""

from __future__ import annotations

import os
from typing import Any

from slab.errors import SlabError


def engines_overview(registry_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """What can be computed here: engines, QE protocols, pseudo families.

    Built-ins plus the cluster registry, rootstock checkpoint ids, the named
    QE input protocols, and the installed pseudopotential family names.
    Shared by ``slab engines list`` and the MCP ``list_engines`` tool.

    Examples:
        >>> import os
        >>> os.environ.pop("SLAB_ENGINES", None) and None
        >>> overview = engines_overview()
        >>> overview["builtin"]
        ['emt', 'lammps', 'lj', 'mace', 'qe', 'rootstock']
    """
    from slab.backends import available_engines
    from slab.engines import find_registry_path, load_registry

    registry = load_registry(registry_path)
    overview: dict[str, Any] = {
        "builtin": list(available_engines(None)),
        "registry": None,
        "rootstock": None,
    }
    if registry is not None:
        located = find_registry_path(registry_path)
        overview["registry"] = {
            "cluster": registry.cluster,
            "path": None if located is None else str(located),
            "notes": registry.notes,
            "engines": {
                name: {
                    "calculator": spec.calculator,
                    "version": spec.version,
                    "description": spec.description,
                    "verified_by_probe": spec.probe is not None,
                }
                for name, spec in sorted(registry.engines.items())
            },
        }
    overview["rootstock"] = _rootstock_checkpoints_overview()
    from slab.protocols import available_protocols

    overview["qe_protocols"] = list(available_protocols())
    try:
        from slab.pseudos import list_families

        overview["pseudo_families"] = [family.name for family, _ in list_families()]
    except SlabError as e:  # corrupt family: report, don't hide the engines
        overview["pseudo_families"] = []
        overview["pseudo_families_error"] = str(e)
    overview["hpc"] = _hpc_overview(overview)
    return overview


def _hpc_overview(overview: dict[str, Any]) -> dict[str, Any] | None:
    """The [hpc] config section as capability data, or None off-cluster."""
    from slab.config import load_config

    try:
        hpc = load_config().hpc
    except SlabError as e:  # broken config: report, don't hide the engines
        overview["hpc_error"] = str(e)
        return None
    if not hpc.partitions:
        return None
    return {
        "cluster": hpc.cluster,
        "default_partition": hpc.default_partition,
        "partitions": {
            name: {
                "description": spec.description,
                "time_limit": spec.time_limit,
                "gres": spec.gres,
            }
            for name, spec in sorted(hpc.partitions.items())
        },
    }


def _rootstock_checkpoints_overview() -> dict[str, Any] | None:
    """Checkpoint ids the ambient rootstock install declares, or None.

    These ids are usable directly as ``engine=`` names (silent serving); the
    install is found via rootstock's own defaults ($ROOTSTOCK_ROOT, then
    ~/.config/rootstock/config.toml).
    """
    try:
        from rootstock.config import resolve_default_root
        from rootstock.environment import list_declared_checkpoints
    except ImportError:
        return None
    root = resolve_default_root()
    if root is None:
        return None
    try:
        declared = list_declared_checkpoints(root)
    except OSError as e:
        return {"root": str(root), "error": str(e), "checkpoints": {}}
    return {
        "root": str(root),
        "checkpoints": {env: sorted(ids) for env, ids in sorted(declared.items())},
    }
