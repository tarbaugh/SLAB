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
        ['emt', 'lammps', 'lj', 'qe', 'rootstock']
        >>> overview["mp"] is None
        True
        >>> overview["gracemaker"] is None
        True
    """
    from slab.backends import available_engines
    from slab.config import ConfigError
    from slab.engines import find_registry_path, load_registry

    overview: dict[str, Any] = {
        "builtin": list(available_engines(None)),
        "registry": None,
        "rootstock": None,
    }
    try:
        registry = load_registry(registry_path)
    except ConfigError:
        # Locating the registry reads [paths] engines from the config. A
        # malformed config must not hide the built-in engines; the [hpc]
        # section below reports the config error once, as hpc_error.
        registry = None
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
    overview["mp"] = _mp_overview()
    overview["gracemaker"] = _gracemaker_overview()
    overview["hpc"] = _hpc_overview(overview)
    return overview


def _mp_overview() -> dict[str, Any] | None:
    """The offline Materials Project snapshot, or None when not configured.

    A configured-but-broken snapshot returns a dict carrying ``error`` —
    the overview must say the machine claims this data and cannot serve it,
    not silently look like a machine that never had it.
    """
    from slab.config import ConfigError, config_value
    from slab.errors import BuilderNotAvailableError
    from slab.mp import snapshot_info

    try:
        info = snapshot_info()
    except BuilderNotAvailableError:
        return None
    except ConfigError:
        # A malformed slab.toml is reported once, by the [hpc] section
        # (hpc_error); do not repeat it here.
        return None
    except SlabError as e:
        root = config_value("builders.mp.root")
        return {"root": None if root is None else str(root), "error": str(e)}
    return {
        "root": info["root"],
        "release": info["release"],
        "materials": info["materials"],
    }


def _gracemaker_overview() -> dict[str, Any] | None:
    """The gracemaker trainer, or None when ``[builders.gracemaker]`` is unset.

    Configured means the table sets a command or setup lines. The version is
    probed through the configured environment; a probe that fails reports
    ``version: None`` — the doctor row is where that becomes a diagnosis.
    """
    from slab.config import ConfigError, config_value

    try:
        command = config_value("builders.gracemaker.command")
        setup = config_value("builders.gracemaker.setup")
    except ConfigError:
        # A malformed slab.toml is reported once, by the [hpc] section
        # (hpc_error); do not repeat it here.
        return None
    if not command and not setup:
        return None
    from slab.gracemaker import gracemaker_command, gracemaker_version

    return {
        "command": gracemaker_command(),
        "version": gracemaker_version(),
    }


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

    These ids are usable directly as ``engine=`` names (silent serving). The
    install resolves with the same precedence :func:`slab.backends.get_calculator`
    uses: ``[engines.rootstock] root`` in slab.toml first, then
    ``[engines.rootstock] cluster``, then rootstock's own defaults
    (``$ROOTSTOCK_ROOT``, ``~/.config/rootstock/config.toml``). The overview
    would show ``null`` otherwise while a served ``engine=<checkpoint-id>``
    call succeeded, which was silently misleading — the calculator and the
    inventory now agree on where the install is.

    ``None`` means exactly one thing: the rootstock package is not
    installed. Every other outcome returns a dict, and a dict with no
    checkpoints carries an ``error`` line saying what to fix.
    """
    try:
        from rootstock.config import resolve_default_root
        from rootstock.environment import list_declared_checkpoints
    except ImportError:
        return None
    root, source, note = _resolved_rootstock_root(resolve_default_root)
    if root is None:
        # The package is installed, so this machine has a rootstock story —
        # a bare null here would hide the one actionable fact (where the
        # install root was looked for and why none was found).
        return {"root": None, "root_source": source, "error": note, "checkpoints": {}}
    try:
        declared = list_declared_checkpoints(root)
    except OSError as e:
        return {"root": str(root), "root_source": source, "error": str(e), "checkpoints": {}}
    return {
        "root": str(root),
        "root_source": source,
        "checkpoints": {env: sorted(ids) for env, ids in sorted(declared.items())},
    }


def _resolved_rootstock_root(default_root: Any) -> tuple[Any, str | None, str | None]:
    """Locate the rootstock install root the calculator layer would use.

    Returns ``(root, source, note)``: the path (or ``None`` when nothing
    declares one), a short label naming where it came from —
    ``"engines.rootstock.root"``, ``"engines.rootstock.cluster"``, or
    ``"rootstock defaults"`` — and, when the root is ``None``, one sentence
    saying what to fix. The label and the note ride into the overview so a
    viewer sees why the id list is what it is (or why it is empty).
    """
    from slab.backends import _rootstock_setting
    from slab.config import ConfigError

    try:
        root_setting = _rootstock_setting("root")
        cluster_setting = _rootstock_setting("cluster") if root_setting is None else None
    except ConfigError:
        # A malformed slab.toml is reported by the [hpc] section (with
        # `hpc_error`); do not double-raise it here — the rootstock section
        # falls back to rootstock's own defaults and stays useful.
        root_setting = None
        cluster_setting = None
    if root_setting is not None:
        from pathlib import Path

        return Path(root_setting).expanduser(), "engines.rootstock.root", None
    if cluster_setting is not None:
        try:
            from rootstock import get_root_for_cluster
        except ImportError:  # pragma: no cover - rootstock present if we got here
            return None, None, None
        try:
            return get_root_for_cluster(cluster_setting), "engines.rootstock.cluster", None
        except Exception as e:
            # An unknown cluster name is a live config problem; surface a
            # readable overview instead of the traceback.
            return (
                None,
                "engines.rootstock.cluster (unknown)",
                f"cluster {cluster_setting!r} is not known to this rootstock client: {e}",
            )
    root = default_root()
    if root is None:
        return (
            None,
            "rootstock defaults",
            "no install root configured — set root or cluster under "
            "[engines.rootstock] in slab.toml, or export ROOTSTOCK_ROOT",
        )
    return root, "rootstock defaults", None
