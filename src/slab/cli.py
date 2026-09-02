"""The machine groups of the ``slab`` command.

These groups describe what can be computed here and how to reach it:
``engines`` inspects and verifies the cluster registry, ``pseudos``
installs and checks pseudopotential families, ``protocols`` shows the
named Quantum ESPRESSO input protocols, ``hpc`` renders and submits SLURM
jobs, and ``config`` explains where every setting came from.

The front door that mounts them is :mod:`slab_stack.cli`. Runs, artifacts,
and verification are ``slab run`` and its siblings; the resident agent is
``slab mason``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, NoReturn

import typer

from slab._ops import engines_overview
from slab.errors import SlabError

app = typer.Typer(
    help="SLAB — access to atomistic engines, registries, protocols, "
    "pseudopotentials, and the scheduler.",
    no_args_is_help=True,
    add_completion=False,
)


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code=1)


engines_app = typer.Typer(
    help="Inspect and verify the cluster engine registry (rootstock-style software management).",
    no_args_is_help=True,
)
app.add_typer(engines_app, name="engines")

_RegistryOpt = Annotated[
    Path | None,
    typer.Option(
        "--registry",
        help="Engine registry JSON file (default: $SLAB_ENGINES, then ~/.config/slab/).",
    ),
]


@engines_app.command("list")
def engines_list(registry_path: _RegistryOpt = None) -> None:
    """List available engines: built-ins plus everything this cluster declares."""
    try:
        overview = engines_overview(registry_path)
    except (SlabError, OSError, ValueError) as e:
        _fail(str(e))
    typer.echo(f"built-in: {', '.join(overview['builtin'])}")
    registry = overview["registry"]
    if registry is None:
        typer.echo("registry: none configured (set $SLAB_ENGINES to a cluster's engines.json)")
    else:
        where = f" [{registry['cluster']}]" if registry["cluster"] else ""
        typer.echo(f"registry{where}: {registry['path']}")
        for name, spec in registry["engines"].items():
            version = spec["version"] or "unversioned"
            probe = "probe" if spec["verified_by_probe"] else "no probe"
            description = f"  {spec['description']}" if spec["description"] else ""
            typer.echo(f"  {name:<14} {version:<14} {spec['calculator']}  ({probe}){description}")
    rootstock = overview["rootstock"]
    if rootstock is not None:
        where = rootstock["root"] or "(no root configured)"
        typer.echo(f"rootstock checkpoints (usable directly as engine=): {where}")
        if rootstock.get("error"):
            typer.echo(f"  error reading install: {rootstock['error']}")
        for env_name, ids in rootstock["checkpoints"].items():
            typer.echo(f"  {env_name}: {', '.join(ids)}")
    typer.echo(f"qe protocols: {', '.join(overview['qe_protocols'])} ('slab protocols show')")
    families = overview["pseudo_families"]
    if overview.get("pseudo_families_error"):
        typer.echo(f"pseudo families: error — {overview['pseudo_families_error']}")
    elif families:
        typer.echo(f"pseudo families: {', '.join(families)}")
    else:
        typer.echo("pseudo families: none installed ('slab pseudos install sssp' fetches one)")
    mp = overview.get("mp")
    if mp is not None:
        if mp.get("error"):
            typer.echo(f"mp snapshot: error — {mp['error']}")
        else:
            release = f"release {mp['release']}" if mp["release"] else "release unknown"
            typer.echo(
                f"mp snapshot: {release}, {mp['materials']} materials ('slab mp info')"
            )
    gracemaker = overview.get("gracemaker")
    if gracemaker is not None:
        version = (
            f"tensorpotential {gracemaker['version']}"
            if gracemaker["version"]
            else "version unprobeable ('slab doctor')"
        )
        typer.echo(f"gracemaker trainer: {version} via {gracemaker['command']}")
    hpc = overview.get("hpc")
    if overview.get("hpc_error"):
        typer.echo(f"hpc partitions: error — {overview['hpc_error']}")
    elif hpc:
        cluster = f" [{hpc['cluster']}]" if hpc["cluster"] else ""
        default = f" (default {hpc['default_partition']})" if hpc["default_partition"] else ""
        typer.echo(
            f"hpc partitions{cluster}: {', '.join(hpc['partitions'])}{default} "
            f"('slab hpc partitions')"
        )


@engines_app.command("verify")
def engines_verify(registry_path: _RegistryOpt = None) -> None:
    """Run every registry engine's probe; exit nonzero if any engine fails."""
    from slab.engines import load_registry, verify_engines

    try:
        registry = load_registry(registry_path)
    except (SlabError, OSError, ValueError) as e:
        _fail(str(e))
    if registry is None:
        _fail("no engine registry configured (set $SLAB_ENGINES or pass --registry)")
    results = verify_engines(registry)
    failed = 0
    for result in results:
        mark = "+" if result.ok else "x"
        typer.echo(f"[{mark}] {result.engine}: {result.detail}")
        failed += 0 if result.ok else 1
    typer.echo(f"{len(results) - failed}/{len(results)} engines verified")
    if failed:
        raise typer.Exit(code=1)


pseudos_app = typer.Typer(
    help="Install and inspect pseudopotential families (aiida-pseudo's pattern).",
    no_args_is_help=True,
)
app.add_typer(pseudos_app, name="pseudos")


@pseudos_app.command("install")
def pseudos_install(
    kind: Annotated[str, typer.Argument(help="Family kind; only 'sssp' installs today.")] = "sssp",
    version: Annotated[
        str, typer.Option("--version", "-v", help="SSSP version, e.g. 1.3.")
    ] = "1.3",
    functional: Annotated[
        str, typer.Option("--functional", "-x", help="XC functional: PBE or PBEsol.")
    ] = "PBEsol",
    precision: Annotated[
        str, typer.Option("--precision", "-p", help="SSSP variant: efficiency or precision.")
    ] = "efficiency",
    force: Annotated[bool, typer.Option("--force", help="Replace an existing install.")] = False,
) -> None:
    """Download and verify a pseudopotential family from its official archive."""
    from slab.pseudos import family_digest, install_sssp, pseudos_root_origin

    if kind.strip().lower() != "sssp":
        _fail(
            f"only 'sssp' families install today, not {kind!r} (PseudoDojo is served over "
            f"unverified HTTP upstream; point pseudo_dir= at your own files instead)"
        )
    try:
        root, origin = pseudos_root_origin()
    except SlabError as e:  # a broken config is a plain error here too
        _fail(str(e))
    typer.echo(f"installing into {root} ({origin})")
    typer.echo(f"downloading SSSP {version} {functional} {precision} from Materials Cloud ...")
    try:
        family, directory = install_sssp(
            version=version, functional=functional, precision=precision, force=force
        )
    except SlabError as e:
        _fail(str(e))
    typer.echo(
        f"installed {family.name} ({len(family.elements)} elements, "
        f"digest {family_digest(family)}) at {directory}"
    )


@pseudos_app.command("list")
def pseudos_list() -> None:
    """List installed pseudopotential families."""
    from slab.pseudos import family_digest, list_families, pseudos_root

    try:
        families = list_families()
    except SlabError as e:
        _fail(str(e))
    typer.echo(f"root: {pseudos_root()}")
    if not families:
        typer.echo("no families installed ('slab pseudos install sssp' fetches one)")
        return
    for family, directory in families:
        typer.echo(
            f"  {family.name:<34} {len(family.elements):>3} elements  "
            f"digest {family_digest(family)}  {directory}"
        )


@pseudos_app.command("verify")
def pseudos_verify(
    name: Annotated[str, typer.Argument(help="Installed family name (version prefix ok).")],
) -> None:
    """Re-hash a family's files against its manifest; exit nonzero on mismatch."""
    from slab.pseudos import find_family, verify_family

    try:
        family, directory = find_family(name)
    except SlabError as e:
        _fail(str(e))
    problems = verify_family(family, directory)
    for problem in problems:
        typer.echo(f"[x] {problem}")
    if problems:
        raise typer.Exit(code=1)
    typer.echo(f"[+] {family.name}: all {len(family.elements)} files match their checksums")


hpc_app = typer.Typer(
    # Plain 'hpc', not '[hpc]': rich would swallow the brackets as markup.
    help="SLURM plumbing driven by the hpc section of the slab config.",
    no_args_is_help=True,
)
app.add_typer(hpc_app, name="hpc")


@hpc_app.command("partitions")
def hpc_partitions() -> None:
    """List the partitions this machine's config declares."""
    from slab.config import load_config

    try:
        hpc = load_config().hpc
    except SlabError as e:
        _fail(str(e))
    if not hpc.partitions:
        typer.echo(
            "no partitions declared (add [hpc.partitions.NAME] tables to the slab "
            "config; 'slab config init' shows the shape)"
        )
        return
    if hpc.cluster:
        typer.echo(f"cluster: {hpc.cluster}")
    for name in sorted(hpc.partitions):
        spec = hpc.partitions[name]
        default = " (default)" if name == hpc.default_partition else ""
        time_limit = spec.time_limit or "no time limit set"
        extras = ", ".join(
            piece
            for piece in (spec.gres, spec.mem and f"mem {spec.mem}", spec.qos and f"qos {spec.qos}")
            if piece
        )
        detail = f"  {extras}" if extras else ""
        description = f"  {spec.description}" if spec.description else ""
        typer.echo(f"  {name:<12}{default:<10} {time_limit}{detail}{description}")


@hpc_app.command("render")
def hpc_render(
    command: Annotated[
        str, typer.Argument(help="Command the job runs, e.g. 'slab run relax.py'.")
    ],
    name: Annotated[str, typer.Option("--name", "-n", help="Job name.")] = "slab-job",
    partition: Annotated[
        str | None, typer.Option("--partition", "-p", help="Partition (default: config's).")
    ] = None,
    time_limit: Annotated[
        str | None, typer.Option("--time", help="Override the partition's time limit.")
    ] = None,
) -> None:
    """Render the sbatch script that submit would use — read before trusting."""
    from slab.hpc import render_sbatch

    try:
        script = render_sbatch(command, job_name=name, partition=partition, time_limit=time_limit)
    except SlabError as e:
        _fail(str(e))
    typer.echo(script)


@hpc_app.command("submit")
def hpc_submit(
    command: Annotated[
        str, typer.Argument(help="Command the job runs, e.g. 'slab run relax.py'.")
    ],
    name: Annotated[str, typer.Option("--name", "-n", help="Job name.")] = "slab-job",
    partition: Annotated[
        str | None, typer.Option("--partition", "-p", help="Partition (default: config's).")
    ] = None,
    time_limit: Annotated[
        str | None, typer.Option("--time", help="Override the partition's time limit.")
    ] = None,
    directory: Annotated[
        Path | None, typer.Option("--dir", help="Where the job runs (default: cwd).")
    ] = None,
) -> None:
    """Render and submit a job; the exact script is kept next to its outputs."""
    from slab.config import load_config
    from slab.hpc import render_sbatch, submit

    try:
        hpc = load_config().hpc
        resolved, _spec = hpc.resolve_partition(partition)
        script = render_sbatch(
            command, job_name=name, partition=resolved, config=hpc, time_limit=time_limit
        )
        job = submit(script, job_name=name, partition=resolved, directory=directory)
    except SlabError as e:
        _fail(str(e))
    typer.echo(f"submitted job {job.job_id} ({job.job_name}) to {job.partition}")
    typer.echo(f"script: {job.script_path}")


@hpc_app.command("status")
def hpc_status(
    job_id: Annotated[str, typer.Argument(help="SLURM job id.")],
) -> None:
    """One job's state (squeue first, sacct fallback)."""
    from slab.hpc import job_state

    try:
        status = job_state(job_id)
    except SlabError as e:
        _fail(str(e))
    raw = f"  ({status.raw})" if status.raw and status.raw != status.state.value.upper() else ""
    detail = f"  {status.detail}" if status.detail else ""
    typer.echo(f"job {status.job_id}: {status.state.value}{raw}{detail}")


@hpc_app.command("cancel")
def hpc_cancel(
    job_id: Annotated[str, typer.Argument(help="SLURM job id.")],
) -> None:
    """Cancel a job (idempotent: cancelling a finished job is a no-op)."""
    from slab.hpc import cancel

    try:
        cancel(job_id)
    except SlabError as e:
        _fail(str(e))
    typer.echo(f"cancel requested for job {job_id}")


config_app = typer.Typer(
    help="Layered TOML configuration: site, user, and project files merged key-by-key.",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show(
    as_json: Annotated[bool, typer.Option("--json", help="Emit machine-readable JSON.")] = False,
) -> None:
    """Show the merged configuration and which file each value came from.

    One file, three packages. SLAB's own tables print through their validated
    model, so a ``~`` or ``$VAR`` shows the path it expanded to. The tables
    Foundation and Mason own print as written, because SLAB does not import
    their models — it knows their names, not their meanings. A value refused
    by its owner therefore still appears here, which keeps this the command
    that answers why a setting was not applied.
    """
    from slab.config import KNOWN_TOP_LEVEL_KEYS, SlabConfig, load_config_with_origins

    try:
        config, merge = load_config_with_origins()
    except SlabError as e:
        _fail(str(e))
    files, origins = merge.files, merge.origins
    # Ownership comes from the model itself, so a table added to SlabConfig
    # can never be misrouted here by a stale hand-kept list.
    slab_tables = frozenset(SlabConfig.model_fields)
    resolved = config.model_dump()

    def value_of(dotted: str) -> object:
        """SLAB's tables resolved; everyone else's exactly as the file says."""
        owned = dotted.split(".", 1)[0] in slab_tables
        return _dig(resolved if owned else merge.raw, dotted)

    if as_json:
        typer.echo(
            json.dumps(
                {
                    "files": [{"layer": layer, "path": str(path)} for layer, path in files],
                    "config": config.model_dump(mode="json"),
                    "set": {dotted: value_of(dotted) for dotted in sorted(origins)},
                    "origins": origins,
                },
                indent=2,
                default=str,
            )
        )
        return
    if not files:
        typer.echo(
            "no config files found (site: $SLAB_SITE_CONFIG, user: "
            "~/.config/slab/config.toml, project: ./slab.toml or $SLAB_CONFIG); "
            "'slab config init' writes a template"
        )
        return
    for layer, path in files:
        typer.echo(f"{layer}: {path}")
    for dotted in sorted(origins):
        typer.echo(f"  {dotted} = {value_of(dotted)!r}  [{origins[dotted]}]")
    unowned = ", ".join(sorted(KNOWN_TOP_LEVEL_KEYS - slab_tables))
    typer.echo(f"[{unowned}] print as written; their owners validate them")
    typer.echo("unset keys use built-in defaults ('slab config init' shows them all)")


def _dig(mapping: object, dotted: str) -> object:
    """Fetch a dotted key path out of nested mappings, and only mappings.

    Both sources are plain dicts (the model is dumped first), so a path that
    leaves the mapping world answers None rather than reaching into whatever
    Python object happens to sit there. A config value must never render as
    a bound method.
    """
    node = mapping
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


@config_app.command("init")
def config_init(
    path: Annotated[
        Path, typer.Argument(help="Where to write the template (default ./slab.toml).")
    ] = Path("slab.toml"),
    user: Annotated[
        bool, typer.Option("--user", help="Write the user-layer file (~/.config/slab/) instead.")
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Replace an existing file.")] = False,
) -> None:
    """Write a fully commented configuration template."""
    from slab.config import user_config_path, write_template

    target = user_config_path() if user else path
    try:
        written = write_template(target, force=force)
    except SlabError as e:
        _fail(str(e))
    typer.echo(f"wrote {written}")


protocols_app = typer.Typer(
    help="Named QE input protocols (adapted from aiida-quantumespresso).",
    no_args_is_help=True,
)
app.add_typer(protocols_app, name="protocols")


@protocols_app.command("list")
def protocols_list() -> None:
    """List the named protocols and what they trade off."""
    from slab.protocols import available_protocols, get_protocol

    for name in available_protocols():
        protocol = get_protocol(name)
        typer.echo(f"{name:<10} {protocol.description}")


@protocols_app.command("show")
def protocols_show(
    name: Annotated[str, typer.Argument(help="fast, balanced, or stringent.")],
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Show one protocol's data (QE-native units) and derived values."""
    from slab.protocols import protocol_details

    try:
        details = protocol_details(name)
    except SlabError as e:
        _fail(str(e))
    if as_json:
        typer.echo(json.dumps(details, indent=1, sort_keys=True))
        return
    for key in sorted(details):
        typer.echo(f"{key}: {details[key]}")


mp_app = typer.Typer(
    help="Query the offline Materials Project snapshot (a local, read-only data source).",
    no_args_is_help=True,
)
app.add_typer(mp_app, name="mp")

#: Columns 'slab mp search' shows when the caller names none. Filtered to
#: what the installed snapshot actually has — schemas vary by build.
_PREFERRED_SEARCH_COLUMNS = (
    "material_id",
    "formula_pretty",
    "energy_above_hull",
    "band_gap",
    "is_stable",
)


@mp_app.command("info")
def mp_info(
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Show the installed snapshot's provenance: root, release, scale."""
    from slab.mp import snapshot_info

    try:
        info = snapshot_info()
    except SlabError as e:
        _fail(str(e))
    if as_json:
        typer.echo(json.dumps(info, indent=1, sort_keys=True, default=str))
        return
    typer.echo(f"root: {info['root']}")
    typer.echo(f"release: {info['release'] or 'not recorded'}")
    typer.echo(f"materials: {info['materials']}")
    manifest = info.get("manifest")
    if isinstance(manifest, dict):
        typer.echo(f"manifest: {len(manifest)} keys ('slab mp info --json' shows all)")
    elif info.get("manifest_error"):
        typer.echo(f"manifest: {info['manifest_error']}")
    else:
        typer.echo("manifest: none")


@mp_app.command("search")
def mp_search(
    element: Annotated[
        list[str] | None,
        typer.Option("--element", "-e", help="Require this element (repeatable)."),
    ] = None,
    exclude_element: Annotated[
        list[str] | None,
        typer.Option("--exclude-element", help="Refuse this element (repeatable)."),
    ] = None,
    filters: Annotated[
        list[str] | None,
        typer.Option(
            "--filter",
            "-f",
            help="key=value filter (repeatable). Bare keys test equality; "
            "suffixes __lte/__gte/__lt/__gt/__ne compare; value 'null' means "
            "SQL NULL. Example: -f energy_above_hull__lte=0.025",
        ),
    ] = None,
    columns: Annotated[
        str | None,
        typer.Option("--columns", help="Comma-separated columns to return."),
    ] = None,
    order_by: Annotated[
        str | None,
        typer.Option("--order-by", help="Column to sort by; prefix - for descending."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Max rows (1-500).")] = 20,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Search the snapshot's materials table with indexed, parameterized SQL."""
    from slab.mp import search_materials

    mapping: dict[str, object] = {}
    if element:
        mapping["elements"] = list(element)
    if exclude_element:
        mapping["exclude_elements"] = list(exclude_element)
    for pair in filters or []:
        key, separator, raw = pair.partition("=")
        if not separator or not key:
            _fail(f"--filter takes key=value, got {pair!r}")
        mapping[key] = _filter_value(raw)
    requested = [name.strip() for name in columns.split(",")] if columns else None
    try:
        rows = search_materials(
            mapping, columns=requested, limit=limit, order_by=order_by
        )
    except SlabError as e:
        _fail(str(e))
    if as_json:
        typer.echo(json.dumps(rows, indent=1, default=str))
        return
    if not rows:
        typer.echo("no materials match")
        return
    shown = requested or [
        name for name in _PREFERRED_SEARCH_COLUMNS if name in rows[0]
    ] or list(rows[0])
    for row in rows:
        typer.echo("  ".join(f"{name}={row.get(name)}" for name in shown))
    typer.echo(f"{len(rows)} row{'s' if len(rows) != 1 else ''} ('slab mp show <id>')")


@mp_app.command("show")
def mp_show(
    material_id: Annotated[str, typer.Argument(help="One material id, e.g. mp-149.")],
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Show one material's full metadata record and its CIF path."""
    from slab.mp import get_material

    try:
        record = get_material(material_id)
    except SlabError as e:
        _fail(str(e))
    if as_json:
        typer.echo(json.dumps(record, indent=1, sort_keys=True, default=str))
        return
    for key in sorted(record):
        value = record[key]
        if key == "elements":
            value = ", ".join(value)
        typer.echo(f"{key}: {value}")


def _filter_value(raw: str) -> object:
    """One --filter value: 'null' is SQL NULL, true/false are 1/0, numbers are numbers.

    The snapshot stores booleans as integers, so ``is_stable=true`` must
    compare against 1, not the string.

    Examples:
        >>> _filter_value("0.025")
        0.025
        >>> _filter_value("Fe2O3")
        'Fe2O3'
        >>> _filter_value("true")
        1
    """
    if raw.lower() in {"null", "none"}:
        return None
    if raw.lower() in {"true", "false"}:
        return 1 if raw.lower() == "true" else 0
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


if __name__ == "__main__":  # pragma: no cover - module execution convenience
    app()
