"""The agent surface: Foundation's CLI verbs exposed as MCP tools over stdio.

LLM agents are SLAB's primary user, so the workspace speaks their native
protocol. Every tool is a thin wrapper over :mod:`foundation._ops` — exactly
the code the CLI runs — returning structured dicts instead of formatted text.
The one exception is ``list_engines``, which reports what SLAB can compute and
so wraps :func:`slab._ops.engines_overview`.

Start it with ``slab mcp`` (or configure your agent to do so):

.. code-block:: json

    {"mcpServers": {"slab": {"command": "slab", "args": ["mcp"]}}}

Requires the ``mcp`` extra: ``pip install 'slab-stack[mcp]'``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # pragma: no cover - mcp 1.x fallback
    # Resolved dynamically: newer 2.x releases keep a ``mcp.server.fastmcp``
    # module without ``FastMCP``, and a static import there fails type
    # checking under exactly the SDK versions that never take this branch.
    from importlib import import_module

    MCPServer = import_module("mcp.server.fastmcp").FastMCP  # type: ignore[misc]
    ToolError = import_module("mcp.server.fastmcp.exceptions").ToolError  # type: ignore[misc]

from foundation import _ops
from foundation.errors import FoundationError
from foundation.lifecycle import LifecycleState
from foundation.runtime import Workspace
from slab._ops import engines_overview
from slab.errors import SlabError

_F = TypeVar("_F", bound=Callable[..., Any])


def _surfaced(fn: _F) -> _F:
    """Re-raise SLAB's own errors as ``ToolError`` so agents read them.

    ``ToolError`` is the SDK's contract for a message meant for the client;
    any other exception type is masked to a generic "Error executing tool
    ..." by mcp >= 2.1 (an internals-leak guard). SLAB's error messages ARE
    the product — "no run matches 'zzzz'" is the evidence an agent corrects
    from — so they must travel under the pass-through type. Unexpected
    exceptions stay masked, which is the guard working as intended.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except (FoundationError, SlabError, ValueError) as e:
            # ValueError is how a bad argument value (an unknown state name,
            # a negative limit, an empty id) is reported from the store.
            raise ToolError(str(e)) from e

    return wrapper  # type: ignore[return-value]


_INSTRUCTIONS = """\
SLAB tracks materials-modeling runs through a lifecycle:
quarantined (ephemeral, expires) -> verified (checks passed) -> promoted (permanent).
Launch workflows with launch_workflow; inspect with list_runs/show_run; promote
what deserves keeping (promotion is the ONLY thing that makes data permanent);
expire_runs + gc reclaim everything else. Prefixes of run ids are accepted.
Runs carry the client session that created them: list_sessions shows which
conversation produced which runs, and promote_session promotes a whole one.
"""


def build_server(root: Path) -> MCPServer:
    """Build the MCP server for the workspace at *root* (does not start it)."""
    server = MCPServer("foundation", instructions=_INSTRUCTIONS)

    @server.tool()
    @_surfaced
    def list_runs(
        state: str | None = None,
        status: str | None = None,
        session: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """List runs, newest first, optionally filtered by lifecycle state
        (quarantined/verified/promoted/archived/expired), execution status
        (pending/running/completed/failed), and/or the session that created
        them (full id or unique prefix; see list_sessions)."""
        with Workspace(root) as ws:
            return [
                _ops.run_summary(r)
                for r in ws.runs.list_runs(
                    state=state, status=status, session=session, limit=limit
                )
            ]

    @server.tool()
    @_surfaced
    def show_run(run_id: str) -> dict[str, Any]:
        """Everything about one run (id or unique prefix): state, intent,
        check results with the observed/expected values they compared, traced
        tasks with recipes, artifacts (and whether their bytes are still
        stored), and the lifecycle history. Failed runs and tasks carry a
        'failure' record — exception type, message, trimmed traceback, and
        diagnostic notes (e.g. relax notes its completed steps and last
        energy, and keeps the partial trajectory as an artifact) — the
        evidence for deciding a specific correction instead of retrying
        blind."""
        with Workspace(root) as ws:
            return _ops.run_details(ws, run_id)

    @server.tool()
    @_surfaced
    def promote_run(run_id: str, reason: str | None = None, force: bool = False) -> dict[str, Any]:
        """Make a run permanent (verified -> promoted). Give a reason — it is
        recorded as provenance. force=True promotes an unverified run and is
        recorded as forced."""
        with Workspace(root) as ws:
            run = ws.runs.transition(
                run_id, LifecycleState.PROMOTED, actor="agent", reason=reason, force=force
            )
            return _ops.run_summary(run)

    @server.tool()
    @_surfaced
    def list_sessions(limit: int = 20) -> dict[str, Any]:
        """List the client sessions that created runs, newest first: the
        session id, how many runs it produced, the lifecycle-state breakdown,
        and when its newest run was created. Runs that carry no session are
        reported once as a count."""
        with Workspace(root) as ws:
            return _ops.sessions_summary(ws, limit=limit)

    @server.tool()
    @_surfaced
    def promote_session(
        session: str, reason: str | None = None, force: bool = False
    ) -> dict[str, Any]:
        """Promote every run one session created (full id or unique prefix).
        Verified runs are promoted; already permanent runs are reported as
        such; unverified runs are skipped unless force=True; failed runs are
        skipped even then — promote those with promote_run, one at a time.
        The result reports every run considered, so read the outcomes."""
        with Workspace(root) as ws:
            return _ops.promote_session(
                ws, session, reason=reason, force=force, actor="agent"
            )

    @server.tool()
    @_surfaced
    def expire_runs(older_than: str | None = None, include_running: bool = False) -> dict[str, Any]:
        """Expire unpromoted runs past their TTL (state change only; gc drops
        bytes). older_than like '30d'/'12h' overrides the policy; '0d' expires
        everything unpromoted immediately. Runs stuck at status 'running'
        (hard-killed processes) are skipped unless include_running=True, which
        marks them failed first."""
        if older_than is not None:
            policy = _ops.ttl_override_policy(_ops.parse_duration_days(older_than))
        else:
            policy = _ops.load_policy(root)
        with Workspace(root) as ws:
            expired = ws.expire_due(policy, include_running=include_running)
            return {"expired": [_ops.run_summary(r) for r in expired], "count": len(expired)}

    @server.tool()
    @_surfaced
    def gc(dry_run: bool = False) -> dict[str, Any]:
        """Drop artifact bytes no retention rule demands. References, hashes,
        and recipes always survive. dry_run=True only reports."""
        policy = _ops.load_policy(root)
        with Workspace(root) as ws:
            return ws.gc(policy, dry_run=dry_run).model_dump()

    @server.tool()
    @_surfaced
    def launch_workflow(
        script_path: str, name: str | None = None, intent: str | None = None
    ) -> dict[str, Any]:
        """Execute a plain-Python workflow script in a fresh traced run.
        Always pass intent — why this run exists. The result includes the run
        id, final state (verified if all checks passed), and captured output;
        on failure it includes the structured 'failure' record (traceback and
        diagnostic notes). If recording the failure itself failed (storage
        died mid-crash), a raw 'traceback' string appears instead and the run
        may be left at status 'running'. Use show_run for per-task failure
        evidence."""
        return _ops.launch_script(root, script_path, name=name, intent=intent, capture_output=True)

    @server.tool()
    @_surfaced
    def list_engines() -> dict[str, Any]:
        """What can be computed here: slab's built-in engines
        (emt/lammps/lj/qe/rootstock — qe drives pw.x and needs only the
        executable plus pseudopotentials; lammps drives lmp and needs the
        executable plus your pair_style/pair_coeff/files potential options),
        everything this cluster's engine registry declares (VASP, site
        aliases, ...) with the
        maintainer's declared versions, and — under 'rootstock' — the
        canonical MLIP checkpoint ids the cluster's rootstock install serves,
        each usable DIRECTLY as the engine= argument (e.g.
        engine='mace-mp-0-medium'). Also lists 'qe_protocols' (named QE input
        protocols: fast/balanced/stringent; expand one with
        slab.protocols.qe_protocol_options(atoms, protocol=...) inside a
        workflow script), 'pseudo_families' (installed pseudopotential
        families, usable as calculator_options={'pseudo_family': ...}), and
        'hpc' (this machine's configured SLURM cluster and partitions, or
        null off-cluster; jobs submit via 'slab hpc submit'). An 'mp' key
        names the offline Materials Project snapshot when one is configured
        (search it with search_materials / get_material)."""
        return engines_overview()

    @server.tool()
    @_surfaced
    def search_materials(
        filters: dict[str, Any] | None = None,
        columns: list[str] | None = None,
        limit: int = 20,
        order_by: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search the offline Materials Project snapshot's materials table
        (parameterized SQL; needs [builders.mp] root in slab.toml). filters
        maps keys to values: 'elements' (all must be present) and
        'exclude_elements' take element-symbol lists; any other key is a
        materials column, bare for equality (null matches SQL NULL) or
        suffixed __lte/__gte/__lt/__gt/__ne for comparisons — e.g.
        {"elements": ["Fe"], "energy_above_hull__lte": 0.025}. Unknown
        columns are refused with the real column list. limit clamps to
        1-500; order_by names a column, leading '-' for descending. NULL
        means "not populated", never zero. Report results as
        (snapshot release, material_id); absence from the snapshot is
        absence — there is no online fallback."""
        from slab.mp import search_materials as mp_search

        return mp_search(filters, columns=columns, limit=limit, order_by=order_by)

    @server.tool()
    @_surfaced
    def get_material(material_id: str) -> dict[str, Any]:
        """One material's full metadata record from the offline Materials
        Project snapshot: the materials row, its 'elements' list, and
        'cif_file' — the absolute path of its archived CIF, readable by
        ase.io.read (or fetched traced via foundation.tasks.fetch_structure
        inside a workflow). Raises when the id is absent: the snapshot is
        the only source, and there is no online fallback."""
        from slab.mp import get_material as mp_get

        return mp_get(material_id)

    return server


def serve(root: Path) -> None:  # pragma: no cover - blocks on stdio
    """Run the MCP server on stdio until the client disconnects."""
    build_server(root).run()
