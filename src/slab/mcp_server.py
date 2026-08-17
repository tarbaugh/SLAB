"""The agent surface: SLAB's CLI verbs exposed as MCP tools over stdio.

LLM agents are SLAB's primary user, so the workspace speaks their native
protocol. Every tool is a thin wrapper over :mod:`slab._ops` — exactly the
code the CLI runs — returning structured dicts instead of formatted text.

Start it with ``slab mcp`` (or configure your agent to do so):

.. code-block:: json

    {"mcpServers": {"slab": {"command": "slab", "args": ["mcp"]}}}

Requires the ``mcp`` extra: ``pip install 'slab[mcp]'``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer
except ImportError:  # pragma: no cover - mcp 1.x fallback
    from mcp.server.fastmcp import FastMCP as MCPServer  # type: ignore[no-redef]

from slab import _ops
from slab.lifecycle import LifecycleState
from slab.runtime import Workspace

_INSTRUCTIONS = """\
SLAB tracks materials-modeling runs through a lifecycle:
quarantined (ephemeral, expires) -> verified (checks passed) -> promoted (permanent).
Launch workflows with launch_workflow; inspect with list_runs/show_run; promote
what deserves keeping (promotion is the ONLY thing that makes data permanent);
expire_runs + gc reclaim everything else. Prefixes of run ids are accepted.
"""


def build_server(root: Path) -> MCPServer:
    """Build the MCP server for the workspace at *root* (does not start it)."""
    server = MCPServer("slab", instructions=_INSTRUCTIONS)

    @server.tool()
    def list_runs(
        state: str | None = None, status: str | None = None, limit: int = 20
    ) -> list[dict[str, Any]]:
        """List runs, newest first, optionally filtered by lifecycle state
        (quarantined/verified/promoted/archived/expired) and/or execution
        status (pending/running/completed/failed)."""
        with Workspace(root) as ws:
            return [
                _ops.run_summary(r)
                for r in ws.runs.list_runs(state=state, status=status, limit=limit)
            ]

    @server.tool()
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
    def gc(dry_run: bool = False) -> dict[str, Any]:
        """Drop artifact bytes no retention rule demands. References, hashes,
        and recipes always survive. dry_run=True only reports."""
        policy = _ops.load_policy(root)
        with Workspace(root) as ws:
            return ws.gc(policy, dry_run=dry_run).model_dump()

    @server.tool()
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
    def list_engines() -> dict[str, Any]:
        """What can be computed here: slab's built-in engines
        (emt/lj/mace/qe/rootstock — qe drives pw.x and needs only the
        executable plus pseudopotentials), everything this cluster's engine
        registry declares (LAMMPS, VASP, site aliases, ...) with the
        maintainer's declared versions, and — under 'rootstock' — the
        canonical MLIP checkpoint ids the cluster's rootstock install serves,
        each usable DIRECTLY as the engine= argument (e.g.
        engine='mace-mp-0-medium'). Also lists 'qe_protocols' (named QE input
        protocols: fast/balanced/stringent; expand one with
        slab.protocols.qe_protocol_options(atoms, protocol=...) inside a
        workflow script), 'pseudo_families' (installed pseudopotential
        families, usable as calculator_options={'pseudo_family': ...}), and
        'hpc' (this machine's configured SLURM cluster and partitions, or
        null off-cluster; jobs submit via 'slab hpc submit')."""
        return _ops.engines_overview()

    return server


def serve(root: Path) -> None:  # pragma: no cover - blocks on stdio
    """Run the MCP server on stdio until the client disconnects."""
    build_server(root).run()
