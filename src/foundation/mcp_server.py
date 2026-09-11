"""The agent surface: Foundation's operations exposed as MCP tools over stdio.

LLM agents are SLAB's primary user, so the workspace speaks their native
protocol. Every tool is a thin wrapper over :mod:`foundation._ops` — the
code the CLI runs — returning structured dicts instead of formatted text.
The exceptions wrap the neighbours ``_ops`` itself calls: ``list_engines``
reports what SLAB can compute (:func:`slab._ops.engines_overview`), and the
materials tools read the offline snapshot (:mod:`slab.mp`).

The tool set is the resident agent's, minus what an external harness has
of its own (files, a shell) and minus the harness mechanisms that belong
to one agent loop: ``delegate``, ``review``, and ``finish`` stay Mason's.
A harness reports what it found with ``report_results``, which records the
session's answer in the workspace, where the benchmark scorer reads it.

Start it with ``slab mcp`` (or configure your agent to do so):

.. code-block:: json

    {"mcpServers": {"slab": {"command": "slab", "args": ["mcp"]}}}

Requires the ``mcp`` extra: ``pip install 'slab-stack[mcp]'``.
"""

from __future__ import annotations

import functools
import os
import sqlite3
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
from foundation import memory as memory_store
from foundation import project as project_files
from foundation.errors import FoundationError, MemoryStoreError
from foundation.lifecycle import LifecycleState
from foundation.runtime import Workspace
from foundation.session_record import SessionRecord, new_session_id, validate_results
from foundation.skills import SkillError, bundled_files, discover_skills
from slab._ops import engines_overview
from slab.config import load_config as load_slab_config
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


def _positive(name: str, value: int | None, *, minimum: int = 1) -> None:
    """Refuse a size argument below *minimum*; None means the size is absent.

    The schema already refuses a non-integer; a zero or a negative count
    is a mistake the client should hear about, not a launch of one.

    Examples:
        >>> _positive("ntasks", 2), _positive("ntasks", None), _positive("gpus", 0, minimum=0)
        (None, None, None)
        >>> _positive("ntasks", 0)
        Traceback (most recent call last):
        ...
        ValueError: ntasks must be a positive integer, not 0
    """
    if value is None:
        return
    if isinstance(value, bool) or value < minimum:
        wanted = "a positive integer" if minimum > 0 else "zero or a positive integer"
        raise ValueError(f"{name} must be {wanted}, not {value!r}")


_INSTRUCTIONS = """\
SLAB tracks materials-modeling runs through a lifecycle:
quarantined (ephemeral, expires) -> verified (checks passed) -> promoted (permanent).
Launch workflows with launch_workflow; inspect with list_runs/show_run; promote
what deserves keeping (promotion is the ONLY thing that makes data permanent);
expire_runs + gc reclaim everything else. Prefixes of run ids are accepted.
Runs carry the client session that created them: list_sessions shows which
conversation produced which runs, and promote_session promotes a whole one.
This server's session id is {session}. Every run it launches carries it, and
report_results records the session's answer (results with units, and the run
ids that produced them) where a benchmark scorer reads it. Call retire_session
last: it promotes the verified runs the answer cites and expires this session's
other runs, and it cannot be undone.
The project directory is {project}: its NOTEBOOK.md and PLAN.md are the notebook
and plan tools' files, and its skills/ directory adds to the skill catalog.
"""


def build_server(
    root: Path, *, project: Path | None = None, session: str | None = None
) -> MCPServer:
    """Build the MCP server for the workspace at *root* (does not start it).

    *project* is the directory whose ``slab.toml``, notebook, plan, and
    skills apply; the current directory when omitted, as for the CLI.
    *session* is the id every run this server launches carries; a fresh
    ``mcp-<stamp>-<pid>`` when omitted.
    """
    project_dir = Path(project if project is not None else Path.cwd()).resolve()
    session_id = session or new_session_id("mcp")
    record = SessionRecord(root, session_id, client="mcp")
    hpc = load_slab_config(project_dir).hpc
    versions: dict[str, dict[str, str]] = {}
    recorded_runs: set[str] = set()

    def _record_run_commands(run_id: str | None, tool: str) -> None:
        # The session record says what a run ran, the way a Mason
        # transcript does: one command event per distinct engine command
        # the run's tasks resolved, recorded once per run.
        if not run_id or run_id in recorded_runs:
            return
        recorded_runs.add(run_id)
        try:
            with Workspace(root) as ws:
                entries = _ops.run_commands(ws, run_id)
        except (FoundationError, sqlite3.Error, OSError) as e:
            record.record(
                {
                    "type": "command",
                    "kind": "engine",
                    "tool": tool,
                    "run_id": run_id,
                    "error": str(e),
                }
            )
            return
        for entry in entries:
            record.record({"type": "command", "kind": "engine", "tool": tool, **entry})

    def software_versions() -> dict[str, str]:
        # Probed once per server: the version checks run engines and builders.
        if "live" not in versions:
            from slab._ops import software_versions as probe

            versions["live"] = probe()
        return versions["live"]

    server = MCPServer(
        "foundation",
        instructions=_INSTRUCTIONS.format(session=session_id, project=project_dir),
    )

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
        them (full id or unique prefix; see list_sessions). A running run
        whose recorded process on this host is gone is marked failed first,
        so 'running' means a live process or one on another host."""
        with Workspace(root) as ws:
            ws.reap_dead(caller="list_runs")
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
        script_path: str,
        name: str | None = None,
        intent: str | None = None,
        ntasks: int | None = None,
        threads: int | None = None,
        gpus: int | None = None,
    ) -> dict[str, Any]:
        """Execute a plain-Python workflow script in a fresh traced run that
        carries this server's session id. Always pass intent — why this run
        exists. Size the launch with ntasks (MPI ranks), threads (per rank),
        and gpus: the server reserves that slice of this host before the
        run starts, the run takes it as an affinity mask plus
        CUDA_VISIBLE_DEVICES, and a slice that does not fit is refused with
        the free amounts (list_engines reports 'budget' and 'free'). A
        sized launch runs as a child process; an unsized one runs here and
        reserves every free cpu, and gpus without ntasks takes every free
        cpu and the gpus asked. The result includes the run id, final
        state (verified if all checks passed), the 'resources' it held, and
        captured output; on failure it includes the structured 'failure'
        record (traceback and diagnostic notes). If recording the failure
        itself failed (storage died mid-crash), a raw 'traceback' string
        appears instead and the run may be left at status 'running'. Use
        show_run for per-task failure evidence."""
        import shlex

        _positive("ntasks", ntasks)
        _positive("threads", threads)
        _positive("gpus", gpus, minimum=0)
        sized = any(value is not None for value in (ntasks, threads, gpus))
        with Workspace(root) as ws:
            ws.reap_dead(caller="launch_workflow")
            reservation = ws.reserve(
                ntasks=ntasks, threads=threads, gpus=gpus or 0, holder_pid=os.getpid()
            )
        driver = ["slab", "run", script_path]
        if name:
            driver += ["--name", name]
        if intent:
            driver += ["--intent", intent]
        driver += ["--session", session_id, "-w", str(root)]
        if sized:
            driver += ["--reservation", reservation.id]
        record.record(
            {
                "type": "command",
                "kind": "launch",
                "tool": "launch_workflow",
                "command": shlex.join(driver),
                "script": script_path,
                "args": [],
                "cwd": str(project_dir),
                "background": False,
                "sized": sized,
                "resources": reservation.slice,
                "reservation": reservation.id,
            }
        )
        try:
            if sized:
                result = _ops.launch_child(
                    root,
                    script_path,
                    reservation=reservation,
                    name=name,
                    intent=intent,
                    session=session_id,
                    cwd=project_dir,
                    wait=True,
                )
            else:
                result = _ops.launch_script(
                    root,
                    script_path,
                    name=name,
                    intent=intent,
                    session=session_id,
                    capture_output=True,
                    reservation=reservation,
                )
        except (FoundationError, SlabError, OSError):
            # A launch that could not start gives its slice back. Anything
            # else (an interrupt, an exit) propagates without touching the
            # row: a detached child may be running on that slice, and the
            # next reap releases it if nothing is.
            with Workspace(root) as ws:
                ws.runs.release_reservation(reservation.id)
            raise
        _record_run_commands(result.get("run_id"), "launch_workflow")
        return result

    @server.tool()
    @_surfaced
    def wait_for_run(run_id: str | None = None, timeout_s: float = 900.0) -> dict[str, Any]:
        """Block until a run finishes or the timeout passes, then report
        where it stands. run_id takes an id, a unique prefix, or the name of
        a run this session created; without it, waits for every running run
        this session created. 'outcome' is finished (with the run and its
        task tally), process_gone (this call found the run's recorded process
        dead on this host and marked the run failed; nothing to wait for),
        still_running (call again to keep waiting; each entry says whether
        its process is alive here or runs on another host), none_running
        (the session's finished runs), or no_runs."""
        waited = _ops.wait_for_run(
            root, run_id=run_id, session=session_id, timeout_s=min(timeout_s, 1800.0)
        )
        if waited["outcome"] in ("finished", "process_gone"):
            _record_run_commands(waited["run"].id, "wait_for_run")
        elif waited["outcome"] == "none_running":
            for finished in waited["runs"]:
                _record_run_commands(finished.id, "wait_for_run")
        answer: dict[str, Any] = {"outcome": waited["outcome"], "note": waited["note"]}
        if "run" in waited:
            answer["run"] = _ops.run_summary(waited["run"]) | {"progress": waited["progress"]}
        if "runs" in waited:
            answer["runs"] = [_ops.run_summary(r) for r in waited["runs"]]
        if "running" in waited:
            answer["running"] = [
                _ops.run_summary(r) | {"progress": progress, "liveness": liveness}
                for r, progress, liveness in waited["running"]
            ]
        return answer

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
        null off-cluster; jobs submit via submit_job when partitions are
        configured). An 'mp' key names the offline Materials Project snapshot
        when one is configured (search it with search_materials /
        get_material / query_materials). 'budget' is what this server's
        process may use on this host (cpu and gpu counts) and 'free' what
        no live reservation holds right now; size launch_workflow within
        'free'. When the run store cannot be opened, 'free' is null and
        'resources_note' says why; the engines are still listed."""
        from slab.resources import budget as discover_budget

        try:
            with Workspace(root) as ws:
                ws.reap_dead(caller="list_engines")
                resources = ws.free_resources()
        except (FoundationError, sqlite3.Error, OSError) as e:
            found = discover_budget()
            return engines_overview() | {
                "budget": {"cpus": len(found.cpus), "gpus": len(found.gpus)},
                "free": None,
                "resources_note": f"run store unavailable: {e}",
            }
        return engines_overview() | {
            "budget": {key: len(ids) for key, ids in resources["budget"].items()},
            "free": {key: len(ids) for key, ids in resources["free"].items()},
        }

    @server.tool()
    @_surfaced
    def free_resources() -> dict[str, Any]:
        """What is free on this host right now. 'budget' and 'free' list the
        cpu ids and gpu ids, 'reservations' the ids of the live
        reservations that hold the difference, and 'held' one line per
        live reservation: its slice, the run that claimed it or the
        process that holds it, and its age. Call it before a concurrent
        launch; list_engines and the environment state free amounts from
        the moment they were read."""
        with Workspace(root) as ws:
            return _ops.free_resources(ws)

    @server.tool()
    @_surfaced
    def list_tasks() -> list[dict[str, str]]:
        """The traced tasks foundation.tasks exposes to workflow scripts: one
        entry per task with its name, signature, and one-sentence summary.
        Call describe_task for the full docstring."""
        return _ops.task_catalog()

    @server.tool()
    @_surfaced
    def describe_task(name: str) -> dict[str, str]:
        """Full signature and docstring of one foundation.tasks task (e.g.
        'relax'), so an agent consults the harness's own vocabulary instead
        of reading its source."""
        return _ops.describe_task(name)

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

    @server.tool()
    @_surfaced
    def query_materials(sql: str, limit: int = 200) -> dict[str, Any]:
        """One read-only SELECT (or WITH) over the snapshot's metadata.sqlite,
        for queries the search_materials filters cannot express. Tables:
        materials (keyed by material_id), material_elements(material_id,
        element), dataset_info, units (consult it instead of guessing units).
        Rows are capped and the result says when it truncated — put LIMIT in
        the query."""
        from slab.mp import query_materials as mp_query

        return mp_query(sql, limit=limit)

    if hpc.partitions:

        @server.tool()
        @_surfaced
        def submit_job(
            command: str,
            name: str,
            partition: str | None = None,
            time_limit: str | None = None,
            nodes: int | None = None,
            ntasks_per_node: int | None = None,
            cpus_per_task: int | None = None,
            gpus_per_node: int | None = None,
            mem: str | None = None,
        ) -> dict[str, Any]:
            """Submit a command as a SLURM batch job (typically 'slab run
            workflow.py' so the result is still a traced, verified run). The
            job exports this server's session id, so the runs it launches
            join this session; the script is kept under the workspace's
            jobs/ directory. time_limit is HH:MM:SS. Size the job with
            ntasks_per_node (required to size), cpus_per_task, gpus_per_node,
            nodes, and mem (e.g. 240G): the size replaces the partition's
            own directives and must fit the node that list_engines reports
            under hpc.partitions.<name>.node; without a size the partition's
            directives apply as declared."""
            from slab.resources import job_size

            _positive("nodes", nodes)
            _positive("ntasks_per_node", ntasks_per_node)
            _positive("cpus_per_task", cpus_per_task)
            _positive("gpus_per_node", gpus_per_node, minimum=0)
            size = job_size(
                nodes=nodes,
                ntasks_per_node=ntasks_per_node,
                cpus_per_task=cpus_per_task,
                gpus_per_node=gpus_per_node,
                mem=mem,
            )
            job = _ops.submit_job(
                root,
                hpc=hpc,
                command=command,
                name=name,
                partition=partition,
                time_limit=time_limit,
                session=session_id,
                project=project_dir,
                size=size,
            )
            record.record(
                {
                    "type": "command",
                    "kind": "job",
                    "tool": "submit_job",
                    "command": command,
                    "job_id": str(job["job_id"]),
                    "job_name": str(job["job_name"]),
                    "partition": str(job["partition"]),
                    "script": str(job["script_path"]),
                    "size": job["size"],
                }
            )
            return job

        @server.tool()
        @_surfaced
        def job_status(job_id: str) -> dict[str, Any]:
            """State of one SLURM job (pending/running/completed/failed/...)."""
            return _ops.job_status(job_id)

        @server.tool()
        @_surfaced
        def cancel_job(job_id: str) -> dict[str, Any]:
            """Cancel a SLURM job (a no-op if it already finished)."""
            return _ops.cancel_job(job_id)

    @server.tool()
    @_surfaced
    def notebook(entry: str | None = None, heading: str | None = None) -> dict[str, Any]:
        """The project's lab notebook (NOTEBOOK.md). With entry, append a
        dated entry — decisions, results with run ids, failures and their
        diagnosis — written for a colleague who has read none of this
        conversation. Without entry, return the latest entries."""
        if entry is not None:
            path = project_files.notebook_append(project_dir, entry, heading=heading)
            return {"path": str(path), "recorded": True}
        return {
            "path": str(project_files.notebook_path(project_dir)),
            "text": project_files.notebook_tail(project_dir),
        }

    @server.tool()
    @_surfaced
    def plan(content: str | None = None) -> dict[str, Any]:
        """The project's living plan (PLAN.md): goal, steps with status, open
        questions. With content, rewrite it whole; without, return the
        current plan. Keep it current — it is what the next session reads."""
        if content is not None:
            path = project_files.plan_write(project_dir, content)
            return {"path": str(path), "text": project_files.plan_read(project_dir)}
        return {
            "path": str(project_files.plan_path(project_dir)),
            "text": project_files.plan_read(project_dir),
        }

    @server.tool()
    @_surfaced
    def list_memories() -> list[dict[str, Any]]:
        """The machine's memories: what earlier sessions on this machine
        recorded about its software (see recall/remember). One entry per
        memory with its name, one-line description, and whether the software
        it was stamped against has changed since."""
        known = memory_store.discover()
        live = software_versions() if any(m.against for m in known.values()) else {}
        return [
            {
                "name": m.name,
                "description": m.description,
                "changed_since": m.drift(live) if m.against else [],
            }
            for _, m in sorted(known.items())
        ]

    @server.tool()
    @_surfaced
    def recall(name: str) -> dict[str, Any]:
        """Read one machine memory in full by name: the fact, who recorded it
        and when, and which of the software it names has changed since it was
        written (confirm the fact before building on it in that case)."""
        known = memory_store.discover()
        found = known.get(name)
        if found is None:
            names = ", ".join(sorted(known)) or "none recorded yet"
            raise MemoryStoreError(f"no memory named {name!r}; memories on this machine: {names}")
        return {
            "name": found.name,
            "description": found.description,
            "body": found.body().rstrip(),
            "provenance": found.provenance(),
            "changed_since": found.drift(software_versions()) if found.against else [],
        }

    @server.tool()
    @_surfaced
    def remember(name: str, description: str, body: str) -> dict[str, Any]:
        """Record one confirmed fact about this machine or its software so
        later sessions start knowing it: a package that behaves unlike its
        documentation, a flag that matters, a workaround. name is
        lowercase-with-hyphens; description is the one line a future session
        reads to decide whether the fact applies. Re-using a name replaces
        that memory. Not for results (they belong to runs), project decisions
        (the notebook), or credentials (nowhere)."""
        written = memory_store.write(
            name,
            description,
            body,
            agent="mcp",
            against=memory_store.stamp(f"{description}\n{body}", software_versions()),
        )
        return {"name": written.name, "path": str(written.path), "against": dict(written.against)}

    @server.tool()
    @_surfaced
    def list_skills() -> list[dict[str, str]]:
        """The skill catalog: procedure packages in the Agent Skills format,
        built in, in ~/.config/slab/skills, and in the project's skills/
        directory. One entry per skill with its name, description, source
        layer, and content digest. Load one with skill(name) before doing a
        task it covers."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "source": s.source,
                "digest": s.digest,
                "root": str(s.root),
            }
            for _, s in sorted(discover_skills(project_dir).items())
        ]

    @server.tool()
    @_surfaced
    def skill(name: str) -> dict[str, Any]:
        """Load a skill by name: its full instructions, its root path, and
        its bundled files (scripts, references, assets) as paths relative to
        the root. Prefer a skill's bundled scripts over writing your own; run
        them with your own shell. The load is recorded in this session."""
        catalog = discover_skills(project_dir)
        found = catalog.get(name)
        if found is None:
            known = ", ".join(sorted(catalog))
            raise SkillError(f"no skill named {name!r}; available skills: {known}")
        record.record(
            {"type": "skill", "name": name, "source": found.source, "digest": found.digest}
        )
        return {
            "name": found.name,
            "source": found.source,
            "digest": found.digest,
            "root": str(found.root),
            "body": found.body(),
            "files": bundled_files(found),
        }

    @server.tool()
    @_surfaced
    def report_results(
        results: dict[str, Any], run_ids: list[str], summary: str = ""
    ) -> dict[str, Any]:
        """Record this session's answer in the workspace: results as
        {name: {value, unit}} and the run ids that produced them (full ids or
        unique prefixes; every one must exist). A benchmark scores the session
        from this record, so report exactly the digits a run produced. The
        session stays open; report again to replace the answer."""
        clean, cited = validate_results(results, run_ids)
        with Workspace(root) as ws:
            resolved = [ws.runs.resolve(run_id) for run_id in cited]
        path = record.record(
            {"type": "results", "results": clean, "run_ids": resolved, "summary": summary}
        )
        return {"session": session_id, "recorded": str(path), "results": clean, "run_ids": resolved}

    @server.tool()
    @_surfaced
    def retire_session(
        run_ids: list[str], uncited: str | None = None, dry_run: bool = False
    ) -> dict[str, Any]:
        """Retire this session, last of all: promote the verified runs the
        answer rests on (run_ids, full ids or unique prefixes, anchors from
        earlier sessions included) and expire this session's other runs.
        A cited run that is not verified is reported and left alone, never
        forced. uncited is keep, expire (the default from the retention
        policy), or purge, which also deletes the expired rows and their
        unshared bytes. This cannot be undone: report_results first, then
        retire. dry_run=True only reports."""
        policy = _ops.load_policy(root)
        mode = uncited if uncited is not None else policy.finish.uncited
        if mode not in _ops.RETIRE_MODES:
            raise ValueError(f"uncited must be keep, expire, or purge, not {mode!r}")
        with Workspace(root) as ws:
            report = _ops.retire_session(
                ws,
                session_id,
                keep=run_ids if policy.finish.promote_cited else [],
                mode=mode,
                actor="agent",
                dry_run=dry_run,
            )
        if not dry_run:
            record.record({"type": "retire", **report})
        return report

    return server


def serve(root: Path, *, project: Path | None = None) -> None:  # pragma: no cover - blocks on stdio
    """Run the MCP server on stdio until the client disconnects."""
    build_server(root, project=project).run()
