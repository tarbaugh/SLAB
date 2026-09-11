"""Mason's tools: few, orthogonal, with crisp machine-checkable failure modes.

The tool set follows the SWE-agent lesson that agent performance is driven
by interface design, not tool count: six file/shell primitives whose error
messages teach recovery, plus the tools that make Mason a *research* agent —
SLAB runs (list/show/launch), the SLURM plumbing when this machine has
partitions configured, and the memory instruments: the notebook and plan
for this project, recall and remember for what this machine has taught
earlier sessions.

Contracts the primitives enforce in code, not prompt text:

* ``edit_file`` is exact-string replacement — the old text must match the
  file exactly once (or ``replace_all``), and the file must have been read
  this session first (the staleness guard).
* ``read_file`` numbers lines (numbers ground later edits) and refuses
  binary content.
* every tool failure is returned as the tool *result*, never raised — the
  loop continues and the model sees the evidence.
* mutating tools pass through the session's approval gate; read-only tools
  never ask.
* large outputs are middle-truncated with an explicit marker (the head
  usually holds the command echo, the tail the verdict).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import shlex
import sqlite3
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from foundation.runtime import Workspace
    from mason.roster import AgentSpec

from foundation import _ops
from foundation import memory as memory_store
from foundation.errors import FoundationError, MemoryStoreError
from foundation.models import Reservation
from foundation.project import plan_write
from foundation.runtime import describe_liveness
from mason.client import ToolCall
from mason.mechanisms import enabled
from mason.session import MasonSession
from mason.skills import Skill, discover_skills, listing, visible_catalog
from slab.errors import SlabError

#: Every tool name any session can build. Agent cards validate their
#: ``tools:`` allowlists against this set, so a typo is refused even when the
#: tool it misspells is absent from the current session (no partitions, no
#: skills). Keep it in step with the builders below; a test enforces that.
TOOL_VOCABULARY = frozenset(
    {
        "read_file",
        "write_file",
        "edit_file",
        "list_dir",
        "search",
        "shell",
        "list_runs",
        "show_run",
        "read_artifact",
        "launch_workflow",
        "wait_for_run",
        "list_engines",
        "list_tasks",
        "describe_task",
        "search_materials",
        "get_material",
        "query_materials",
        "submit_job",
        "job_status",
        "cancel_job",
        "notebook",
        "plan",
        "skill",
        "recall",
        "remember",
        "delegate",
        "review",
        "finish",
    }
)

#: The tools that only look: shell, files, listings, lookups. A step made of
#: these alone changes nothing the workspace records: no run, no plan, no
#: note, no brief, no report. The loop counts consecutive such steps and
#: says so to the model; the science review flags a long run of them.
LOOKING_TOOLS = frozenset(
    {
        "shell",
        "read_file",
        "read_artifact",
        "list_dir",
        "search",
        "list_runs",
        "show_run",
        "job_status",
        "wait_for_run",
        "describe_task",
        "list_tasks",
        "list_engines",
        "get_material",
        "search_materials",
        "query_materials",
        "recall",
        "skill",
    }
)

#: Lookups whose answer holds for the session: what the machine has, what a
#: task takes, what a material is. The session keeps the newest answer to
#: each, and the review tool hands them to the critic, so it spends its
#: steps on the observable and the contract instead of re-gathering the
#: fingerprint (two real critic passes re-ran all of these).
FACT_TOOLS = frozenset({"list_engines", "list_tasks", "describe_task", "get_material"})
#: One fact in the critic's brief, and all of them together.
_FACT_CHARS = 2_000
_FACTS_CHARS = 8_000

#: The tools that observe and never act: no file write, no launch, no shell,
#: no entry in the notebook, the plan, or machine memory. A card that
#: ``reviews`` gets exactly this slice of the session, whatever its
#: allowlist says, so a critic cannot fix the plan it was asked to judge.
READ_ONLY_TOOLS = frozenset(
    {
        "read_file",
        "list_dir",
        "search",
        "list_runs",
        "show_run",
        "read_artifact",
        "list_engines",
        "list_tasks",
        "describe_task",
        "search_materials",
        "get_material",
        "query_materials",
        "job_status",
        "skill",
        "recall",
        "finish",
    }
)

#: What a ``review_first`` card may not call before a critic approves the
#: plan: the three tools that spend compute.
_GATED_UNTIL_REVIEWED = ("launch_workflow", "submit_job", "delegate")

_MAX_READ_LINES = 400
_MAX_LINE_CHARS = 500
_MAX_DIR_ENTRIES = 200
_MAX_SEARCH_MATCHES = 100
_MAX_SHELL_TIMEOUT_S = 600.0
# wait_for_run: one blocking call replaces a chain of sleep-and-poll shell
# commands. The cap bounds a single tool call, not the wait — the tool says
# to call again, and each re-issue costs one step instead of six.
_MAX_WAIT_TIMEOUT_S = 1800.0
_WAIT_POLL_S = 5.0
_WAIT_GRACE_S = 10.0

Handler = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class Tool:
    """One tool: its schema for the model, its handler, its permission class."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Handler
    requires_approval: bool = False
    # A dynamic gate (e.g. shell allowlists) overrides requires_approval per call.
    gate: Callable[[dict[str, Any]], bool] | None = None

    def needs_approval(self, arguments: dict[str, Any]) -> bool:
        if self.gate is not None:
            return self.gate(arguments)
        return self.requires_approval


@dataclass
class Toolbox:
    """The session's tools, in a stable order (stable prompts cache well)."""

    session: MasonSession
    tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def specs(self) -> list[dict[str, Any]]:
        """OpenAI-style tool declarations for the request body."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self.tools.values()
        ]

    def catalog_text(self) -> str:
        """A plain-text catalog for the fenced protocol (no tools= support)."""
        lines = []
        for tool in self.tools.values():
            properties = tool.parameters.get("properties", {})
            required = set(tool.parameters.get("required", ()))
            arguments = ", ".join(
                f"{name}{'' if name in required else '?'}: {spec.get('type', 'any')}"
                for name, spec in properties.items()
            )
            lines.append(f"- {tool.name}({arguments}): {tool.description}")
        return "\n".join(lines)

    def dispatch(self, call: ToolCall) -> str:
        """Execute one tool call; the answer is always a string, never a raise."""
        tool = self.tools.get(call.name)
        if tool is None:
            known = ", ".join(self.tools)
            return f"unknown tool {call.name!r}; available tools: {known}"
        if call.arguments_error is not None:
            return f"tool {call.name} not run: {call.arguments_error}"
        missing = [
            key
            for key in tool.parameters.get("required", ())
            if key not in call.arguments
        ]
        if missing:
            properties = tool.parameters.get("properties", {})
            required = list(tool.parameters.get("required", ()))
            optional = [key for key in properties if key not in required]
            return (
                f"tool {call.name} not run: missing required argument(s) "
                f"{', '.join(missing)} (required: {', '.join(required)}"
                + (f"; optional: {', '.join(optional)}" if optional else "")
                + ")"
            )
        preview = self.session.attribution() + _preview(call)
        if tool.needs_approval(call.arguments):
            try:
                approved = self.session.allows(call.name, preview, requires_approval=True)
            except Exception:
                # dispatch never raises: a crashing approver (closed stdin,
                # broken terminal) is a refusal, not a dead process.
                approved = False
            if not approved:
                return (
                    f"tool {call.name} was not approved by the user; explain what you "
                    f"wanted it for, or work another way"
                )
        try:
            result = tool.handler(call.arguments)
        except Exception as e:  # evidence for the model, never a dead loop
            result = f"tool {call.name} failed: {type(e).__name__}: {e}"
        # The output cap is context hygiene's first layer: off, a result
        # reaches the model whole, however long, so the ablation measures
        # the cap together with the clearing it precedes.
        agent = self.session.agent
        shown = (
            _truncate_middle(result, agent.max_tool_output_chars)
            if enabled(agent, "context-hygiene")
            else result
        )
        if call.name in FACT_TOOLS and not _is_refusal(shown):
            self.session.facts[_fact_key(call)] = shown
        return shown


def _fact_key(call: ToolCall) -> str:
    """``describe_task(name=relax_cell)``: the call as the critic will read it.

    Examples:
        >>> _fact_key(ToolCall(id="c", name="list_engines"))
        'list_engines()'
        >>> _fact_key(ToolCall(id="c", name="describe_task", arguments={"name": "relax_cell"}))
        'describe_task(name=relax_cell)'
    """
    inner = ", ".join(f"{key}={value}" for key, value in sorted(call.arguments.items()))
    return f"{call.name}({inner})"


def _is_refusal(result: str) -> bool:
    """A result that answered nothing: a failure, a refusal, an unknown name."""
    head = result.lstrip()[:40].lower()
    return head.startswith(("tool ", "refused", "unknown", "no such", "no task", "no run"))


def _preview(call: ToolCall) -> str:
    """What the human sees before approving — the load-bearing keys first.

    A raw JSON dump truncates arbitrarily: a giant ``content`` can push the
    ``path`` it writes to right out of the preview. Name the keys that
    decide whether to approve.
    """
    arguments = call.arguments
    if "command" in arguments:
        return str(arguments["command"])
    parts = []
    for key in (
        "path",
        "script",
        "name",
        "partition",
        "old_string",
        "new_string",
        "description",
        "content",
        "body",
    ):
        if key in arguments:
            value = str(arguments[key])
            # content and body are the thing being approved — a human shown
            # 120 chars of a workflow script, or of a fact about to be
            # written into every future session's prompt, is approving
            # blind. Head and tail, with the elision announced, is enough to
            # actually read it.
            limit = 1200 if key in ("content", "body") else 120
            shown = value if len(value) <= limit else _truncate_middle(value, limit)
            parts.append(f"{key}={shown!r}")
    return " ".join(parts)[:1600] if parts else json.dumps(arguments)[:200]


def _truncate_middle(text: str, limit: int) -> str:
    """Keep head and tail; announce exactly what was dropped.

    Examples:
        >>> _truncate_middle("x" * 30, 21)
        'xxxxxxxxxxxxxx\\n[... 15 characters truncated ...]\\nx'
        >>> _truncate_middle("short", 21)
        'short'
    """
    if len(text) <= limit:
        return text
    marker = "\n[... {} characters truncated ...]\n"
    head = (limit * 2) // 3
    tail = limit - head - len(marker) + 2  # the {} placeholder frees ~2 chars
    tail = max(tail, 40) if limit > 200 else max(tail, 1)
    dropped = len(text) - head - tail
    return text[:head] + marker.format(dropped) + text[-tail:]


def _schema(properties: dict[str, dict[str, Any]], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


def build_toolbox(
    session: MasonSession,
    spec: AgentSpec | None = None,
    *,
    depth: int = 0,
    skills: dict[str, Skill] | None = None,
    roster: dict[str, AgentSpec] | None = None,
    parent_client: Any | None = None,
) -> Toolbox:
    """Every tool this session gets; SLURM tools only where partitions exist.

    *spec* is the agent card in force: its ``tools`` allowlist narrows the
    box (``finish`` always stays), and ``None`` means no narrowing. *depth*
    is the delegation depth: a delegated agent (depth > 0) loses ``plan``,
    because ``PLAN.md`` belongs to the turn owner, and can never delegate
    onward. *skills* is the full catalog: the ``skill`` tool sees the
    card's slice of it, while ``delegate`` hands the whole catalog down so
    each child re-narrows by its own card; ``None`` discovers the catalog
    from the session's project directory. *roster* and *parent_client*
    feed the ``delegate`` tool, which exists only when the card delegates,
    the depth is zero, ``[agent] delegation`` is on, and the roster holds
    someone to delegate to; the ``review`` tool exists under the same
    switch for a card that delegates or reviews first, when the roster
    holds a critic. A card that ``reviews`` keeps only the read-only tools.
    A card that reviews first has its compute-spending tools refused until
    the plan is approved. Every mechanism the box embodies (the traced-run
    tools, the skill tool, memory, delegation, the critic) is added only
    when its switch is on (:mod:`mason.mechanisms`).
    """
    if skills is None:
        skills = discover_skills(session.cwd)
    visible = (
        visible_catalog(skills, spec.name, spec.skills_scope) if spec is not None else skills
    )
    box = Toolbox(session)
    # The file fence (_out_of_scope): writes stay in the project and the
    # workspace; reads and launches also reach the skill directories the
    # harness itself advertises.
    write_roots = (
        _scope_root(session, session.cwd),
        _scope_root(session, session.workspace_root),
    )
    snapshot_root = _mp_snapshot_root(session)
    read_roots = (
        write_roots
        + tuple(_scope_root(session, s.root) for s in skills.values())
        + _installed_package_roots(session)
        + ((snapshot_root,) if snapshot_root is not None else ())
    )
    _add_file_tools(box, session, read_roots, write_roots)
    _add_shell_tool(box, session)
    if enabled(session.agent, "check-gating"):
        _add_workflow_tools(box, session, read_roots)
    _add_engine_tools(box, session)
    if snapshot_root is not None:
        _add_mp_tools(box, snapshot_root)
    if session.hpc.partitions:
        _add_hpc_tools(box, session)
    _add_memory_tools(box, session)
    if enabled(session.agent, "machine-memory"):
        _add_machine_memory_tools(box, session)
    if visible and enabled(session.agent, "skills"):
        _add_skill_tool(box, session, visible)
    on_team = enabled(session.agent, "delegation")
    if spec is not None and depth == 0 and on_team and roster is not None:
        from mason.roster import critics, hands

        if spec.delegates and hands(spec, roster):
            _add_delegate_tool(box, session, spec, roster, skills, parent_client)
        if (
            (spec.delegates or spec.review_first)
            and critics(roster)
            and enabled(session.agent, "critic-gate")
        ):
            _add_review_tool(box, session, spec, roster, skills, parent_client, read_roots)
    if spec is not None and spec.tools is not None:
        for name in [n for n in box.tools if n not in spec.tools and n != "finish"]:
            del box.tools[name]
    if spec is not None and spec.reviews:
        for name in [n for n in box.tools if n not in READ_ONLY_TOOLS]:
            del box.tools[name]
    if depth > 0:
        box.tools.pop("plan", None)
    box.add(
        Tool(
            name="finish",
            description=(
                "End the current task with a final report. Cite run ids for every "
                "number; list what was verified and what remains open. When the "
                "task names a result key, also pass the quantity in `results` "
                "under that name with its unit, and list the run ids that "
                "produced it in `run_ids` — that is how a campaign is scored, "
                "and it is the keep decision: the verified runs in `run_ids` are "
                "promoted, and this session's other runs are expired, so cite "
                "every run a number rests on, anchors from earlier sessions "
                "included. "
                "Before calling it: a machine fact this session learned the hard "
                "way (a workaround, a missing utility, a device limit) belongs in "
                "`remember` first, or the next session pays for it again. Call "
                "finish alone, as the only tool call of its message, after the "
                "evidence it cites has been read. When you were asked to review, "
                "pass the verdict in `verdict`; a review without one is no verdict."
            ),
            parameters=_schema(
                {
                    "report": {"type": "string"},
                    "verdict": {
                        "type": "string",
                        "enum": ["approve", "revise"],
                        "description": (
                            "for a review only: approve when no finding is blocking, "
                            "revise when one is"
                        ),
                    },
                    "results": {
                        "type": "object",
                        "description": "result name -> {value, unit}",
                        "additionalProperties": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "number"},
                                "unit": {"type": "string"},
                            },
                            "required": ["value", "unit"],
                        },
                    },
                    "run_ids": {"type": "array", "items": {"type": "string"}},
                },
                ["report"],
            ),
            handler=lambda arguments: str(arguments.get("report", "")),
        )
    )
    gate_on = enabled(session.agent, "critic-gate")
    if spec is not None and spec.review_first and depth == 0 and gate_on:
        _gate_until_reviewed(box, session)
    return box


# -- file primitives ---------------------------------------------------------


def _resolve(session: MasonSession, path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else session.cwd / candidate


def _scope_root(session: MasonSession, root: Path) -> Path:
    """A scope root in comparable form: absolute (against the session's cwd,
    not the process's) and symlink-resolved."""
    absolute = root if root.is_absolute() else session.cwd / root
    return absolute.resolve()


def _installed_package_roots(session: MasonSession) -> tuple[Path, ...]:
    """The roots of the four packages in the slab-stack distribution.

    Adds read-only reach into the harness's own source so ``read_file`` can
    answer questions the vocabulary alone cannot — 'does foundation.tasks
    have a cell relaxation task?' is one Read tool call, not thirty shell
    calls poking at sed and grep. In an editable install the roots are the
    ``src/<pkg>/`` directories; in a site-packages install they point there.
    Write scope is unchanged: the fence still refuses edits into the source.
    """
    import importlib.resources

    roots: list[Path] = []
    for name in ("slab", "foundation", "mason", "slab_stack"):
        try:
            root = Path(str(importlib.resources.files(name)))
        except (ModuleNotFoundError, TypeError):  # pragma: no cover - always installed
            continue
        if root.is_dir():
            roots.append(_scope_root(session, root))
    return tuple(roots)


def _mp_snapshot_root(session: MasonSession) -> Path | None:
    """The configured Materials Project snapshot root, or None.

    A configured snapshot gains read-only reach (the agent opens archived
    CIFs with ``read_file`` and hands their paths to scripts) and switches
    on the search tools. A broken slab.toml answers None here — the config
    error surfaces where config is read for real, not from fence assembly.
    """
    from slab.config import config_value
    from slab.errors import SlabError

    try:
        root = config_value("builders.mp.root", session.cwd)
    except SlabError:
        return None
    if root is None:
        return None
    return _scope_root(session, Path(str(root)))


#: Variables that hold model credentials whatever the config names: a
#: subprocess the model drives must never see them, or `env` in a shell
#: call puts the key into the tool result, the context, and the transcript.
_CREDENTIAL_VARS = frozenset({"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "PORTKEY_API_KEY"})


def _subprocess_env(session: MasonSession) -> dict[str, str]:
    """The process environment minus the model's credentials.

    Examples:
        >>> import os, types
        >>> os.environ["SLAB_DOCTEST_SECRET"] = "sk-x"
        >>> fake = types.SimpleNamespace(agent=types.SimpleNamespace(
        ...     resolved_api_key_env="SLAB_DOCTEST_SECRET"))
        >>> "SLAB_DOCTEST_SECRET" in _subprocess_env(fake)
        False
        >>> del os.environ["SLAB_DOCTEST_SECRET"]
    """
    hidden = set(_CREDENTIAL_VARS)
    named = getattr(session.agent, "resolved_api_key_env", None)
    if named:
        hidden.add(str(named))
    return {key: value for key, value in os.environ.items() if key not in hidden}


class RunStoreUnavailable(Exception):
    """The workspace's run store could not be opened, and what to do about it.

    Raised in place of the bare ``StorageError`` or ``sqlite3`` error so the
    text the model reads carries the recovery. A database another process
    holds, or a filesystem refusing locks, is a fault of the workspace and
    not of the work: a real session read a bare "database is locked", spent
    an hour on byte-level forensics, and deleted the store's write-ahead
    log by hand.
    """

    def __init__(self, root: Path, cause: BaseException) -> None:
        super().__init__(
            f"the run store at {root} could not be opened ({cause}). This is a fault "
            f"of the workspace, not of your work: another process may hold the "
            f"database, or the filesystem may be refusing locks. Wait about a "
            f"minute and retry this call once. If it fails again, record the fault "
            f"in the notebook and finish with a report naming it. Do not inspect, "
            f"modify, or delete files under {root}."
        )


def _open_workspace(session: MasonSession) -> Workspace:
    """Open the session's workspace, or raise :class:`RunStoreUnavailable`."""
    from foundation.errors import FoundationError
    from foundation.runtime import Workspace

    try:
        return Workspace(session.workspace_root)
    except (FoundationError, sqlite3.Error, OSError) as e:
        # OSError: a root that cannot be created or opened (a read-only
        # filesystem, a permission refused) is the same fault to the model.
        raise RunStoreUnavailable(Path(session.workspace_root), e) from e


_STORE_MUTATION = re.compile(r"(?:^|[\s;|&(])(?:rm|rmdir|mv|truncate|shred|unlink|dd)\s")


def _store_mutation(command: str, workspace_root: Path) -> str | None:
    """The refusal when a shell command would rewrite the run store by hand.

    The store's database and its sidecar files (``runs.db`` and its
    ``-wal``, ``-shm``, and ``-journal`` companions) belong to SQLite: a
    session that cannot open them reports the fault, it does not repair
    them. The guard is a name match, not a parser: a deleting verb in a
    command that names the database or the workspace root. A copy made to
    inspect elsewhere passes.

    Examples:
        >>> root = Path("/ws")
        >>> _store_mutation("rm -f runs.db-wal runs.db-shm", root) is None
        False
        >>> _store_mutation("cd /ws && rm -rf cas/ab", root) is None
        False
        >>> _store_mutation("cp /ws/runs.db /tmp/copy.db", root) is None
        True
        >>> _store_mutation("rm -f build/*.o", root) is None
        True
    """
    if not _STORE_MUTATION.search(command):
        return None
    root = str(workspace_root)
    if "runs.db" not in command and root not in command:
        return None
    return (
        f"refused: this command would delete or move files of the run store under "
        f"{root}. The workspace is SLAB's record of every run, and its database is "
        f"never repaired by hand. If the store cannot be opened, wait about a "
        f"minute, retry the tool once, then report the fault with finish."
    )


def _state_text(run: Any) -> str:
    """The lifecycle state as a reader should take it.

    Every run is born quarantined, so a running run that reads
    ``quarantined`` is in its initial state, not in trouble. A real
    critic took the pair for an anomaly; the word now says so.

    Examples:
        >>> from foundation.models import Run
        >>> _state_text(Run(status="running"))
        'quarantined (initial state)'
        >>> _state_text(Run(status="completed"))
        'quarantined'
    """
    if run.state.value == "quarantined" and run.status.value == "running":
        return "quarantined (initial state)"
    return str(run.state.value)


def _rendered_run(summary: dict[str, Any], run: Any) -> dict[str, Any]:
    """The run half of ``show_run`` with its state worded and its liveness named."""
    rendered = dict(summary)
    rendered["state"] = _state_text(run)
    if run.status.value == "running":
        rendered["liveness"] = describe_liveness(run)
    return rendered


def _compact_details(details: dict[str, Any]) -> dict[str, Any]:
    """The run record with its finished tasks folded to one line each.

    A full record carries every task's recipe, inputs, and outputs. A
    labeling run of 88 tasks made that 190,000 characters, and a real
    session polled it six times, spending its context on setup lines.
    Checks, the run's failure record, and every field of a failed task
    stay verbatim, because they are what a correction is computed from.

    Examples:
        >>> details = {"run": {"id": "r"}, "checks": [], "tasks": [
        ...     {"seq": 1, "name": "single_point", "status": "completed",
        ...      "cache_hit": True, "duration_s": 0.0, "error": None, "failure": None,
        ...      "recipe": {"params": {"label": "rattle_T300_0"},
        ...                 "extra": {"setup": ["..."] * 32}},
        ...      "inputs": {"atoms": "sha256:..."}, "outputs": {}},
        ...     {"seq": 2, "name": "single_point", "status": "failed",
        ...      "cache_hit": False, "duration_s": 1.5, "error": "boom",
        ...      "failure": {"message": "boom"}, "recipe": {}, "inputs": {},
        ...      "outputs": {}}], "artifacts": [], "history": []}
        >>> compact = _compact_details(details)
        >>> compact["tasks_summary"]
        '1 completed, 1 failed (1 cache hit)'
        >>> sorted(compact["tasks"][0])
        ['cache_hit', 'duration_s', 'label', 'name', 'seq', 'status']
        >>> compact["tasks"][1]["failure"]
        {'message': 'boom'}
        >>> list(compact)
        ['run', 'checks', 'tasks_summary', 'tasks', 'artifacts', 'history', 'note']
    """
    tasks = list(details.get("tasks") or [])
    folded: list[dict[str, Any]] = []
    for task in tasks:
        if task.get("status") == "failed" or task.get("error") or task.get("failure"):
            folded.append(task)
            continue
        line = {key: task.get(key) for key in ("seq", "name", "status", "cache_hit", "duration_s")}
        recipe = task.get("recipe")
        params = recipe.get("params") if isinstance(recipe, dict) else None
        label = params.get("label") if isinstance(params, dict) else None
        if label:
            line["label"] = label
        folded.append(line)
    summary = _ops.tally_line(
        [str(task.get("status")) for task in tasks],
        sum(1 for task in tasks if task.get("cache_hit")),
    )
    compact: dict[str, Any] = {}
    for key, value in details.items():
        if key == "tasks":
            compact["tasks_summary"] = summary
            compact["tasks"] = folded
        else:
            compact[key] = value
    compact["note"] = (
        "finished tasks are folded to one line each; call show_run with task=<label "
        "or seq> for one task's recipe, inputs, and outputs, or full=true for all"
    )
    return compact


def _without_failure_records(details: dict[str, Any]) -> dict[str, Any]:
    """*details* with every structured failure record removed.

    The ``failure-records`` switch off: a failed run and a failed task keep
    their status and one-line error, and lose the trimmed traceback and
    diagnostic notes the record carries.

    Examples:
        >>> stripped = _without_failure_records({"run": {"id": "r", "failure": {"m": 1}},
        ...     "tasks": [{"seq": 1, "error": "boom", "failure": {"message": "boom"}}]})
        >>> stripped["run"], stripped["tasks"]
        ({'id': 'r'}, [{'seq': 1, 'error': 'boom'}])
    """
    stripped = dict(details)
    run = stripped.get("run")
    if isinstance(run, dict):
        stripped["run"] = {k: v for k, v in run.items() if k != "failure"}
    tasks = stripped.get("tasks")
    if isinstance(tasks, list):
        stripped["tasks"] = [
            {k: v for k, v in task.items() if k != "failure"} if isinstance(task, dict) else task
            for task in tasks
        ]
    return stripped


def _one_task(details: dict[str, Any], wanted: object) -> dict[str, Any]:
    """The run line, its checks, and one task in full, or the choice on offer.

    A real session read a 12,000-character full record, saw it cut at the
    cap, and went digging in the artifact store by hand for the one output
    it wanted. One task at a time fits.

    Examples:
        >>> details = {"run": {"id": "r"}, "checks": [], "tasks": [
        ...     {"seq": 1, "name": "relax_cell", "recipe": {"params": {"label": "w-ref"}},
        ...      "outputs": {"a0": 3.17}},
        ...     {"seq": 2, "name": "single_point", "recipe": {}, "outputs": {}}]}
        >>> _one_task(details, "w-ref")["task"]["outputs"]
        {'a0': 3.17}
        >>> _one_task(details, 2)["task"]["name"]
        'single_point'
        >>> _one_task(details, "nope")["error"]
        "no task 'nope'; the tasks: 1 relax_cell (w-ref), 2 single_point"
    """
    tasks = list(details.get("tasks") or [])

    def label(task: dict[str, Any]) -> str | None:
        recipe = task.get("recipe")
        params = recipe.get("params") if isinstance(recipe, dict) else None
        value = params.get("label") if isinstance(params, dict) else None
        return str(value) if value else None

    key = str(wanted)
    for task in tasks:
        if str(task.get("seq")) == key or label(task) == key:
            return {"run": details.get("run"), "checks": details.get("checks"), "task": task}
    for task in tasks:
        if task.get("name") == key:
            return {"run": details.get("run"), "checks": details.get("checks"), "task": task}
    offer = ", ".join(
        f"{t.get('seq')} {t.get('name')}" + (f" ({label(t)})" if label(t) else "") for t in tasks
    )
    return {"run": details.get("run"), "error": f"no task {wanted!r}; the tasks: {offer}"}


def _out_of_scope(session: MasonSession, path: Path, roots: tuple[Path, ...]) -> str | None:
    """The refusal observation when *path* leaves the file fence, else None.

    The fence is the sandbox principle at the tool layer: the agent reads and
    runs within its project, its workspace, and the skills it was shown, and
    writes only within the project and workspace. One carve-out inside the
    workspace: the sessions directory (transcripts and compaction files) is
    refused, because past sessions are not context. Comparison happens on
    fully resolved paths, so a symlink pointing out of the fence counts as
    outside it. This is a workflow control, not a security boundary — the
    shell tool remains the honest escape, behind its own gate.
    """
    if session.agent.file_scope == "anywhere":
        return None
    resolved = path.expanduser().resolve()
    # The sessions directory sits inside the workspace but is not context.
    # A past transcript is a losing substitute for durable state: it is
    # huge, it records what *seemed* true mid-investigation, and it may
    # describe a different campaign entirely. Everything a past session
    # kept on purpose arrives through the project files and the memories.
    # Resolved against resolved: a relative workspace root (the default
    # .slab) or a symlinked one would otherwise never match, and the fence
    # would silently open.
    if resolved.is_relative_to(session.sessions_dir.expanduser().resolve()):
        return (
            f"refused: {path} is a session transcript, and past sessions are "
            f"not context. What earlier sessions kept for you arrives three "
            f"ways: the goal text, the project files (BRIEF/PLAN/notebook), "
            f"and machine memories — call `recall`. A fact worth carrying "
            f"between sessions belongs in `remember`, not in a transcript."
        )
    for root in roots:
        if resolved.is_relative_to(root):
            return None
    listed = "\n".join(f"  {root}" for root in roots)
    return (
        f"refused: {path} is outside this session's file scope. This "
        f"operation works within these roots — retry with a path under one "
        f"of them:\n{listed}\n"
        f"Otherwise use the shell tool (approval-gated), or set "
        f"[agent] file_scope = \"anywhere\"."
    )


def _python_syntax_note(path: Path, content: str) -> str:
    """Post-write verification for Python files: a syntax check, immediately.

    Open models sometimes write ``\\n`` as literal text or truncate a file;
    running the broken script later wastes a whole round trip. ``compile``
    parses without executing.
    """
    if path.suffix != ".py":
        return ""
    try:
        compile(content, str(path), "exec")
    except SyntaxError as e:
        return (
            f"\nWARNING: the file does not parse as Python (line {e.lineno}: {e.msg}); "
            f"read it back and fix it before running"
        )
    return ""


def _add_file_tools(
    box: Toolbox,
    session: MasonSession,
    read_roots: tuple[Path, ...],
    write_roots: tuple[Path, ...],
) -> None:
    def read_file(arguments: dict[str, Any]) -> str:
        path = _resolve(session, str(arguments["path"]))
        if denied := _out_of_scope(session, path, read_roots):
            return denied
        offset = int(arguments.get("offset", 1))
        limit = int(arguments.get("limit", _MAX_READ_LINES))
        if not path.is_file():
            return f"no such file: {path}"
        raw = path.read_bytes()
        if b"\x00" in raw[:8192]:
            return f"{path} looks binary ({len(raw)} bytes); read_file only reads text"
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        if digested := _digest_unless_raw(arguments, path.name, text, len(lines)):
            session.read_files.add(path)
            return digested
        window = lines[max(offset - 1, 0) : max(offset - 1, 0) + limit]
        numbered = []
        for i, line in enumerate(window, start=max(offset, 1)):
            if len(line) > _MAX_LINE_CHARS:
                line = line[:_MAX_LINE_CHARS] + " [line truncated]"
            numbered.append(f"{i:6d}\t{line}")
        session.read_files.add(path)
        shown = "\n".join(numbered) if numbered else "(no lines in this window)"
        note = "" if len(lines) <= len(window) + offset - 1 else (
            f"\n[file has {len(lines)} lines; showing {max(offset,1)}"
            f"-{max(offset,1) + len(window) - 1}]"
        )
        return shown + note

    box.add(
        Tool(
            name="read_file",
            description=(
                "Read a text file with line numbers. Use offset/limit to window "
                "large files. You must read a file before you may edit it. A "
                "recognised engine output (a pw.x .pwo, a LAMMPS log, an extended "
                "XYZ file) comes back as a digest first: system, convergence trace, "
                "final numbers, warnings, and whether the job finished. Pass "
                "raw=true, or offset/limit, to read the text itself."
            ),
            parameters=_schema(
                {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "description": "1-based first line"},
                    "limit": {"type": "integer"},
                    "raw": {
                        "type": "boolean",
                        "description": "the text itself, not the digest of an engine output",
                    },
                },
                ["path"],
            ),
            handler=read_file,
        )
    )

    def write_file(arguments: dict[str, Any]) -> str:
        path = _resolve(session, str(arguments["path"]))
        if denied := _out_of_scope(session, path, write_roots):
            return denied
        content = str(arguments["content"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        session.read_files.add(path)
        return f"wrote {len(content)} characters to {path}" + _python_syntax_note(path, content)

    box.add(
        Tool(
            name="write_file",
            description="Create or overwrite one text file with the given content.",
            parameters=_schema(
                {"path": {"type": "string"}, "content": {"type": "string"}},
                ["path", "content"],
            ),
            handler=write_file,
            requires_approval=True,
        )
    )

    def edit_file(arguments: dict[str, Any]) -> str:
        path = _resolve(session, str(arguments["path"]))
        if denied := _out_of_scope(session, path, write_roots):
            return denied
        old = str(arguments["old_string"])
        new = str(arguments["new_string"])
        replace_all = bool(arguments.get("replace_all", False))
        if not path.is_file():
            return f"no such file: {path}"
        if path not in session.read_files:
            return f"read {path} with read_file before editing it (staleness guard)"
        if old == new:
            return "old_string and new_string are identical; nothing to do"
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            return (
                "no exact match for old_string — re-read the file; whitespace and "
                "line numbers from read_file output are not part of the text"
            )
        if count > 1 and not replace_all:
            return (
                f"old_string matches {count} places; extend it until it is unique, "
                f"or set replace_all: true"
            )
        updated = text.replace(old, new)
        path.write_text(updated, encoding="utf-8")
        return f"replaced {count if replace_all else 1} occurrence(s) in {path}" + (
            _python_syntax_note(path, updated)
        )

    box.add(
        Tool(
            name="edit_file",
            description=(
                "Exact-string replacement in a text file: old_string must match the "
                "current content exactly once (or set replace_all)."
            ),
            parameters=_schema(
                {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                ["path", "old_string", "new_string"],
            ),
            handler=edit_file,
            requires_approval=True,
        )
    )

    def list_dir(arguments: dict[str, Any]) -> str:
        path = _resolve(session, str(arguments.get("path", ".")))
        if denied := _out_of_scope(session, path, read_roots):
            return denied
        if not path.is_dir():
            return f"no such directory: {path}"
        entries = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        lines = []
        for entry in entries[:_MAX_DIR_ENTRIES]:
            try:
                if entry.is_dir():
                    lines.append(f"{entry.name}/")
                else:
                    lines.append(f"{entry.name}  ({entry.stat().st_size} B)")
            except OSError:  # dangling symlink: report it, don't fail the listing
                lines.append(f"{entry.name}  (unreadable: broken link?)")
        if len(entries) > _MAX_DIR_ENTRIES:
            lines.append(f"[... {len(entries) - _MAX_DIR_ENTRIES} more entries]")
        return "\n".join(lines) or "(empty directory)"

    box.add(
        Tool(
            name="list_dir",
            description="List one directory (directories end with '/').",
            parameters=_schema({"path": {"type": "string"}}, []),
            handler=list_dir,
        )
    )

    def search(arguments: dict[str, Any]) -> str:
        pattern = str(arguments["pattern"])
        root = _resolve(session, str(arguments.get("path", ".")))
        if denied := _out_of_scope(session, root, read_roots):
            return denied
        glob = str(arguments.get("glob", "*"))
        try:
            expression = re.compile(pattern)
        except re.error as e:
            return f"bad regex {pattern!r}: {e}"
        if not root.is_dir():
            return f"no such directory: {root}"
        matches: list[str] = []
        for candidate in sorted(root.rglob(glob)):
            if not candidate.is_file():
                continue
            # Hidden-dir filtering must look below *root* only: the project may
            # itself live under a dotted parent (~/.research/proj) and still
            # deserves search results.
            relative_parts = candidate.relative_to(root).parts
            if any(part.startswith(".") or part == "__pycache__" for part in relative_parts):
                continue
            # A symlink inside the project can point anywhere; the fence
            # that guards read_file guards what search prints too.
            if _out_of_scope(session, candidate, read_roots):
                continue
            try:
                text = candidate.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if expression.search(line):
                    shown = line.strip()[:200]
                    matches.append(f"{candidate.relative_to(root)}:{number}: {shown}")
                    if len(matches) >= _MAX_SEARCH_MATCHES:
                        matches.append("[... more matches exist; narrow the pattern]")
                        return "\n".join(matches)
        return "\n".join(matches) or f"no matches for {pattern!r}"

    box.add(
        Tool(
            name="search",
            description="Regex search across text files under a directory (recursive).",
            parameters=_schema(
                {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string", "description": "filename filter, e.g. *.py"},
                },
                ["pattern"],
            ),
            handler=search,
        )
    )


# A parallel launch spelled out in a command or script: mpirun/mpiexec/srun
# with an explicit rank count. The configured engines size their own
# launches; this catches the hand-written ones.
_RANK_FLAG = re.compile(r"\b(?:mpirun|mpiexec|srun)\b[^\n;|&]*?(?:-np|--ntasks|-n)[=\s]+(\d+)")
# A '#' at the start of a line or after whitespace opens a comment in
# Python and in shell; what follows it on the line is not a command.
_LINE_COMMENT = re.compile(r"(?m)(?:^|(?<=\s))#.*$")


def _rank_overcommit(text: str, *, limit: int | None = None) -> str | None:
    """A refusal when *text* asks for more MPI ranks than *limit* cpus.

    Guards the two surfaces that execute HERE (the shell and
    launch_workflow); submit_job is deliberately exempt, because its
    payload runs in its own allocation with its own budget. The shell
    compares against the session's whole cpu budget (the default); a
    launch compares against its own slice, because that slice is all the
    run may use whatever the budget holds. Comments are stripped before
    the scan, so a note about an mpirun is not an mpirun. The refusal is
    a tool result the model reads and adapts to, never an exception.

    Examples:
        >>> _rank_overcommit("# note: mpirun -np 64 was too wide", limit=2) is None
        True
        >>> _rank_overcommit("mpirun -np 64 pw.x  # too wide", limit=2) is not None
        True
    """
    from slab.hpc import cpu_budget

    requested = max(
        (int(m.group(1)) for m in _RANK_FLAG.finditer(_LINE_COMMENT.sub("", text))),
        default=0,
    )
    budget = cpu_budget() if limit is None else limit
    if requested > budget:
        where = "usable in this session" if limit is None else "in this launch's slice"
        return (
            f"refused: this launches {requested} MPI rank(s) but only {budget} "
            f"cpu(s) are {where}. Size the launch within that "
            f"budget — and prefer the configured engine (engine='qe' with "
            f"calculator_options) over a hand-written mpirun: it already "
            f"launches at the right width."
        )
    return None


# The run driver spelled out in a shell command. A run the shell starts
# holds no reservation, so the store could not account for it, and every
# other launch on this host would size itself against a free count that
# omits it. The rule matches the existing one: physics goes through
# launch_workflow.
_DRIVER_INVOCATION = re.compile(
    r"(?:\b(?:slab|foundation)\s+run\b|\b(?:foundation|slab_stack)\.cli\s+run\b)"
)


def _driver_in_shell(command: str) -> str | None:
    """A refusal when a shell *command* invokes the run driver by hand.

    Examples:
        >>> _driver_in_shell("slab run wf.py") is not None
        True
        >>> _driver_in_shell("python -m foundation.cli run wf.py") is not None
        True
        >>> _driver_in_shell("python -m slab_stack.cli run wf.py") is not None
        True
        >>> _driver_in_shell("slab list") is None
        True
    """
    if _DRIVER_INVOCATION.search(command) is None:
        return None
    return (
        "refused: 'slab run' from the shell starts a run that holds no reservation, "
        "so the store cannot account for its cpus and gpus, and every other launch "
        "on this host would size itself against a free count that omits it. Use "
        "launch_workflow (with ntasks, threads, and gpus to size it; background=true "
        "for a long run) instead."
    )


# -- shell -------------------------------------------------------------------


def _add_shell_tool(box: Toolbox, session: MasonSession) -> None:
    def shell(arguments: dict[str, Any]) -> str:
        command = str(arguments["command"])
        if refused := _rank_overcommit(command):
            return refused
        if refused := _driver_in_shell(command):
            return refused
        if refused := _store_mutation(command, Path(session.workspace_root)):
            return refused
        record_command(
            session, kind="shell", tool="shell", command=command, cwd=str(session.cwd)
        )
        timeout = min(
            float(arguments.get("timeout_s", session.agent.shell_timeout_s)),
            _MAX_SHELL_TIMEOUT_S,
        )
        # start_new_session puts the whole pipeline in its own process
        # group: on timeout, killing only the immediate /bin/sh would leave
        # pipeline stages, backgrounded children, or mpirun ranks running
        # detached while the model reads "timed out" as the command being
        # gone — so the timeout kills the group.
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=session.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,  # decoded below with replacement: binary output is evidence too
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=_subprocess_env(session),
            )
        except OSError as e:
            return f"could not start the command: {e}"
        import contextlib as _contextlib
        import os as _os
        import signal as _signal

        def _kill_group() -> None:
            with _contextlib.suppress(ProcessLookupError, PermissionError):  # raced exit
                _os.killpg(process.pid, _signal.SIGKILL)

        try:
            raw_stdout, raw_stderr = process.communicate(timeout=timeout)
        except KeyboardInterrupt:
            # Ctrl-C in the REPL reaches only this process now (the child
            # runs in its own session), so the interrupt must kill the
            # command tree itself or it keeps running detached.
            _kill_group()
            raise
        except subprocess.TimeoutExpired as e:
            _kill_group()
            # Reap the shell itself — wait() cannot block on pipes — but do
            # NOT communicate(): a child that escaped the group (setsid)
            # still holds the pipe ends, and reading to EOF would hang the
            # agent turn on a daemon that never exits. Abandon the pipes.
            with _contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5.0)
            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    with _contextlib.suppress(OSError):
                        stream.close()
            raw_partial = e.stdout
            if isinstance(raw_partial, bytes):
                partial = raw_partial.decode(errors="replace")
            else:
                partial = raw_partial or ""
            return (
                f"command timed out after {timeout:.0f}s; the command and its "
                f"process group were killed; partial output:\n{partial}"
            )
        # A stray byte (an ELF header, a Latin-1 log) once failed the whole
        # call as a UnicodeDecodeError; a replacement character keeps the
        # evidence and the exit code.
        stdout = raw_stdout.decode("utf-8", errors="replace")
        stderr = raw_stderr.decode("utf-8", errors="replace")
        completed = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        output = completed.stdout + (
            f"\n[stderr]\n{completed.stderr}" if completed.stderr.strip() else ""
        )
        return f"exit {completed.returncode}\n{output.rstrip()}"

    box.add(
        Tool(
            name="shell",
            description=(
                "Run one shell command in the project directory (stdout+stderr, "
                "with the exit code). A timeout (timeout_s, default 120 s) kills "
                "the command's whole process group — nohup and '&' do not survive "
                "it — so scope filesystem searches to a directory. Not for long "
                "calculations — use launch_workflow (background=true for long "
                "ones) or submit_job for those."
            ),
            parameters=_schema(
                {"command": {"type": "string"}, "timeout_s": {"type": "number"}},
                ["command"],
            ),
            handler=shell,
            requires_approval=True,
            gate=lambda arguments: not session.shell_allowlisted(
                str(arguments.get("command", ""))
            ),
        )
    )


# -- workflows (foundation) ---------------------------------------------------


def record_command(session: MasonSession, **event: Any) -> None:
    """Record one ``command`` event: what ran, by which card, through which tool.

    The transcript is the record of a campaign, and a reader checking
    what was run must not have to open the run store. Every path that
    starts a process records here: ``shell`` its command line, a
    workflow launch the driver's own command, a job submission the
    payload and the job, and a finished run the engine commands its
    tasks resolved (:func:`foundation._ops.run_commands`). ``kind`` says
    which, ``by`` is the agent card, and ``at`` is stamped by the session.
    """
    session.record({"type": "command", "by": session.agent_name, **event})


def describe_size(size: dict[str, Any] | None) -> str:
    """One phrase for a job's size, for a tool result or a transcript line.

    Examples:
        >>> describe_size({"nodes": 1, "ntasks_per_node": 8, "cpus_per_task": 1,
        ...                "gpus_per_node": 2, "mem": None})
        '1 node(s) x 8 rank(s) x 1 cpu(s), 2 gpu(s) per node'
        >>> describe_size({"nodes": 2, "ntasks_per_node": 64, "cpus_per_task": 2,
        ...                "gpus_per_node": 0, "mem": "240G"})
        '2 node(s) x 64 rank(s) x 2 cpu(s), mem 240G'
        >>> describe_size(None)
        'not sized'
    """
    if not size:
        return "not sized"
    text = (
        f"{size.get('nodes', 1)} node(s) x {size.get('ntasks_per_node', 1)} rank(s) x "
        f"{size.get('cpus_per_task', 1)} cpu(s)"
    )
    if size.get("gpus_per_node"):
        text += f", {size['gpus_per_node']} gpu(s) per node"
    if size.get("mem"):
        text += f", mem {size['mem']}"
    return text


def _count_argument(arguments: dict[str, Any], key: str, *, minimum: int = 1) -> int | None:
    """The integer a size argument holds, or None when it is absent.

    A value that is not an integer, or one below *minimum*, is a
    :class:`ValueError` whose text names the argument; the tools return
    it as a refusal. A zero is never clamped to one: a launch of no ranks
    is a mistake the model should hear about.

    Examples:
        >>> _count_argument({"ntasks": "4"}, "ntasks"), _count_argument({}, "ntasks")
        (4, None)
        >>> _count_argument({"ntasks": "two"}, "ntasks")
        Traceback (most recent call last):
        ...
        ValueError: ntasks must be a positive integer, not 'two'
        >>> _count_argument({"ntasks": 0}, "ntasks")
        Traceback (most recent call last):
        ...
        ValueError: ntasks must be a positive integer, not 0
        >>> _count_argument({"gpus": 0}, "gpus", minimum=0)
        0
    """
    value = arguments.get(key)
    if value is None:
        return None
    wanted = "a positive integer" if minimum > 0 else "zero or a positive integer"
    count: int | None = None
    if isinstance(value, bool):
        count = None
    elif isinstance(value, int):
        count = value
    elif isinstance(value, float) and value.is_integer():
        count = int(value)
    elif isinstance(value, str) and value.strip().lstrip("-").isdigit():
        count = int(value.strip())
    if count is None or count < minimum:
        raise ValueError(f"{key} must be {wanted}, not {value!r}")
    return count


def _reserve_for(session: MasonSession, arguments: dict[str, Any]) -> Reservation | str:
    """Check out the slice a launch asks for, or the refusal text.

    ``ntasks``, ``threads``, and ``gpus`` size the slice; with none of
    them the whole free cpu budget is taken, so an unsized launch is
    accounted for like any other. The session process is the holder
    until the run claims the reservation, and the dead are reaped first
    so a crashed launch never keeps a slice. A slice that does not fit
    comes back as text carrying the free amounts, never as an exception,
    and so does a size that is not a positive integer.
    """
    from foundation.errors import ResourcesError

    try:
        ntasks = _count_argument(arguments, "ntasks")
        threads = _count_argument(arguments, "threads")
        gpus = _count_argument(arguments, "gpus", minimum=0) or 0
    except ValueError as e:
        return f"refused: {e}"
    try:
        with _open_workspace(session) as ws:
            ws.reap_dead(caller="launch_workflow")
            return ws.reserve(ntasks=ntasks, threads=threads, gpus=gpus, holder_pid=os.getpid())
    except ResourcesError as e:
        return f"refused: {e}"
    except RunStoreUnavailable as e:
        return str(e)


def _add_workflow_tools(
    box: Toolbox, session: MasonSession, read_roots: tuple[Path, ...]
) -> None:
    recorded_runs: set[str] = set()

    def _record_run_commands(run_id: str | None, tool: str) -> None:
        """Record the engine commands a finished run resolved, once per run."""
        if not run_id or run_id in recorded_runs:
            return
        recorded_runs.add(run_id)
        try:
            with _open_workspace(session) as ws:
                entries = _ops.run_commands(ws, run_id)
        except (FoundationError, SlabError, sqlite3.Error, OSError) as e:
            record_command(session, kind="engine", tool=tool, run_id=run_id, error=str(e))
            return
        for entry in entries:
            record_command(session, kind="engine", tool=tool, **entry)

    def _run_line(run: Any) -> str:
        line = (
            f"{run.id[:10]}  {_state_text(run):<11} {run.status.value:<10} "
            f"{run.name[:24]:<24} {run.intent or ''}"
        )
        if run.status.value == "running":
            line += f"  [{describe_liveness(run)}]"
        return line

    def _reaped_note(reaped: list[Any], caller: str) -> str:
        if not reaped:
            return ""
        ids = ", ".join(r.id[:10] for r in reaped)
        return (
            f"(marked failed by {caller}: {ids}; each was at status running and its "
            f"recorded process on this host is gone)\n"
        )

    def _session_filter(raw: object) -> str | None:
        # 'this' names the current session, so the agent never has to know
        # its own id — a real transcript showed the guess being made.
        value = str(raw) if raw else None
        return session.session_id if value == "this" else value

    def list_runs(arguments: dict[str, Any]) -> str:
        state = arguments.get("state")
        status = arguments.get("status")
        if status is None and state in ("running", "completed", "failed"):
            # The two words are easy to swap; take the meaning, not the key.
            state, status = None, state
        limit = int(arguments.get("limit", 10))
        session_filter = _session_filter(arguments.get("session"))
        with _open_workspace(session) as ws:
            reaped = _reaped_note(ws.reap_dead(caller="list_runs"), "list_runs")
            runs = ws.runs.list_runs(
                state=state, status=status, session=session_filter, limit=limit
            )
            if not runs:
                where = f" for session {session_filter!r}" if session_filter else ""
                return f"{reaped}no runs in this workspace yet{where}"
            lines = [_run_line(run) for run in runs]
        return reaped + "\n".join(lines)

    box.add(
        Tool(
            name="list_runs",
            description=(
                "List SLAB runs in this workspace, newest first. Pass "
                "session='this' to see only runs this session created "
                "(a full id or unique prefix also works — the same filter "
                "'slab list --session' takes)."
            ),
            parameters=_schema(
                {
                    "state": {
                        "type": "string",
                        "description": "quarantined | verified | promoted | expired",
                    },
                    "status": {
                        "type": "string",
                        "description": "running | completed | failed",
                    },
                    "session": {
                        "type": "string",
                        "description": (
                            "'this' for the current session; or a session id "
                            "(unique prefix ok); omit for all runs"
                        ),
                    },
                    "limit": {"type": "integer"},
                },
                [],
            ),
            handler=list_runs,
        )
    )

    def _resolve_run(ws: Workspace, value: str) -> tuple[str, str]:
        """A run id, a unique prefix, or this session's newest run of that name."""
        return _ops.resolve_run(ws, value, session=session.session_id)

    def show_run(arguments: dict[str, Any]) -> str:
        from foundation._ops import run_details

        with _open_workspace(session) as ws:
            run_id, note = _resolve_run(ws, str(arguments["run_id"]))
            details = run_details(ws, run_id)
            details["run"] = _rendered_run(details["run"], ws.runs.get(run_id))
        if arguments.get("task") is not None:
            details = _one_task(details, arguments["task"])
        elif not arguments.get("full"):
            details = _compact_details(details)
        if not enabled(session.agent, "failure-records"):
            details = _without_failure_records(details)
        return note + json.dumps(details, indent=1, ensure_ascii=False)

    def read_artifact(arguments: dict[str, Any]) -> str:
        from foundation._ops import run_details

        wanted = str(arguments["name"])
        offset = max(int(arguments.get("offset", 1)), 1)
        limit = int(arguments.get("limit", _MAX_READ_LINES))
        with _open_workspace(session) as ws:
            run_id, note = _resolve_run(ws, str(arguments["run_id"]))
            artifacts = run_details(ws, run_id)["artifacts"]
            found = [a for a in artifacts if a["name"] == wanted or a["hash"].startswith(wanted)]
            if not found:
                names = ", ".join(a["name"] for a in artifacts) or "none"
                return f"{note}no artifact named {wanted!r} on run {run_id[:10]}; it has: {names}"
            artifact = found[0]
            if not ws.artifacts.has(artifact["hash"]):
                return (
                    f"{note}the bytes of {wanted!r} are no longer stored (retention "
                    f"reclaimed them); the record keeps its hash {artifact['hash'][:12]}"
                )
            raw = ws.artifacts.get(artifact["hash"]).read_bytes()
        head = (
            f"{note}{artifact['name']} ({artifact['size_bytes']} bytes, "
            f"sha256 {artifact['hash'][:12]})"
        )
        if b"\x00" in raw[:8192]:
            return f"{head}\nlooks binary; read_artifact only reads text"
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        if digested := _digest_unless_raw(arguments, artifact["name"], text, len(lines)):
            return f"{head}\n{digested}"
        window = lines[offset - 1 : offset - 1 + limit]
        numbered = []
        for i, line in enumerate(window, start=offset):
            if len(line) > _MAX_LINE_CHARS:
                line = line[:_MAX_LINE_CHARS] + " [line truncated]"
            numbered.append(f"{i:6d}\t{line}")
        shown = "\n".join(numbered) if numbered else "(no lines in this window)"
        tail = ""
        if len(lines) > offset - 1 + len(window):
            last = offset + len(window) - 1
            tail = f"\n[artifact has {len(lines)} lines; showing {offset}-{last}]"
        return f"{head}\n{shown}{tail}"

    box.add(
        Tool(
            name="show_run",
            description=(
                "One run's record: its fields, checks with observed/expected values, "
                "the task tally with finished tasks folded to one line each, failed "
                "tasks and failure records in full, artifacts, and history. Pass "
                "task=<label or seq> for one task's recipe, inputs, and outputs, or "
                "full=true for every task's. run_id takes an id, a unique prefix, or "
                "the name of a run this session created. Read this before "
                "correcting a failed run; read_artifact reads its files."
            ),
            parameters=_schema(
                {
                    "run_id": {"type": "string"},
                    "task": {
                        "type": "string",
                        "description": "one task's label or seq: its recipe, inputs, outputs",
                    },
                    "full": {
                        "type": "boolean",
                        "description": "include every task's recipe, inputs, and outputs",
                    },
                },
                ["run_id"],
            ),
            handler=show_run,
        )
    )
    box.add(
        Tool(
            name="read_artifact",
            description=(
                "Read one of a run's artifacts. name is the artifact's name from show_run "
                "(or a hash prefix); run_id as for show_run. This is how to read an "
                "engine's output file (a .pwo, a LAMMPS log) after the run: the store "
                "is content-addressed, so do not go looking for the path by hand. A "
                "recognised engine output comes back as a digest first: system, "
                "convergence trace, final numbers, warnings, and whether the job "
                "finished. Pass raw=true, or offset/limit, for the line-numbered text "
                "itself, windowed like read_file."
            ),
            parameters=_schema(
                {
                    "run_id": {"type": "string"},
                    "name": {"type": "string"},
                    "offset": {"type": "integer", "description": "first line, 1-based"},
                    "limit": {"type": "integer", "description": "lines to show (default 400)"},
                    "raw": {
                        "type": "boolean",
                        "description": "the text itself, not the digest of an engine output",
                    },
                },
                ["run_id", "name"],
            ),
            handler=read_artifact,
        )
    )

    def _release(reservation_id: str) -> None:
        """Give a reservation back when the launch it was made for never claims it."""
        with (
            contextlib.suppress(FoundationError, sqlite3.Error, OSError),
            _open_workspace(session) as ws,
        ):
            ws.runs.release_reservation(reservation_id)

    def _launch_child(
        script: Path,
        reservation: Reservation,
        *,
        name: str | None,
        intent: str | None,
        args: list[str],
        wait: bool,
    ) -> dict[str, Any]:
        """Run the script as a child 'foundation run --reservation' process.

        A sized launch never runs in this process: the child takes the
        affinity mask and the GPU variables of its slice, and every rank
        it starts inherits them. A background launch is the same child,
        detached (:func:`foundation._ops.launch_child`), so no tool
        timeout can kill it. The log sits next to the script.
        """
        return _ops.launch_child(
            session.workspace_root,
            script,
            reservation=reservation,
            name=name,
            intent=intent,
            session=session.session_id,
            argv=tuple(args),
            cwd=session.cwd,
            # Block-buffered stdout reaches the log only at exit; a real
            # session polled an empty log eight times while the run was
            # labeling structures. launch_child asks for line by line.
            env=_subprocess_env(session),
            wait=wait,
            # A background log sits next to the script, where the agent
            # has always found it; a waited launch gets a fresh log per
            # reservation, so its answer carries this run's output alone.
            log_path=Path(script).with_suffix(".launch.log") if not wait else None,
        )

    def launch_workflow(arguments: dict[str, Any]) -> str:
        from foundation._ops import launch_script

        script = _resolve(session, str(arguments["script"]))
        if denied := _out_of_scope(session, script, read_roots):
            return denied
        try:
            script_text = Path(script).read_text(encoding="utf-8", errors="replace")
        except OSError:
            script_text = ""  # launch_script reports the unreadable file itself
        args = [str(a) for a in arguments.get("args") or []]
        name = str(arguments["name"]) if arguments.get("name") else None
        intent = str(arguments["intent"]) if arguments.get("intent") else None
        background = bool(arguments.get("background"))
        sized = any(arguments.get(key) is not None for key in ("ntasks", "threads", "gpus"))
        reserved = _reserve_for(session, arguments)
        if isinstance(reserved, str):
            return reserved
        reservation = reserved
        # A hand-written mpirun in the script is judged against the slice
        # the run will hold, not the whole budget: the mask bounds it.
        if refused := _rank_overcommit(script_text, limit=len(reservation.cpus)):
            _release(reservation.id)
            return refused
        as_child = sized or background
        driver = ["slab", "run", str(script), *args]
        if name:
            driver += ["--name", name]
        if intent:
            driver += ["--intent", intent]
        driver += ["--session", session.session_id, "-w", str(session.workspace_root)]
        if as_child:
            driver += ["--reservation", reservation.id]
        record_command(
            session,
            kind="launch",
            tool="launch_workflow",
            command=shlex.join(driver),
            script=str(script),
            args=args,
            cwd=str(session.cwd),
            background=background,
            sized=sized,
            resources=reservation.slice,
            reservation=reservation.id,
        )
        held = _ops.describe_resources(reservation.slice)
        try:
            if background:
                launched = _launch_child(
                    script, reservation, name=name, intent=intent, args=args, wait=False
                )
                return (
                    f"launched in the background: pid {launched['pid']}, output -> "
                    f"{launched['log']}; holding {held}\n"
                    f"the run appears in list_runs (session='this') once it starts; "
                    f"block on it with wait_for_run instead of polling in shell."
                )
            if sized:
                result = _launch_child(
                    script, reservation, name=name, intent=intent, args=args, wait=True
                )
            else:
                result = launch_script(
                    session.workspace_root,
                    script,
                    name=name,
                    intent=intent,
                    session=session.session_id,
                    argv=tuple(args),
                    capture_output=True,
                    reservation=reservation,
                )
        except (FoundationError, SlabError, OSError) as e:
            _release(reservation.id)
            return f"could not start the run: {e}"
        _record_run_commands(result.get("run_id"), "launch_workflow")
        lines = [
            f"run {result['run_id']}: state={result['state']} status={result['status']} "
            f"checks={result['checks_passed']}/{result['checks_total']} "
            f"tasks={result['tasks_recorded']}"
        ]
        if result.get("failure"):
            if enabled(session.agent, "failure-records"):
                lines.append("failure record:")
                lines.append(json.dumps(result["failure"], indent=1, ensure_ascii=False))
        elif result.get("traceback"):
            lines.append(str(result["traceback"]))
        lines.append(f"resources held: {held}")
        output = str(result.get("output") or "").rstrip()
        if output:
            lines.append(f"script output:\n{output}")
        return "\n".join(lines)

    box.add(
        Tool(
            name="launch_workflow",
            description=(
                "Execute a SLAB workflow script (plain Python with @task calls and "
                "@check verification) as a traced run. This is how calculations "
                "run: results get provenance, caching, and verification gates. "
                "Size the run with ntasks (MPI ranks), threads (per rank), and gpus: "
                "the slice is reserved on this host before the run starts, the run "
                "takes it as an affinity mask plus CUDA_VISIBLE_DEVICES, and an "
                "engine command with {ntasks}/{threads}/{gpus} placeholders fills "
                "from it. A size that does not fit what is free is refused with the "
                "free amounts (list_engines reports budget and free). A launch "
                "without ntasks or threads takes every free cpu: unsized, it takes "
                "every free cpu and no gpu; gpus without ntasks takes every free cpu "
                "and the gpus asked. "
                "For work longer than a few minutes, pass background=true: the "
                "run detaches from this process (no tool timeout can kill it) "
                "and wait_for_run blocks until it finishes."
            ),
            parameters=_schema(
                {
                    "script": {"type": "string", "description": "path to the workflow script"},
                    "name": {"type": "string"},
                    "intent": {"type": "string", "description": "why this run exists"},
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "arguments passed to the script (sys.argv[1:])",
                    },
                    "background": {
                        "type": "boolean",
                        "description": (
                            "detach and return immediately; follow with wait_for_run"
                        ),
                    },
                    "ntasks": {
                        "type": "integer",
                        "description": "MPI ranks the run may start (cpus = ntasks x threads)",
                    },
                    "threads": {
                        "type": "integer",
                        "description": "threads per rank (OMP_NUM_THREADS); default 1",
                    },
                    "gpus": {
                        "type": "integer",
                        "description": "gpus the run holds (CUDA_VISIBLE_DEVICES); default 0",
                    },
                },
                ["script"],
            ),
            handler=launch_workflow,
            requires_approval=True,
        )
    )

    def wait_for_run(arguments: dict[str, Any]) -> str:
        wanted = arguments.get("run_id")
        timeout = min(float(arguments.get("timeout_s", 900.0)), _MAX_WAIT_TIMEOUT_S)
        with _open_workspace(session):
            pass  # a store that cannot be opened is named here, with the recovery
        # The wait reaps the dead on every poll (foundation._ops.wait_for_run).
        waited = _ops.wait_for_run(
            session.workspace_root,
            run_id=str(wanted) if wanted else None,
            session=session.session_id,
            timeout_s=timeout,
            poll_s=_WAIT_POLL_S,
            grace_s=_WAIT_GRACE_S,
        )
        note = waited["note"]
        outcome = waited["outcome"]
        if outcome in ("finished", "process_gone"):
            _record_run_commands(waited["run"].id, "wait_for_run")
        elif outcome == "none_running":
            for finished in waited["runs"]:
                _record_run_commands(finished.id, "wait_for_run")
        if outcome == "process_gone":
            run = waited["run"]
            return (
                f"{note}this run's process is gone: run {run.id} was running as "
                f"process {run.pid} on {run.host}, and that process no longer exists; "
                f"marked failed by wait_for_run. {waited['progress']}; read it with "
                f"show_run, and launch again if the work is still wanted"
            )
        if outcome == "finished":
            run = waited["run"]
            return (
                f"{note}run {run.id}: state={_state_text(run)} "
                f"status={run.status.value}; {waited['progress']}; "
                f"read it with show_run"
            )
        if outcome == "no_runs":
            return (
                "this session has no runs yet; launch one first "
                "(a fresh background launch needs a few seconds to register)"
            )
        if outcome == "none_running":
            finished = "\n".join(_run_line(r) for r in waited["runs"])
            return f"no run of this session is running; the record:\n{finished}"
        # The tally answers "is it moving?" without a show_run of the whole
        # record: a real session queried the database by hand for this count.
        lines = "\n".join(
            f"{r.id[:10]}  {r.name}  running; {progress}; {liveness}"
            for r, progress, liveness in waited["running"]
        ) or "(none registered yet)"
        return (
            f"{note}still running after {timeout:.0f}s:\n{lines}\n"
            f"call wait_for_run again to keep waiting"
        )

    box.add(
        Tool(
            name="wait_for_run",
            description=(
                "Block until a run finishes (or the timeout passes), then report "
                "its state and task tally. run_id takes an id, a unique prefix, or "
                "the name of a run this session created; without it, waits for "
                "every running run this session created — the partner of "
                "launch_workflow background=true. One call replaces a chain of "
                "sleep-and-poll shell commands, and the timeout answer says how "
                "far each run has got."
            ),
            parameters=_schema(
                {
                    "run_id": {
                        "type": "string",
                        "description": "run id or unique prefix; omit for this session's runs",
                    },
                    "timeout_s": {
                        "type": "number",
                        "description": "seconds to wait before reporting back (default 900)",
                    },
                },
                [],
            ),
            handler=wait_for_run,
        )
    )


# -- engines (slab) -----------------------------------------------------------


def _add_engine_tools(box: Toolbox, session: MasonSession) -> None:
    def list_engines(arguments: dict[str, Any]) -> str:
        from slab._ops import engines_overview
        from slab.resources import budget as discover_budget

        overview = engines_overview()
        try:
            with _open_workspace(session) as ws:
                ws.reap_dead(caller="list_engines")
                resources = ws.free_resources()
        except RunStoreUnavailable as e:
            # The engines are still the answer; the budget needs no store,
            # and what is free cannot be known without one.
            found = discover_budget()
            overview["budget"] = {"cpus": len(found.cpus), "gpus": len(found.gpus)}
            overview["free"] = None
            overview["resources_note"] = f"run store unavailable: {e}"
            return json.dumps(overview, indent=1, ensure_ascii=False)
        overview["budget"] = {key: len(ids) for key, ids in resources["budget"].items()}
        overview["free"] = {key: len(ids) for key, ids in resources["free"].items()}
        return json.dumps(overview, indent=1, ensure_ascii=False)

    box.add(
        Tool(
            name="list_engines",
            description=(
                "What can be computed here: engines, QE protocols, pseudopotential "
                "families, HPC partitions with each node's declared size, and this "
                "host's cpu/gpu 'budget' with what is 'free' right now (size "
                "launch_workflow within it). Call this BEFORE choosing an engine — "
                "there is no in-process MLIP fallback, so the available checkpoint "
                "ids are the entire runnable-MLIP surface on this machine "
                "(training a new one is the train_potential task, not an engine)."
            ),
            parameters=_schema({}, []),
            handler=list_engines,
        )
    )

    def list_tasks(arguments: dict[str, Any]) -> str:
        entries = [
            f"{t['name']}({t['signature']}) — {t['summary']}" for t in _ops.task_catalog()
        ]
        return "\n".join(entries)

    box.add(
        Tool(
            name="list_tasks",
            description=(
                "The traced tasks foundation.tasks exposes to workflow scripts. "
                "One line per task: name, signature, one-sentence summary. Call "
                "describe_task for the full docstring."
            ),
            parameters=_schema({}, []),
            handler=list_tasks,
        )
    )

    def describe_task(arguments: dict[str, Any]) -> str:
        task = _ops.describe_task(str(arguments.get("name", "")))
        return f"{task['name']}({task['signature']})\n\n{task['doc']}"

    box.add(
        Tool(
            name="describe_task",
            description=(
                "Full signature and docstring of one foundation.tasks task, so an "
                "agent can consult the harness's own vocabulary instead of reading "
                "its source through shell."
            ),
            parameters=_schema(
                {"name": {"type": "string", "description": "e.g. 'relax'"}},
                ["name"],
            ),
            handler=describe_task,
        )
    )


def _add_mp_tools(box: Toolbox, snapshot_root: Path) -> None:
    """The offline Materials Project snapshot ([builders.mp] is configured).

    Search and lookup only: the traced route from a material id to a
    structure is ``foundation.tasks.fetch_structure`` inside a workflow
    script. All three tools are read-only by construction, and they bind
    the root resolved from the session's own project config — the process
    cwd may be elsewhere.
    """

    def search_materials(arguments: dict[str, Any]) -> str:
        from slab.mp import search_materials as mp_search

        rows = mp_search(
            arguments.get("filters") or {},
            columns=arguments.get("columns"),
            limit=int(arguments.get("limit", 20)),
            order_by=arguments.get("order_by"),
            root=snapshot_root,
        )
        return json.dumps(rows, indent=1, ensure_ascii=False, default=str)

    box.add(
        Tool(
            name="search_materials",
            description=(
                "Search the offline Materials Project snapshot's materials table "
                "(local, read-only; there is no online fallback). filters maps "
                "keys to values: 'elements' (all must be present) and "
                "'exclude_elements' take element lists; other keys are columns, "
                "bare for equality (null matches SQL NULL) or suffixed "
                "__lte/__gte/__lt/__gt/__ne — e.g. {\"elements\": [\"Fe\"], "
                "\"energy_above_hull__lte\": 0.025}. A wrong column name is "
                "refused with the real column list. Search first and fetch one "
                "structure second (fetch_structure in a workflow); never "
                "enumerate the cifs/ tree. NULL means not populated, never "
                "zero. Report results as (snapshot release, material_id)."
            ),
            parameters=_schema(
                {
                    "filters": {"type": "object"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "limit": {"type": "integer", "description": "1-500, default 20"},
                    "order_by": {
                        "type": "string",
                        "description": "column name; leading - for descending",
                    },
                },
                [],
            ),
            handler=search_materials,
        )
    )

    def get_material(arguments: dict[str, Any]) -> str:
        from slab.mp import get_material as mp_get

        record = mp_get(str(arguments["material_id"]), root=snapshot_root)
        return json.dumps(record, indent=1, ensure_ascii=False, default=str)

    box.add(
        Tool(
            name="get_material",
            description=(
                "One material's full metadata record from the snapshot: the "
                "materials row, its elements, and cif_file — the absolute path "
                "of its archived CIF, readable here or via "
                "fetch_structure(material_id) in a workflow script. Absence is "
                "absence: an id the snapshot lacks is an error, not a reason "
                "to look elsewhere."
            ),
            parameters=_schema(
                {"material_id": {"type": "string", "description": "e.g. 'mp-149'"}},
                ["material_id"],
            ),
            handler=get_material,
        )
    )

    def query_materials(arguments: dict[str, Any]) -> str:
        from slab.mp import query_materials as mp_query

        result = mp_query(
            str(arguments["sql"]),
            limit=int(arguments.get("limit", 200)),
            root=snapshot_root,
        )
        return json.dumps(result, indent=1, ensure_ascii=False, default=str)

    box.add(
        Tool(
            name="query_materials",
            description=(
                "One read-only SELECT (or WITH) over the snapshot's "
                "metadata.sqlite, for queries the search_materials filters "
                "cannot express. Tables: materials (keyed by material_id), "
                "material_elements(material_id, element), dataset_info, units "
                "(consult it instead of guessing units). Rows are capped and "
                "the result says when it truncated — put LIMIT in the query."
            ),
            parameters=_schema(
                {
                    "sql": {"type": "string"},
                    "limit": {"type": "integer", "description": "row cap, default 200"},
                },
                ["sql"],
            ),
            handler=query_materials,
        )
    )


# -- hpc ---------------------------------------------------------------------


def _add_hpc_tools(box: Toolbox, session: MasonSession) -> None:
    def submit_job(arguments: dict[str, Any]) -> str:
        from slab.errors import JobSizeError
        from slab.resources import job_size

        # A size the node cannot hold, or a size on a partition without a
        # node table, is a refusal the model reads: it names the cap and
        # the config field, so the next call fits or the operator adds it.
        # So is a size that is not a positive integer.
        try:
            size = job_size(
                nodes=_count_argument(arguments, "nodes"),
                ntasks_per_node=_count_argument(arguments, "ntasks_per_node"),
                cpus_per_task=_count_argument(arguments, "cpus_per_task"),
                gpus_per_node=_count_argument(arguments, "gpus_per_node", minimum=0),
                mem=None if arguments.get("mem") is None else str(arguments["mem"]),
            )
        except (JobSizeError, ValueError) as e:
            return f"refused: {e}"
        try:
            job = _ops.submit_job(
                session.workspace_root,
                hpc=session.hpc,
                command=str(arguments["command"]),
                name=str(arguments["name"]),
                partition=arguments.get("partition"),
                time_limit=arguments.get("time_limit"),
                session=session.session_id,
                project=session.cwd,
                size=size,
            )
        except JobSizeError as e:
            return f"refused: {e}"
        record_command(
            session,
            kind="job",
            tool="submit_job",
            command=str(arguments["command"]),
            job_id=str(job["job_id"]),
            job_name=str(job["job_name"]),
            partition=str(job["partition"]),
            script=str(job["script_path"]),
            cwd=str(session.cwd),
            size=job["size"],
        )
        sized = f"; sized {describe_size(job['size'])}" if job["size"] else ""
        return (
            f"submitted job {job['job_id']} ({job['job_name']}) to partition "
            f"{job['partition']}{sized}; script kept at {job['script_path']}; "
            f"poll with job_status"
        )

    box.add(
        Tool(
            name="submit_job",
            description=(
                "Submit a command as a SLURM batch job (typically 'slab run "
                "workflow.py' so the result is still a traced, verified run). "
                "Size the job with ntasks_per_node (required to size), cpus_per_task, "
                "gpus_per_node, nodes, and mem (e.g. 240G): the size replaces the "
                "partition's own directives and must fit the node that list_engines "
                "reports under hpc.partitions.<name>.node; without a size the "
                "partition's directives apply as declared."
            ),
            parameters=_schema(
                {
                    "command": {"type": "string"},
                    "name": {"type": "string"},
                    "partition": {"type": "string"},
                    "time_limit": {"type": "string", "description": "HH:MM:SS"},
                    "nodes": {"type": "integer", "description": "nodes (default 1)"},
                    "ntasks_per_node": {
                        "type": "integer",
                        "description": "MPI ranks per node; naming it makes the job sized",
                    },
                    "cpus_per_task": {"type": "integer", "description": "cpus per rank"},
                    "gpus_per_node": {"type": "integer", "description": "gpus per node"},
                    "mem": {"type": "string", "description": "memory per node, e.g. 240G"},
                },
                ["command", "name"],
            ),
            handler=submit_job,
            requires_approval=True,
        )
    )

    def job_status(arguments: dict[str, Any]) -> str:
        from slab.hpc import job_state

        status = job_state(str(arguments["job_id"]))
        pieces = [f"job {status.job_id}: {status.state.value}"]
        if status.raw and status.raw != status.state.value.upper():
            pieces.append(f"({status.raw})")
        if status.detail:
            pieces.append(status.detail)
        return " ".join(pieces)

    box.add(
        Tool(
            name="job_status",
            description="State of one SLURM job (pending/running/completed/failed/...).",
            parameters=_schema({"job_id": {"type": "string"}}, ["job_id"]),
            handler=job_status,
        )
    )

    def cancel_job(arguments: dict[str, Any]) -> str:
        from slab.hpc import cancel

        cancel(str(arguments["job_id"]))
        return f"cancel requested for job {arguments['job_id']}"

    box.add(
        Tool(
            name="cancel_job",
            description="Cancel a SLURM job (a no-op if it already finished).",
            parameters=_schema({"job_id": {"type": "string"}}, ["job_id"]),
            handler=cancel_job,
            requires_approval=True,
        )
    )


# -- delegation ---------------------------------------------------------------


def _run_child(
    session: MasonSession,
    roster: dict[str, AgentSpec],
    skills: dict[str, Skill],
    parent_client: Any | None,
    target: AgentSpec,
    brief: str,
) -> tuple[Any, MasonSession]:
    """Run *target*'s own loop on *brief*, one level down; (result, child session).

    The one place a tool spins a loop of its own, shared by ``delegate``
    and ``review``. The child derives from the *base* config so the entry
    agent's own [agent.roster] table never leaks into it; CLI flags are
    re-asserted on top because a flag outranks config for everyone.
    """
    # Local imports: tools must not import the loop at module scope (the
    # loop imports tools).
    from mason.client import LlmError
    from mason.config import override_agent, roster_agent_config
    from mason.loop import Mason, TurnResult, client_from_config, connection_profile

    effective = roster_agent_config(session.base_agent, target.name)
    if session.flag_updates:
        effective = override_agent(effective, dict(session.flag_updates))
    child_session = session.spawn(target.name, effective)
    reuse = parent_client is not None and connection_profile(
        child_session.agent
    ) == connection_profile(session.agent)
    client = (
        parent_client
        if reuse
        else client_from_config(child_session.agent, child_session.api_keys)
    )
    child = Mason(
        child_session, client=client, skills=skills, spec=target, roster=roster, depth=1
    )
    try:
        result = child.run_turn(brief)
    except LlmError as e:
        # The server failed the child mid-turn, after the client's own
        # retries. The steps it took are in its transcript, so the parent
        # gets a result that says so instead of an exception that discards
        # them: one real critic lost five steps and twenty minutes to a
        # gateway 502, and the lead paid for the review twice.
        result = TurnResult(
            text=(
                f"stopped: the model server failed mid-turn after step "
                f"{child.steps_taken} ({e}); the transcript holds the steps taken"
            ),
            stop_reason="error",
            steps=child.steps_taken,
        )
    return result, child_session


def _digest_unless_raw(arguments: dict[str, Any], name: str, text: str, n_lines: int) -> str | None:
    """The digest of an engine output, unless the caller asked for the text.

    A window (``offset`` or ``limit``) or ``raw=true`` is a request for the
    text. Anything the digest module does not recognise returns ``None``,
    and the caller shows the text as it always did. One real session read
    a 305 KB pw.x output in 400-line windows, mistook a block of band
    eigenvalues in eV for a diverging SCF energy in Ry, and compacted six
    times in sixteen minutes arguing with itself about it.
    """
    if arguments.get("raw") or "offset" in arguments or "limit" in arguments:
        return None
    from slab.outputs import digest

    digested = digest(name, text)
    if digested is None:
        return None
    return f"{digested}\n[digest of {n_lines} lines; pass raw=true, or offset/limit, for the text]"


def _harness_footer(name: str, result: Any, child_session: MasonSession) -> str:
    """The bracketed line a lead reads before trusting a child's report."""
    return (
        f"[{name}: {result.stop_reason} after {result.steps} step(s); "
        f"tokens {child_session.prompt_tokens}+{child_session.completion_tokens}; "
        f"transcript {child_session.transcript_path.name}]"
    )


def _add_delegate_tool(
    box: Toolbox,
    session: MasonSession,
    spec: AgentSpec,
    roster: dict[str, AgentSpec],
    skills: dict[str, Skill],
    parent_client: Any | None,
) -> None:
    def delegate(arguments: dict[str, Any]) -> str:
        from mason.roster import hands

        name = str(arguments["agent"])
        team = hands(spec, roster)
        others = ", ".join(team)
        if name == spec.name:
            return f"you cannot delegate to yourself; your team: {others}"
        target = team.get(name)
        if target is None:
            if name in roster and roster[name].reviews:
                return (
                    f"{name} reviews and takes no briefs; hand it the plan or a file "
                    f"with the review tool. your team: {others}"
                )
            if name in roster:
                return f"{name} leads a group of its own and takes no briefs; your team: {others}"
            return f"no agent named {name!r}; your team: {others}"
        task = str(arguments["task"])
        context = arguments.get("context")
        brief = task if not context else f"{task}\n\nContext from {spec.name}:\n{context}"
        result, child_session = _run_child(session, roster, skills, parent_client, target, brief)
        session.record(
            {
                "type": "delegate",
                "agent": name,
                "task": task,
                "transcript": child_session.transcript_path.name,
                "stop": result.stop_reason,
                "steps": result.steps,
            }
        )
        return f"{result.text}\n\n{_harness_footer(name, result, child_session)}"

    box.add(
        Tool(
            name="delegate",
            description=(
                "Hand one scoped task to a specialist from your team. The "
                "specialist runs its own tool loop against the shared workspace "
                "and notebook, and you receive its final report. Brief it with "
                "the goal, the constraints (engine, protocol, budget), and what "
                "to return; its report ends with a bracketed harness line "
                "stating how it stopped."
            ),
            parameters=_schema(
                {
                    "agent": {"type": "string", "description": "a name from Your team"},
                    "task": {
                        "type": "string",
                        "description": "the scoped goal, self-contained and checkable",
                    },
                    "context": {
                        "type": "string",
                        "description": "optional background the task needs",
                    },
                },
                ["agent", "task"],
            ),
            handler=delegate,
        )
    )


# -- review: a critic before compute -----------------------------------------


#: How much of the prior review's findings a re-review brief carries.
_PRIOR_FINDINGS_CHARS = 4_000


def _review_brief(
    label: str,
    text: str,
    focus: object,
    facts: dict[str, str] | None = None,
    prior: Any | None = None,
) -> str:
    """The critic's brief: what to judge, how to report, the lead's facts, the text.

    *facts* are the lead's own catalog and material lookups this session,
    newest first, each cut to ``_FACT_CHARS`` and all of them to
    ``_FACTS_CHARS``, and the brief says not to re-run them. *prior* is
    this session's last review of the same subject: the brief then asks
    first whether its findings are resolved. Both were once left to the
    lead's ``focus``; a real lead learned to write "do NOT re-verify
    machine facts" only after a review had run for 78 minutes without a
    verdict, and the scoped re-review then took ten.
    """
    parts = [
        f"Review {label} before compute is spent on it. Judge it by the questions "
        f"in your card, quote what you judge, and number every finding as blocking "
        f"or advisory with the change that would resolve it.",
    ]
    if prior is not None and prior.findings.strip():
        parts.append(
            f"This is a re-review. Your prior review of {label} (verdict: "
            f"{prior.verdict}) found:\n"
            f"{_truncate_middle(prior.findings.strip(), _PRIOR_FINDINGS_CHARS)}\n"
            f"Take those findings in order and say for each whether the text as it "
            f"reads now resolves it. Do not re-verify what the text records as "
            f"verified this session unless a finding turns on it. Add a new finding "
            f"only when it is blocking."
        )
    if focus:
        parts.append(f"Focus from the lead: {focus}")
    if facts:
        parts.append(
            "The evidence block below holds the lead's recorded lookups from this "
            "session. Do not re-run them; re-check only what contradicts the text."
        )
    parts.append(
        "Decide the verdict before you write, and keep each finding to one line "
        "plus the change that resolves it. Then call finish alone, with the verdict "
        "in its `verdict` argument: approve when no finding is blocking, revise "
        "when one is."
    )
    if facts:
        parts.append(_facts_block(facts))
    parts.append(f"--- {label} ---\n{text.rstrip()}\n--- end ---")
    return "\n\n".join(parts)


def _facts_block(facts: dict[str, str]) -> str:
    """The lead's lookups as one block, newest first, within the budgets.

    Examples:
        >>> facts = {"list_engines()": "emt", "describe_task(name=relax)": "relax(...)"}
        >>> block = _facts_block(facts)
        >>> block.splitlines()[0]
        '--- evidence the lead gathered this session (re-check what you doubt) ---'
        >>> block.index("describe_task") < block.index("list_engines")
        True
    """
    lines = ["--- evidence the lead gathered this session (re-check what you doubt) ---"]
    used = 0
    for key, body in reversed(list(facts.items())):
        entry = f"## {key}\n{_truncate_middle(body.rstrip(), _FACT_CHARS)}"
        if used + len(entry) > _FACTS_CHARS:
            break
        lines.append(entry)
        used += len(entry)
    lines.append("--- end of the lead's evidence ---")
    return "\n".join(lines)


_VERDICT_HEAD = {
    "approve": "verdict: approve. No finding is blocking; compute may be spent on it.",
    "revise": "verdict: revise. Resolve the blocking findings, then review again.",
    "none": (
        "verdict: none. The critic ended without a verdict; nothing is approved. "
        "Read the harness line, then review again."
    ),
}
#: The head when the critic's answer was cut at its reply-token ceiling on
#: the retry too. Written for the lead, which cannot change the config
#: mid-run: it names the lead's one move and the operator's.
_CUT_VERDICT_HEAD = (
    "verdict: none. The critic's reply was cut at its reply-token ceiling, on the "
    "low-effort retry too, so nothing is approved. Review again: the brief will "
    "carry this review's findings, so keep the focus to whether they are resolved. "
    "Lowering the critic's effort or raising its max_reply_tokens is the operator's "
    "move, in [agent.roster.critic], not yours."
)


def _add_review_tool(
    box: Toolbox,
    session: MasonSession,
    spec: AgentSpec,
    roster: dict[str, AgentSpec],
    skills: dict[str, Skill],
    parent_client: Any | None,
    read_roots: tuple[Path, ...],
) -> None:
    from mason.reviews import PLAN_SUBJECT, Verdict, digest, prior_review, write_review

    def review(arguments: dict[str, Any]) -> str:
        from mason.roster import critics

        available = critics(roster)
        chosen = arguments.get("agent")
        if chosen is None:
            target = next(iter(available.values()))
        else:
            found = available.get(str(chosen))
            if found is None:
                return (
                    f"no critic named {chosen!r}; the critics on the roster: "
                    f"{', '.join(available)}"
                )
            target = found
        subject = str(arguments.get("subject") or PLAN_SUBJECT)
        if subject == PLAN_SUBJECT:
            text = session.plan_text()
            if not text.strip():
                return (
                    "nothing to review: PLAN.md does not exist or is empty; write the "
                    "plan with the plan tool first"
                )
            label = "the plan (PLAN.md)"
        else:
            path = _resolve(session, subject)
            if refused := _out_of_scope(session, path, read_roots):
                return refused
            if not path.is_file():
                return f"nothing to review: {path} is not a file; pass 'plan' or a file path"
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return f"refused: {path} is not a text file"
            label = f"the file {path}"
        prior = prior_review(session, subject)
        brief = _review_brief(label, text, arguments.get("focus"), session.facts, prior)
        result, child_session = _run_child(session, roster, skills, parent_client, target, brief)
        verdict: Verdict = (
            result.verdict
            if result.stop_reason == "finish" and result.verdict in ("approve", "revise")
            else "none"
        )
        record = write_review(
            session,
            subject=subject,
            text=text,
            verdict=verdict,
            reviewer=target.name,
            transcript=child_session.transcript_path.name,
            findings=result.text,
        )
        if (
            subject == PLAN_SUBJECT
            and verdict == "approve"
            and digest(session.plan_text()) == digest(text)
        ):
            session.plan_approved = True
        session.record(
            {
                "type": "review",
                "agent": target.name,
                "subject": subject,
                "verdict": verdict,
                "record": record.name,
                "transcript": child_session.transcript_path.name,
                "stop": result.stop_reason,
                "steps": result.steps,
            }
        )
        head = _CUT_VERDICT_HEAD if getattr(result, "truncated", False) else _VERDICT_HEAD[verdict]
        return (
            f"{head}\n\n{result.text}\n\n"
            f"{_harness_footer(target.name, result, child_session)}\n"
            f"[review recorded in {record}]"
        )

    box.add(
        Tool(
            name="review",
            description=(
                "Hand the plan, or a file, to the critic before compute is spent on "
                "it. The critic is read-only: it runs its own loop against the "
                "shared workspace, returns numbered findings marked blocking or "
                "advisory, and ends with a verdict, approve or revise. The findings "
                "are kept as a review record. Review the plan before the first "
                "brief or launch, and again when the plan changes in substance."
            ),
            parameters=_schema(
                {
                    "subject": {
                        "type": "string",
                        "description": "'plan' (the default) for PLAN.md, or a file path",
                    },
                    "focus": {
                        "type": "string",
                        "description": "optional: what the critic should scrutinize most",
                    },
                    "agent": {
                        "type": "string",
                        "description": "optional: a critic's name when the roster has several",
                    },
                },
                [],
            ),
            handler=review,
        )
    )


def _plan_gate_refusal(session: MasonSession) -> str | None:
    """Why a review_first card may not spend compute yet, or None when it may."""
    if session.plan_approved:
        return None
    if not session.plan_text().strip():
        return (
            "refused: this card spends no compute before the critic approves the plan, "
            "and PLAN.md is empty; write the plan with the plan tool, then call review"
        )
    return (
        "refused: the plan has not been approved by the critic; call review (subject "
        "'plan'), resolve the blocking findings in the plan, and review again until "
        "the verdict is approve"
    )


def _gated(tool: Tool, session: MasonSession) -> Tool:
    """*tool*, refusing until the plan is approved — before any approval prompt."""

    def handler(arguments: dict[str, Any]) -> str:
        refusal = _plan_gate_refusal(session)
        return refusal if refusal is not None else tool.handler(arguments)

    def gate(arguments: dict[str, Any]) -> bool:
        # A call the gate will refuse never asks the user to approve it.
        return _plan_gate_refusal(session) is None and tool.needs_approval(arguments)

    return replace(tool, handler=handler, gate=gate)


def _gate_until_reviewed(box: Toolbox, session: MasonSession) -> None:
    for name in _GATED_UNTIL_REVIEWED:
        tool = box.tools.get(name)
        if tool is not None:
            box.tools[name] = _gated(tool, session)


# -- skills ------------------------------------------------------------------


def _add_skill_tool(box: Toolbox, session: MasonSession, skills: dict[str, Skill]) -> None:
    def skill_tool(arguments: dict[str, Any]) -> str:
        name = str(arguments["name"])
        found = skills.get(name)
        if found is None:
            known = ", ".join(sorted(skills))
            return f"no skill named {name!r}; available skills: {known}"
        # The digest names the revision that loaded, so a benchmark flag
        # raised on this campaign is attributable to one revision of the skill.
        session.record(
            {"type": "skill", "name": name, "source": found.source, "digest": found.digest}
        )
        return listing(found)

    box.add(
        Tool(
            name="skill",
            description=(
                "Load a skill by name: returns its full instructions, its root "
                "path, and its bundled files (scripts, references, assets). Call "
                "this before doing a task a listed skill covers, and prefer its "
                "bundled scripts over writing your own."
            ),
            parameters=_schema({"name": {"type": "string"}}, ["name"]),
            handler=skill_tool,
        )
    )


# -- project memory: the notebook and the plan --------------------------------


def _add_memory_tools(box: Toolbox, session: MasonSession) -> None:
    def notebook(arguments: dict[str, Any]) -> str:
        session.notebook_append(
            str(arguments["entry"]),
            heading=arguments.get("heading"),
        )
        return f"recorded in {session.notebook_path.name}"

    box.add(
        Tool(
            name="notebook",
            description=(
                "Append an entry to the lab notebook (NOTEBOOK.md): decisions, "
                "results with run ids, failures and their diagnosis. The notebook "
                "outlives the context window — write it as if for a colleague."
            ),
            parameters=_schema(
                {"entry": {"type": "string"}, "heading": {"type": "string"}},
                ["entry"],
            ),
            handler=notebook,
        )
    )

    def plan(arguments: dict[str, Any]) -> str:
        content = str(arguments["content"]).rstrip() + "\n"
        plan_write(session.cwd, content)
        return f"PLAN.md updated:\n{content}"

    box.add(
        Tool(
            name="plan",
            description=(
                "Rewrite the living plan (PLAN.md): goal, steps with status, open "
                "questions. Keep it current — it is re-read at session start and "
                "after compaction."
            ),
            parameters=_schema({"content": {"type": "string"}}, ["content"]),
            handler=plan,
        )
    )


# -- machine memory: what this machine taught an earlier session --------------


def _add_machine_memory_tools(box: Toolbox, session: MasonSession) -> None:
    """``recall`` and ``remember``: the store in :mod:`foundation.memory`.

    The notebook holds the project's record; this holds the machine's. Both
    are memory, and the split is the scope: a fact about how software behaves
    here outlives the project that discovered it, so it must not be buried in
    one project's notebook.
    """

    def recall(arguments: dict[str, Any]) -> str:
        name = str(arguments["name"])
        memories = memory_store.discover()
        found = memories.get(name)
        if found is None:
            known = ", ".join(sorted(memories)) or "none recorded yet"
            return f"no memory named {name!r}; memories on this machine: {known}"
        session.record({"type": "recall", "name": name})
        answer = f"{found.body().rstrip()}\n\n[{found.provenance()}]"
        changed = found.drift(session.software_versions()) if found.against else []
        if changed:
            answer += (
                f"\n[changed since: {'; '.join(changed)}. Confirm the fact before "
                f"you build on it; remember it again once you have.]"
            )
        return answer

    box.add(
        Tool(
            name="recall",
            description=(
                "Read one memory in full by name. The catalog under '# Memory' "
                "lists what this machine knows; each line is a summary, and this "
                "returns the fact itself with who recorded it and when."
            ),
            parameters=_schema({"name": {"type": "string"}}, ["name"]),
            handler=recall,
        )
    )

    def remember(arguments: dict[str, Any]) -> str:
        name = str(arguments["name"])
        description = str(arguments["description"])
        body = str(arguments["body"])
        try:
            written = memory_store.write(
                name,
                description,
                body,
                agent=session.agent_name,
                model=session.agent.model,
                # Stamped with the software the text names, at today's
                # versions, so a later session can tell whether it still holds.
                against=memory_store.stamp(f"{description}\n{body}", session.software_versions()),
            )
        except MemoryStoreError as e:
            # A refusal is an observation the model can act on, not a crash:
            # the message says which rule stopped the write.
            return f"not recorded: {e}"
        session.record({"type": "remember", "name": name, "path": str(written.path)})
        answer = (
            f"recorded as memory {written.name!r} in {written.path}; "
            f"every later session on this machine reads it"
        )
        if written.against:
            stamped = ", ".join(f"{n} {v}" for n, v in written.against.items())
            answer += f" (stamped against {stamped})"
        return answer

    box.add(
        Tool(
            name="remember",
            description=(
                "Record one confirmed fact about this machine or its software so "
                "later sessions start knowing it: a package that behaves unlike "
                "its documentation, a flag that matters, a workaround. Write the "
                "description as the line a future session reads when deciding "
                "whether the fact applies. Re-using a name replaces that memory, "
                "which is how you consolidate. Not for results (they belong to "
                "runs), project decisions (the notebook), or credentials (nowhere)."
            ),
            parameters=_schema(
                {
                    "name": {
                        "type": "string",
                        "description": "lowercase-with-hyphens, e.g. 'vllm-mamba-cache'",
                    },
                    "description": {
                        "type": "string",
                        "description": "one line: the fact and when it applies",
                    },
                    "body": {"type": "string", "description": "the fact in full"},
                },
                ["name", "description", "body"],
            ),
            handler=remember,
            requires_approval=True,
        )
    )
